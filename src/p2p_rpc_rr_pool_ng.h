// Copyright (c) Adithya Kumar, The Pennsylvania State University. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <rte_ring.h>
#include <unordered_map>

#include "p2p_rpc_rr_ng.h"

//static const int RR_POOL_SIZE = MAX_WI_SIZE - 1;
static const int RR_POOL_SIZE = 4096 - 1;

class P2pRpcRRPool
{
  // FIXME: This also has to consider func-id in addition to this
  // Map of in-flight req-tokens to rpc_rr objects
  std::unordered_map<uint64_t, P2pRpcRr *> rr_cache;
  struct rte_ring *free_rr_pool;
  struct rte_ring *rx_ready_rr_pool;
  struct rte_ring *tx_ready_rr_pool;
  std::vector<P2pRpcRr*> rr_pool;
  P2pRpcRr *proto_rr;
  int pool_size;

  public:
  P2pRpcRRPool(int _pool_size = -1)
  {
    if(_pool_size == -1)
      pool_size = RR_POOL_SIZE;
    else
      pool_size = _pool_size - 1;
    std::string free_ring_name = std::string("FREE_RR_POOL");
    free_rr_pool = rte_ring_create(free_ring_name.c_str(), pool_size + 1,
        rte_socket_id(), RING_F_MP_RTS_ENQ | RING_F_MC_RTS_DEQ);
    if(free_rr_pool == NULL) {
      printf("P2pRpcRRPool Error!: Unable to create POOL\n");
    }

    std::string rx_ready_ring_name = std::string("RX_READY_RR_POOL");
    rx_ready_rr_pool = rte_ring_create(rx_ready_ring_name.c_str(), pool_size + 1,
        rte_socket_id(), RING_F_MP_RTS_ENQ | RING_F_MC_RTS_DEQ);
    if(rx_ready_rr_pool == NULL) {
      printf("P2pRpcRRPool Error!: Unable to create POOL\n");
    }

    std::string tx_ready_ring_name = std::string("TX_READY_RR_POOL");
    tx_ready_rr_pool = rte_ring_create(tx_ready_ring_name.c_str(), pool_size + 1,
        rte_socket_id(), RING_F_MP_RTS_ENQ | RING_F_MC_RTS_DEQ);
    if(tx_ready_rr_pool == NULL) {
      printf("P2pRpcRRPool Error!: Unable to create POOL\n");
    }
  }

  ~P2pRpcRRPool()
  {
    printf("Cleaning up RR Pool\n");
    printf("Cache has %ld elements\n", rr_cache.size());
    for(auto e : rr_cache) {
      if(e.second)
        printf("cache_token: (%ld, %p), rpc_rr: %p, rpc_rr_token: (%ld, %p), payload_size: %ld, num-items: %d\n", 
            e.first, (void*)e.first, (void*)e.second, e.second->req_token, (void*)e.second->req_token,
            e.second->payload_size, e.second->payload_bufs->num_items);
    }

    delete proto_rr;
    for(auto rr : rr_pool) {
      delete rr;
    }
  }

  std::vector<P2pRpcRr*>& get_rr_pool_container() { return rr_pool; }

  P2pRpcRr** get_rr_pool() { return rr_pool.data(); }

  int get_pool_size() { return pool_size; }

  void setup_and_init_rr_pool(p2p_rpc_conn_info *template_conn_info, size_t payload_size)
  {
    // Setup a proto-rr for the future
    proto_rr = new P2pRpcRr(template_conn_info, payload_size);

    for (int i = 0; i < pool_size; i++) {
      P2pRpcRr *new_rr = rr_clone_alloc(proto_rr); 
      rr_pool.push_back(new_rr);
    }
    
    for(int i = 0 ; i < pool_size; i++) {
      if(rte_ring_enqueue(free_rr_pool, (void *)rr_pool[i]) != 0) {
        printf("P2pRpcRRPool: Free pool full!\n");
      }
    }
    TRACE_PRINTF("P2pRPCRRPool initialized with %d elements\n", pool_size);
  }

  // Gets a cached rr or consumes a new one and adds it to the cache
  inline __attribute__((always_inline)) P2pRpcRr* get_rr(p2p_rpc_hdr *rpc_hdr) 
  {
    uint64_t req_token = get_req_token(rpc_hdr);
    P2pRpcRr *rr = NULL;
    if(rr_cache.find(req_token) != rr_cache.end()) {
      return rr_cache[req_token];
    }
    rr = consume_rr();
    set_conn_info(&rr->_client_conn_info, rpc_hdr);
    rr->req_token = req_token;
    rr_cache[req_token] = rr;
    return rr;
  }

  inline __attribute__((always_inline)) void append_to_rr(p2p_rpc_hdr *rpc_hdr, uintptr_t transport_mbuf, 
      uintptr_t hdr_buf, uintptr_t payload_buf, size_t payload_item_size)
  {
    P2pRpcRr *call_id = get_rr(rpc_hdr);
    call_id->rr_emplace_mbuf(transport_mbuf, hdr_buf, payload_buf, payload_item_size, rpc_hdr->seq_num);
    if(call_id->is_rr_ready()) {
      mark_rx_ready_rr(call_id);
    }
  }

  inline __attribute__((always_inline)) P2pRpcRr* consume_rr()
  {
    P2pRpcRr *consumed_rr = NULL;
    if(rte_ring_dequeue(free_rr_pool, (void**)&consumed_rr) == -ENOENT) {
      printf("Warning!, P2pRpcRRPool is empty, allocating new rr\n");
      consumed_rr = rr_clone_alloc(proto_rr);
      rr_pool.push_back(consumed_rr);
    }
    if(unlikely(consumed_rr == NULL)) {
      printf("Out of memory, exiting...");
      exit(1);
    }
    consumed_rr->rr_state = P2P_RPC_RR_STATES::RR_PROCESS;
    return consumed_rr;
  }

  inline __attribute__((always_inline)) void reap_rr(P2pRpcRr* rr)
  {
    rr->rr_state = P2P_RPC_RR_STATES::RR_UNUSED;
    rr->hdr_bufs->num_items = 0;
    rr->payload_bufs->num_items = 0;
    rr->req_token = 0;
    if(rte_ring_enqueue(free_rr_pool, (void *)rr) == -ENOBUFS) {
      printf("Warning!, P2pRpcRRPool is full, de-allocating rr: %p\n", (void*)rr);
      delete rr;
    }
  }

  inline __attribute__((always_inline)) void mark_rx_ready_rr(P2pRpcRr *rr)
  {
    rr_cache.erase(rr->req_token);
    rr->rr_state = P2P_RPC_RR_STATES::RR_READY;
    while(rte_ring_enqueue(rx_ready_rr_pool, (void*)rr) != 0)
      ;
  }

  inline __attribute__((always_inline)) void mark_tx_ready_rr(P2pRpcRr *rr)
  {
    while(rte_ring_enqueue(tx_ready_rr_pool, (void*)rr) != 0)
      ;
  }

  inline __attribute__((always_inline)) P2pRpcRr* consume_rx_ready_rr()
  {
    P2pRpcRr *consumed_rr = NULL;
    if(rte_ring_dequeue(rx_ready_rr_pool, (void**)&consumed_rr) == -ENOENT) {
      return NULL; 
    }
    return consumed_rr;
  }

  inline __attribute__((always_inline)) int get_next_rx_ready_rr(P2pRpcRr **next_rr) 
  {
    *next_rr = consume_rx_ready_rr();
    if(*next_rr != NULL)
      return 1;
    return 0;
  }

  inline __attribute__((always_inline)) P2pRpcRr* consume_tx_ready_rr()
  {
    P2pRpcRr *consumed_rr = NULL;
    if(rte_ring_dequeue(tx_ready_rr_pool, (void**)&consumed_rr) == -ENOENT) {
      return NULL; 
    }
    return consumed_rr;
  }
};
