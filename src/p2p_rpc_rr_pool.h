// Copyright (c) Adithya Kumar, The Pennsylvania State University. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <rte_ring.h>
#include <unordered_map>

#include "p2p_rpc_rr.h"

#ifndef GPU_DISABLED
#include "gdr_mem_manager.h"
#endif

static inline uint8_t* alloc_cpu_mem(size_t mem_size)
{
  void *addr;
  addr = (void*)malloc(mem_size * sizeof(uint8_t));
  return (uint8_t*)addr;
}

static inline void free_cpu_mem(uint8_t*mem)
{
  free(mem);
  return;
}

static inline uint8_t* alloc_mem(size_t mem_size, int device_id)
{
#ifndef GPU_DISABLED
  if(device_id >= 0)
    return alloc_gpu_mem(mem_size, device_id);
  else 
#endif
  if(device_id == -1)
    return alloc_cpu_mem(mem_size);
  return NULL;
}

static inline void free_mem(uint8_t* mem, int device_id)
{
#ifndef GPU_DISABLED
  if(device_id >= 0)
    return free_gpu_mem(mem, device_id);
  else 
#endif
  if(device_id == -1)
    return free_cpu_mem(mem);
  return;
}

static const int RR_POOL_SIZE = MAX_WI_SIZE - 1;

class P2pRpcRRPool
{
  // FIXME: This also has to consider func-id in addition to this
  // Map of in-flight req-tokens to rpc_rr objects
  std::unordered_map<uint64_t, struct p2p_rpc_rr *> rr_cache;
  struct rte_ring *free_rr_pool;
  std::vector<struct p2p_rpc_rr*> rr_pool;
  struct p2p_rpc_rr *proto_rr;

  inline p2p_rpc_rr* 
    alloc_new_rr(p2p_rpc_conn_info *src_conn_info, 
        size_t req_size, size_t resp_size, 
        int req_device_id=-2, int resp_device_id=-2)
  {
    struct p2p_rpc_rr* new_rr = rr_proto_alloc(src_conn_info, req_size, resp_size);
    new_rr->req_payload = alloc_mem(req_size, req_device_id);
    new_rr->req_device_id = req_device_id;
    new_rr->resp_payload = alloc_mem(resp_size, resp_device_id);
    new_rr->resp_device_id = resp_device_id;
    return new_rr;
  }

  public:
  P2pRpcRRPool()
  {
    std::string ring_name = std::string("RR_POOL");
    free_rr_pool = rte_ring_create(ring_name.c_str(), RR_POOL_SIZE + 1,
        rte_socket_id(), RING_F_MP_RTS_ENQ | RING_F_MC_RTS_DEQ);
    if(free_rr_pool == NULL) {
      printf("P2pRpcRRPool Error!: Unable to create POOL\n");
    }
  }

  ~P2pRpcRRPool()
  {
    free_mem(proto_rr->req_payload, proto_rr->req_device_id);
    free_mem(proto_rr->resp_payload, proto_rr->resp_device_id);
    rr_base_free(proto_rr);
    for(auto rr : rr_pool) {
      free_mem(rr->req_payload, rr->req_device_id);
      free_mem(rr->resp_payload, rr->resp_device_id);
      rr_base_free(rr);
    }
  }

  std::vector<struct p2p_rpc_rr*>& get_rr_pool_container()
  {
    return rr_pool;
  }

  p2p_rpc_rr** get_rr_pool()
  {
    return rr_pool.data();
  }

  int get_pool_size()
  {
    return RR_POOL_SIZE;
  }

  void setup_and_init_rr_pool(p2p_rpc_conn_info *src_conn_info, size_t req_size, size_t resp_size, int device_id)
  {
    // Setup a proto-rr for the future
    proto_rr = alloc_new_rr(src_conn_info, req_size, resp_size, device_id, device_id);

    for (int i = 0; i < RR_POOL_SIZE; i++) {
      rr_pool.push_back(alloc_new_rr(src_conn_info, req_size, resp_size, device_id, device_id));
    }

    for(int i = 0 ; i < RR_POOL_SIZE; i++) {
      if(rte_ring_enqueue(free_rr_pool, (void *)rr_pool[i]) != 0) {
        printf("P2pRpcRRPool: Free pool full!\n");
      }
    }
    TRACE_PRINTF("P2pRPCRRPool initialized with %d elements\n", RR_POOL_SIZE);
  }

  // Gets a cached rr or consumes a new one and adds it to the cache
  inline p2p_rpc_rr* get_rr(p2p_rpc_hdr *rpc_hdr) 
  {
    uint64_t req_token = get_req_token(rpc_hdr);
    struct p2p_rpc_rr *rr;
    if(rr_cache.find(req_token) != rr_cache.end()) {
      // req_token is recycled, so clear it out
      // Cache shootdown done here because map is not multi-thread safe
      if(rr_cache[req_token]->rr_state == RR_STATUS::UNUSED) {
        rr_cache.erase(req_token);
      } else {
        return rr_cache[req_token];
      }
    }
    rr = consume_rr();
    set_conn_info(&rr->_client_conn_info, rpc_hdr);
    rr->req_token = req_token;
    rr->rr_state = RR_STATUS::PROCESS;
    rr_cache[req_token] = rr;
    rr = rr_cache[req_token];
    return rr;
  }

  inline struct p2p_rpc_rr* consume_rr()
  {
    struct p2p_rpc_rr *consumed_rr = NULL;
    if(rte_ring_dequeue(free_rr_pool, (void**)&consumed_rr) == -ENOENT) {
      printf("Warning!, P2pRpcRRPool is empty, allocating new rr\n");
      consumed_rr = rr_clone_alloc(proto_rr);
      consumed_rr->req_payload = alloc_mem(consumed_rr->max_req_size, consumed_rr->req_device_id);
      consumed_rr->resp_payload = alloc_mem(consumed_rr->max_resp_size, consumed_rr->resp_device_id);
      rr_pool.push_back(consumed_rr);
    }
    if(unlikely(consumed_rr == NULL)) {
      printf("Out of memory, exiting...");
      exit(1);
    }
    return consumed_rr;
  }

  inline void reap_rr(struct p2p_rpc_rr* rr)
  {
    rr->rr_state = RR_STATUS::UNUSED;
    rr->hdr_bufs->num_items = 0;
    rr->payload_bufs->num_items = 0;
    rr->req_token = 0;
    if(rte_ring_enqueue(free_rr_pool, (void *)rr) == -ENOBUFS) {
      printf("Warning!, P2pRpcRRPool is full, de-allocating rr: %p\n", (void*)rr);
      free_mem(rr->req_payload, rr->req_device_id);
      free_mem(rr->resp_payload, rr->resp_device_id);
      rr_base_free(rr);
    }
  }
};
