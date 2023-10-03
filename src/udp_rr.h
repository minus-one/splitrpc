// Copyright (c) Adithya Kumar, The Pennsylvania State University. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "p2p_bufs.h"
#include "config_utils.h"
#include <queue>
#include "eth_common.h"

#ifndef GPU_DISABLED
#include <cuda.h>
#include <cuda_runtime.h>
#define CUDA_DRIVER_API
#include <helper_functions.h>
#include <helper_cuda.h>
#endif

#include "concurrentqueue.h"

//#include <emmintrin.h>

struct UdpRpcHdr {
  //uint8_t pkt_hdr_t[UDP_HEADER_LEN];
  uint64_t req_token;
  int64_t sock_perf_header;
  uint16_t seq_num;
  uint8_t _pad[64 - UDP_HEADER_LEN - (sizeof(uint64_t) + sizeof(int64_t) + sizeof(uint16_t))];
  //uint16_t func_id;
} __attribute__((packed));

static const size_t UDP_RPC_HEADER_LEN = sizeof(struct UdpRpcHdr);
static const size_t UDP_RPC_HEADER_TAIL_LEN = UDP_RPC_HEADER_LEN;
static const size_t UDP_RPC_MAX_MTU = RPC_MTU + UDP_RPC_HEADER_LEN;

static inline uintptr_t get_req_token(UdpRpcHdr *hdr) {
  return static_cast<uintptr_t>(hdr->req_token);
}

static inline uint16_t get_seq_num(UdpRpcHdr *hdr) {
  return static_cast<uint16_t>(hdr->seq_num);
}

static inline void* get_req_tail(UdpRpcHdr *hdr) {
  return (void*)((uint8_t*)hdr); 
}

enum UDP_RR_STATES { UDP_RR_CREATE, UDP_RR_PROCESS, UDP_RR_READY, UDP_RR_FINISH, UDP_RR_UNUSED};

class UdpRr {
  private:

  public:
    uint64_t req_token;
    size_t rr_idx;

    // Corresponds to g_params
    uint8_t *req_payload, *resp_payload;
    size_t req_size, resp_size;
    int req_device_id, resp_device_id;

    size_t max_req_size, max_resp_size;
    struct p2p_bufs *req_bufs;
    struct p2p_bufs *resp_bufs;

    uint32_t rr_state;
    sockaddr_in si_server;
    sockaddr_in si_other;
    int my_sock;

    UdpRr(size_t _max_req_size = 0, size_t _max_resp_size = 0) {
      req_bufs = new p2p_bufs;
      resp_bufs = new p2p_bufs;

      rr_state = UDP_RR_STATES::UDP_RR_CREATE;

      req_bufs->num_items = 0;
      resp_bufs->num_items = 0;
      max_req_size = _max_req_size;
      max_resp_size = _max_resp_size;
      req_size = 0;
      resp_size = 0;
      req_token = 0;
      rr_idx = 0;
    }

    ~UdpRr() {
      delete req_bufs;
      delete resp_bufs;
    }


    inline __attribute__((always_inline)) void alloc_req_bufs() {
      // Allocate buffers for header and payload
      req_bufs->num_items = (max_req_size + RPC_MTU - 1) / RPC_MTU;
      size_t payload_size = max_req_size;
      size_t curr_pkt_size;
      for(int i = 0 ; i < req_bufs->num_items; i++) {
        curr_pkt_size = (payload_size <= RPC_MTU) ? payload_size : RPC_MTU;
        req_bufs->burst_items[i] = (uintptr_t)(new uint8_t[UDP_RPC_HEADER_LEN + curr_pkt_size]);
        UdpRpcHdr *rpc_hdr = (UdpRpcHdr*)req_bufs->burst_items[i];
        std::memset((void*)rpc_hdr, 0, sizeof(UdpRpcHdr));
        rpc_hdr->req_token = req_token;
        rpc_hdr->seq_num = i;
        req_bufs->item_size[i] = curr_pkt_size + UDP_RPC_HEADER_LEN;
        payload_size -= curr_pkt_size;
      }
    }

    inline __attribute__((always_inline)) void alloc_resp_bufs() {
      // Allocate buffers for header and payload
      resp_bufs->num_items = (max_resp_size + RPC_MTU - 1) / RPC_MTU;
      size_t payload_size = max_resp_size;
      size_t curr_pkt_size;
      for(int i = 0 ; i < resp_bufs->num_items; i++) {
        curr_pkt_size = (payload_size <= RPC_MTU) ? payload_size : RPC_MTU;
        resp_bufs->burst_items[i] = (uintptr_t)(new uint8_t[UDP_RPC_HEADER_LEN + curr_pkt_size]);
        UdpRpcHdr *rpc_hdr = (UdpRpcHdr*)resp_bufs->burst_items[i];
        std::memset((void*)rpc_hdr, 0, sizeof(UdpRpcHdr));
        rpc_hdr->req_token = req_token;
        rpc_hdr->seq_num = i;
        resp_bufs->item_size[i] = curr_pkt_size + UDP_RPC_HEADER_LEN;
        payload_size -= curr_pkt_size;
      }
      resp_size = max_resp_size;
    }

    inline __attribute__((always_inline)) void release_req_bufs() {
      for(int i = 0 ; i < req_bufs->num_items; i++) {
        uint8_t* buf = (uint8_t*)req_bufs->burst_items[i];
        //TRACE_PRINTF("Releasing buf: %p\n", (void*)buf);
        delete buf;
        req_bufs->burst_items[i] = 0;
        req_bufs->item_size[i] = 0;
      }
      req_bufs->num_items = 0;
      req_size = 0;
    }

    inline __attribute__((always_inline)) void release_resp_bufs() {
      for(int i = 0 ; i < resp_bufs->num_items; i++) {
        uint8_t* buf = (uint8_t*)resp_bufs->burst_items[i];
        delete buf;
        resp_bufs->burst_items[i] = 0;
        resp_bufs->item_size[i] = 0;
      }
      resp_bufs->num_items = 0;
      resp_size = 0;
    }

    inline __attribute__((always_inline)) void rr_merge_req_buf(
        uintptr_t req_buf,
        size_t item_size,
        int idx) {
      req_bufs->burst_items[idx] = req_buf;
      req_bufs->item_size[idx] = item_size;
      req_bufs->num_items++;
      req_size += item_size - UDP_RPC_HEADER_LEN;
    }

    inline __attribute__((always_inline)) void rr_merge_resp_buf(
        uintptr_t resp_buf,
        size_t item_size,
        int idx) {
      resp_bufs->burst_items[idx] = resp_buf;
      resp_bufs->item_size[idx] = item_size;
      resp_bufs->num_items++;
      resp_size += item_size - UDP_RPC_HEADER_LEN;
    }

    inline __attribute__((always_inline)) size_t req_to_bufs() {
#ifdef TRACE_MODE
      printf("Copying req to bufs for rr: %p...\n", (void*)this);
      print_bufs(req_bufs);
#endif
      size_t curr_pkt_size = 0;
      size_t byte_offset = 0;

      // Assumes the memory-buffers have been pre-allocated
      // 1. Scatter the payload to the memory-buffers
      for(int i = 0 ; i < req_bufs->num_items; i++) {
        curr_pkt_size = req_bufs->item_size[i] - UDP_RPC_HEADER_LEN; 
        UdpRpcHdr *rpc_hdr = (UdpRpcHdr*)req_bufs->burst_items[i];
        rpc_hdr->req_token = req_token;
        uint8_t *buf = (uint8_t*)req_bufs->burst_items[i] + UDP_RPC_HEADER_LEN;
        memcpy(buf, &req_payload[byte_offset], curr_pkt_size);
        byte_offset += curr_pkt_size;
      }
      if(byte_offset != req_size) {
        printf("Warning, couldn't copy req to bufs, exp: %ld, actual: %ld\n",
            req_size, byte_offset);
      }
      return byte_offset;
    }

    inline __attribute__((always_inline)) size_t bufs_to_req() {
      uint8_t *payload_buf;
      size_t byte_offset = 0;
      for(int i = 0; i < req_bufs->num_items; i++) {
        payload_buf = (uint8_t*)req_bufs->burst_items[i] + UDP_RPC_HEADER_LEN;
        TRACE_PRINTF("Copying buf: %p, payload: %p, into: %p, size: %ld\n", 
            (void*)req_bufs->burst_items[i], (void*)payload_buf, (void*)&req_payload[byte_offset], req_bufs->item_size[i] - UDP_RPC_HEADER_LEN);
        memcpy(&req_payload[byte_offset], payload_buf, req_bufs->item_size[i] - UDP_RPC_HEADER_LEN);
        byte_offset += req_bufs->item_size[i] - UDP_RPC_HEADER_LEN;
      }
      return byte_offset;
    } 

    inline __attribute__((always_inline)) size_t resp_to_bufs() {
      size_t curr_pkt_size = 0;
      size_t byte_offset = 0;

      // Assumes the memory-buffers have been pre-allocated
      // 1. Scatter the payload to the memory-buffers
      for(int i = 0 ; i < resp_bufs->num_items; i++) {
        curr_pkt_size = resp_bufs->item_size[i] - UDP_RPC_HEADER_LEN; 
        UdpRpcHdr *rpc_hdr = (UdpRpcHdr*)resp_bufs->burst_items[i];
        rpc_hdr->req_token = req_token;
        uint8_t *buf = (uint8_t*)resp_bufs->burst_items[i] + UDP_RPC_HEADER_LEN;
        memcpy(buf, &resp_payload[byte_offset], curr_pkt_size);
        byte_offset += curr_pkt_size;
      }

      if(byte_offset != resp_size) {
        printf("Scatter failed did: %ld, exp: %ld\n", byte_offset, resp_size);
      }

      return byte_offset;
    }

    inline __attribute__((always_inline)) size_t bufs_to_resp() {
      uint8_t *payload_buf;
      size_t byte_offset = 0;
      for(int i = 0; i < resp_bufs->num_items; i++) {
        payload_buf = (uint8_t*)resp_bufs->burst_items[i] + UDP_RPC_HEADER_LEN;
        memcpy(&resp_payload[byte_offset], payload_buf, resp_bufs->item_size[i] - UDP_RPC_HEADER_LEN);
        byte_offset += resp_bufs->item_size[i] - UDP_RPC_HEADER_LEN;
      }
      return byte_offset;
    } 

    inline __attribute__((always_inline)) bool is_req_ready() { return req_size == max_req_size; }

    inline __attribute__((always_inline)) bool is_resp_ready() { return resp_size == max_resp_size; }

    void rr_print_req_bufs()
    {
      printf("==========================================================================\n");
      printf("RR: %p, token: %ld, req_bufs: %d\n", 
          (void*)this, req_token, req_bufs->num_items);
      for(int i = 0 ; i < req_bufs->num_items ; i++) {
        void *start_addr = (void*)req_bufs->burst_items[i];
        size_t item_size = req_bufs->item_size[i];
        void *end_addr = (void*)((uint8_t*)start_addr + item_size);
        printf("Requests:- rr: %p, start_addr: %p, end_addr: %p, seq_num: %d\n", 
            (void*)this, start_addr, end_addr, i);
      }
      printf("==========================================================================\n");
    }

    void rr_print_resp_bufs()
    {
      printf("==========================================================================\n");
      printf("RR: %p, token: %ld, resp_bufs: %d\n", 
          (void*)this, req_token, resp_bufs->num_items);
      for(int i = 0 ; i < resp_bufs->num_items ; i++) {
        void *start_addr = (void*)resp_bufs->burst_items[i];
        size_t item_size = resp_bufs->item_size[i];
        void *end_addr = (void*)((uint8_t*)start_addr + item_size);
        printf("response:- rr: %p, start_addr: %p, end_addr: %p, seq_num: %d\n", 
            (void*)this, start_addr, end_addr, i);
      }
      printf("==========================================================================\n");
    }
};

class UdpRrPool 
{
  private:
    std::unordered_map<uint64_t, UdpRr*> rr_cache;
    std::vector<UdpRr*> rr_pool;

    int pool_size;
    size_t req_size, resp_size;

    uint8_t *_req_arena_internal, *_resp_arena_internal;

    int rr_pi_idx;

    //std::queue<UdpRr*> ready_rr_pool;
    //std::queue<UdpRr*> free_rr_pool;

    moodycamel::ConcurrentQueue<UdpRr*> ready_rr_pool;
    moodycamel::ConcurrentQueue<UdpRr*> process_rr_pool;
    moodycamel::ConcurrentQueue<UdpRr*> free_rr_pool;

    //std::mutex consumer_mutex;

  public:

    UdpRrPool(int _pool_size = MAX_WI_SIZE - 1) 
    {
      pool_size = _pool_size;
      rr_pi_idx = 0;
      _req_arena_internal = NULL;
      _resp_arena_internal = NULL;
    }

    ~UdpRrPool()
    {
      printf("Cleaning up UDP RR pool, ready queue has: %ld elements\n", ready_rr_pool.size_approx());
      printf("Cleaning up UDP RR pool, free queue has: %ld elements\n", free_rr_pool.size_approx());
      if(_req_arena_internal)
        delete _req_arena_internal;
      if(_resp_arena_internal)
        delete _resp_arena_internal;
      for(auto rr : rr_pool)
        delete rr;
    }

    UdpRr** get_rr_pool() { return rr_pool.data(); }

    int get_pool_size() { return pool_size;}

    void setup_and_init_rr_pool_with_preallocation(uint8_t* req_arena, size_t _req_size, uint8_t* resp_arena, size_t _resp_size)
    {
      req_size = _req_size;
      resp_size = _resp_size;
      for(int i = 0 ; i < pool_size; i++) {
        UdpRr *new_rr = new UdpRr(req_size, resp_size);
        new_rr->req_payload = (i * req_size) + req_arena;
        new_rr->resp_payload = (i * resp_size) + resp_arena;
        new_rr->rr_state = UDP_RR_STATES::UDP_RR_UNUSED;
        new_rr->rr_idx = i;
        rr_pool.push_back(new_rr);
        //free_rr_pool.push(new_rr);
        free_rr_pool.enqueue(new_rr);
      }
    }

    void setup_and_init_rr_pool(size_t _req_size, size_t _resp_size)
    {
      req_size = _req_size;
      resp_size = _resp_size;
      _req_arena_internal = new uint8_t[req_size * pool_size];
      _resp_arena_internal = new uint8_t[resp_size * pool_size];

      setup_and_init_rr_pool_with_preallocation(_req_arena_internal, req_size, _resp_arena_internal, resp_size);
    }

    inline UdpRr* get_rr(UdpRpcHdr *rpc_hdr)
    {
      uint64_t req_token = get_req_token(rpc_hdr);

      // Search cache
      if(rr_cache.find(req_token) != rr_cache.end()) {
        return rr_cache[req_token];
      }

      // Create(consume) new
      UdpRr *new_rr = consume_rr();
      // Update conn-info
      new_rr->req_token = req_token;
      new_rr->rr_state = UDP_RR_STATES::UDP_RR_PROCESS;
      rr_cache[req_token] = new_rr;
      new_rr = rr_cache[req_token];

      TRACE_PRINTF("Consuming new_rr: %p, %ld\n", (void*)new_rr, new_rr->rr_idx);
      return new_rr;
    }

    inline UdpRr* consume_rr()
    {
      UdpRr* consumed_rr = NULL;
      while(!free_rr_pool.try_dequeue(consumed_rr))
        ;
      return consumed_rr;

      //const std::lock_guard<std::mutex> lock(consumer_mutex);
      //{
      //  while(rr_pool[rr_pi_idx]->rr_state != UDP_RR_STATES::UDP_RR_UNUSED)
      //    rr_pi_idx = (rr_pi_idx + 1) % pool_size;
      //  consumed_rr = rr_pool[rr_pi_idx];
      //  consumed_rr->rr_state = UDP_RR_STATES::UDP_RR_PROCESS;
      //  rr_pi_idx = (rr_pi_idx + 1) % pool_size;
      //}
      //return consumed_rr;
    }

    inline void reap_rr(UdpRr *reaped_rr)
    {
      reaped_rr->req_token = 0;
      reaped_rr->rr_state = UDP_RR_STATES::UDP_RR_UNUSED;
      free_rr_pool.enqueue(reaped_rr);
      //const std::lock_guard<std::mutex> lock(consumer_mutex);
      //{
      //  reaped_rr->req_token = 0;
      //  reaped_rr->rr_state = UDP_RR_STATES::UDP_RR_UNUSED;
      //}
    }

    inline void mark_process_rr(UdpRr *rr)
    {
      rr_cache.erase(rr->req_token);
      rr->rr_state = UDP_RR_STATES::UDP_RR_READY;
      process_rr_pool.enqueue(rr);
    }

    inline UdpRr* consume_process_rr()
    {
      UdpRr *consumed_rr = NULL;
      if(process_rr_pool.try_dequeue(consumed_rr))
        return consumed_rr;
      return NULL;
    }

    inline void mark_ready_rr(UdpRr *rr)
    {
      //rr_cache.erase(rr->req_token);
      //rr->rr_state = UDP_RR_STATES::UDP_RR_READY;
      ready_rr_pool.enqueue(rr);
      //ready_rr_pool.push(rr);
    }

    inline UdpRr* consume_ready_rr()
    {
      UdpRr *consumed_rr = NULL;
      if(ready_rr_pool.try_dequeue(consumed_rr))
        return consumed_rr;
      return NULL;

      //if(ready_rr_pool.empty())
      //  return NULL;
      //consumed_rr = ready_rr_pool.front();
      //ready_rr_pool.pop();
      //return consumed_rr;
    }

    inline __attribute__((always_inline)) int get_next_rx_ready_rr(UdpRr **next_rr) 
    {
      *next_rr = consume_ready_rr();
      if(*next_rr != NULL)
        return 1;
      return 0;
    }
};
