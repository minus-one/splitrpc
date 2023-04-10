// Copyright (c) Adithya Kumar, The Pennsylvania State University. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include "p2p_rpc.h"
#include "p2p_rpc_rr.h"
#include "config_utils.h"
#include "time_utils.h"

#include "p2p_rpc_app_stub.h"

#ifndef GPU_DISABLED
#include "p2p_buf_pool.h"
#include "transport/g_copy_ng.cuh"
#endif

// Gathers a bunch of buf_ptrs into an app_buf
// Assumes that buf_ptrs are arranged in-order and are sized correctly
inline __attribute__((always_inline)) static void
gather_skbs(p2p_sk_buf *skb, p2p_bufs *buf_ptrs, uint8_t *app_buf)
{
  uintptr_t app_addr = (uintptr_t)(app_buf);
  skb->num_items = 0;
  skb->i_buf[skb->num_items] = buf_ptrs->burst_items[0];
  skb->o_buf[skb->num_items] = app_addr;
  skb->len[skb->num_items] = buf_ptrs->item_size[0];
  app_addr += buf_ptrs->item_size[0]; 
  uintptr_t last_end_ptr = buf_ptrs->burst_items[0] + buf_ptrs->item_size[0];

  for(int i = 1 ; i < buf_ptrs->num_items ; i++) {
    if(buf_ptrs->burst_items[i] == last_end_ptr) {
      // extend
      skb->len[skb->num_items] += buf_ptrs->item_size[i];
    } else {
      // stop and start new
      skb->num_items++;
      skb->i_buf[skb->num_items] = buf_ptrs->burst_items[i];
      skb->o_buf[skb->num_items] = app_addr; 
      skb->len[skb->num_items] = buf_ptrs->item_size[i];
    }
    last_end_ptr = buf_ptrs->burst_items[i] + buf_ptrs->item_size[i];
    app_addr += buf_ptrs->item_size[i];
  }
  skb->num_items++;
}

// Given an app_buf, it generates skbs to write into a bunch of mbufs
// Assumes that mbufs are in-order and sized correctly
inline __attribute__((always_inline)) static void
scatter_skbs(p2p_sk_buf *skb, p2p_bufs *buf_ptrs, uint8_t *app_buf)
{
  uintptr_t app_addr = (uintptr_t)(app_buf);
  skb->num_items = 0;
  skb->i_buf[skb->num_items] = app_addr; 
  skb->o_buf[skb->num_items] = buf_ptrs->burst_items[0];
  skb->len[skb->num_items] = buf_ptrs->item_size[0];
  app_addr += buf_ptrs->item_size[0];
  uintptr_t last_end_ptr = buf_ptrs->burst_items[0] + buf_ptrs->item_size[0];

  for(int i = 1 ; i < buf_ptrs->num_items ; i++) {
    if(buf_ptrs->burst_items[i] == last_end_ptr) {
      skb->len[skb->num_items] += buf_ptrs->item_size[i];
    } else {
      skb->num_items++;
      skb->i_buf[skb->num_items] = app_addr; 
      skb->o_buf[skb->num_items] = buf_ptrs->burst_items[i];
      skb->len[skb->num_items] = buf_ptrs->item_size[i];
    }
    last_end_ptr = buf_ptrs->burst_items[i] + buf_ptrs->item_size[i];
    app_addr += buf_ptrs->item_size[i];
  }
  skb->num_items++;
}

class P2pRpcSgEngine
{
  public:
    // Number of parallel instances of the copy engine
    int num_instances;
    int device_id;
    // Maximum queue length for each instance
    uint16_t app_queue_length;
#ifdef PROFILE_MODE
    uint64_t SStartNs, GStartNs;
    std::vector<uint64_t> SDelay, GDelay;
    std::vector<uint64_t> numSkbs;
#endif
    virtual ~P2pRpcSgEngine() {
      PROF_PRINT("Gather", GDelay);
      PROF_PRINT("Scatter", SDelay);
      PROF_PRINT("Gather-NumSKBS", numSkbs);
    }

    // API v0
    virtual void gather(P2pRpcRr *, uint8_t*) = 0;
    virtual void scatter(P2pRpcRr *, uint8_t*) = 0;

    // API v1
    virtual void gather(P2pRpcRr *, g_params*) = 0;
    virtual void scatter(P2pRpcRr *, g_params*) = 0;

    //// API v2
    //virtual void gather(P2pRpcRr *, g_params_v2*) = 0;
    //virtual void scatter(P2pRpcRr *, g_params_v2*) = 0;
};

// A zero-copy engine simply copies the pointer to the
// buffer, and not copy the data.
class P2pRpcSgZcEngine : public P2pRpcSgEngine
{
  public:
    P2pRpcSgZcEngine() {}
    ~P2pRpcSgZcEngine() {}

    inline void gather(P2pRpcRr *rpc_rr, g_params *app_stub)
    {
#ifdef PROFILE_MODE
      GStartNs = getCurNs();
#endif
      app_stub->req = (uint8_t*)rpc_rr->payload_bufs->burst_items[0];  
      app_stub->resp = (uint8_t*)rpc_rr->payload_bufs->burst_items[0];
      std::atomic_thread_fence(std::memory_order_release);
      //_mm_mfence();
#ifdef PROFILE_MODE
      GDelay.push_back(getCurNs() - GStartNs);
#endif
    }

    inline void scatter(P2pRpcRr *, g_params *) {}

    inline void gather(P2pRpcRr *, uint8_t *) {}

    inline void scatter(P2pRpcRr *, uint8_t *) {}
};

class P2pRpcSgCpuEngine : public P2pRpcSgEngine
{
  private:
    inline int sg_on_cpu(p2p_sk_buf *skb)
    {
      for(size_t i = 0 ; i < skb->num_items; i++) {
        memcpy((void*)skb->o_buf[i], (void*)skb->i_buf[i], skb->len[i]);
      }
      return 1;
    }
  public:
    P2pRpcSgCpuEngine() {}
    ~P2pRpcSgCpuEngine() {}

    inline void gather(P2pRpcRr *rpc_rr, g_params *app_stub)
    {
      TRACE_PRINTF("P2pRpcSgCpuEngine GATHER req_token: %ld, rpc_rr: %p, app_stub: %p, req: %p, resp: %p\n", 
          rpc_rr->req_token, (void*)rpc_rr, (void*)app_stub, (void*)app_stub->req, (void*)app_stub->resp);
      gather_skbs(rpc_rr->payload_sk_bufs, rpc_rr->payload_bufs, app_stub->req);
      sg_on_cpu(rpc_rr->payload_sk_bufs);
    }

    inline void scatter(P2pRpcRr *rpc_rr, g_params *app_stub)
    {
      TRACE_PRINTF("P2pRpcSgCpuEngine SCATTER req_token: %ld, rpc_rr: %p, app_stub: %p, req: %p, resp: %p\n", 
          rpc_rr->req_token, (void*)rpc_rr, (void*)app_stub, (void*)app_stub->req, (void*)app_stub->resp);
      scatter_skbs(rpc_rr->payload_sk_bufs, rpc_rr->payload_bufs, app_stub->resp);
      sg_on_cpu(rpc_rr->payload_sk_bufs);
    }

    inline void gather(P2pRpcRr *rpc_rr, uint8_t *payload)
    {
      TRACE_PRINTF("P2pRpcSgCpuEngine GATHER req_token: %ld, rpc_rr: %p, payload: %p\n", 
          rpc_rr->req_token, (void*)rpc_rr, payload);
      gather_skbs(rpc_rr->payload_sk_bufs, rpc_rr->payload_bufs, payload);
      sg_on_cpu(rpc_rr->payload_sk_bufs);
    }

    inline void scatter(P2pRpcRr *rpc_rr, uint8_t *payload)
    {
      TRACE_PRINTF("P2pRpcAppServer SCATTER req_token: %ld, rpc_rr: %p, payload: %p\n", 
          rpc_rr->req_token, (void*)rpc_rr, payload);
      scatter_skbs(rpc_rr->payload_sk_bufs, rpc_rr->payload_bufs, payload);
      sg_on_cpu(rpc_rr->payload_sk_bufs);
    }
};

#ifndef GPU_DISABLED

class P2pRpcSgGpuEngine : public P2pRpcSgEngine
{
  private:
    g_copy_params *h_stubs;
    g_copy_params *d_stubs;
    uint32_t *door_bells;
    uint32_t *d_door_bells;

    P2pRpcSgGpuEngine() {}
  public:
    // Shared across instances 
    CopyCtx *gather_ctx, *scatter_ctx;

    inline void gather(P2pRpcRr *rpc_rr, g_params *app_stub)
    {
#ifdef PROFILE_MODE
      GStartNs = getCurNs();
#endif
      TRACE_PRINTF("P2pRpcSgGpuEngine GATHER req_token: %ld, rpc_rr: %p, app_stub: %p, req: %p, resp: %p\n", 
        rpc_rr->req_token, (void*)rpc_rr, (void*)app_stub, (void*)app_stub->req, (void*)app_stub->resp);
      gather_skbs(rpc_rr->payload_sk_bufs, rpc_rr->payload_bufs, app_stub->req);
      //sg_on_gpu(gather_ctx, rpc_rr->payload_sk_bufs); 
      std::memcpy((void*)(gather_ctx->h_stub), rpc_rr->payload_sk_bufs, sizeof(p2p_sk_buf));
      _mm_mfence();
      ACCESS_ONCE(*(gather_ctx->door_bell)) = 1;
      _mm_mfence();
      while (*ACCESS_ONCE(gather_ctx->door_bell) != 2)
        ;
#ifdef PROFILE_MODE
      GDelay.push_back(getCurNs() - GStartNs);
      numSkbs.push_back(rpc_rr->payload_sk_bufs->num_items);
#endif
      //gather_ctx->WaitForAppRunComplete();
    }

    inline void scatter(P2pRpcRr *rpc_rr, g_params *app_stub)
    {
#ifdef PROFILE_MODE
      SStartNs = getCurNs();
#endif
      TRACE_PRINTF("P2pRpcSgGpuEngine SCATTER req_token: %ld, rpc_rr: %p, app_stub: %p, req: %p, resp: %p\n", 
        rpc_rr->req_token, (void*)rpc_rr, (void*)app_stub, (void*)app_stub->req, (void*)app_stub->resp);
      scatter_skbs(rpc_rr->payload_sk_bufs, rpc_rr->payload_bufs, app_stub->resp);
      //sg_on_gpu(scatter_ctx, rpc_rr->payload_sk_bufs);
      std::memcpy((void*)(scatter_ctx->h_stub), rpc_rr->payload_sk_bufs, sizeof(p2p_sk_buf));
      _mm_mfence();
      ACCESS_ONCE(*(scatter_ctx->door_bell)) = 1;
      _mm_mfence();
      while (*ACCESS_ONCE(scatter_ctx->door_bell) != 2)
        ;
#ifdef PROFILE_MODE
      SDelay.push_back(getCurNs() - SStartNs);
#endif
      //scatter_ctx->WaitForAppRunComplete();
    }

    /***************** DEPRECATED *********************/
    inline void gather(P2pRpcRr *rpc_rr, uint8_t *payload)
    {
      TRACE_PRINTF("P2pRpcAppServer GATHER req_token: %ld, rpc_rr: %p, payload: %p\n", 
          rpc_rr->req_token, (void*)rpc_rr, payload);
      gather_skbs(rpc_rr->payload_sk_bufs, rpc_rr->payload_bufs, payload);
      sg_on_gpu(gather_ctx, rpc_rr->payload_sk_bufs); 
      gather_ctx->WaitForAppRunComplete();
    }

    inline void scatter(P2pRpcRr *rpc_rr, uint8_t *payload)
    {
      TRACE_PRINTF("P2pRpcSgGpuEngine SCATTER req_token: %ld, rpc_rr: %p, payload: %p\n", 
          rpc_rr->req_token, (void*)rpc_rr, payload);
      scatter_skbs(rpc_rr->payload_sk_bufs, rpc_rr->payload_bufs, payload);
      sg_on_gpu(scatter_ctx, rpc_rr->payload_sk_bufs);
      scatter_ctx->WaitForAppRunComplete();
    }
    /***************** END OF DEPRECATED *********************/

    P2pRpcSgGpuEngine(int _device_id)
    {
      device_id = _device_id;
      int srv_qlen = 1;

      TRACE_PRINTF("Setting up P2pRpcSgGpuEngine, device_id: %d\n", device_id);
      h_stubs = BufItemPool<g_copy_params>::create_buf_item_pool(2 * srv_qlen, device_id);
      d_stubs = BufItemPool<g_copy_params>::get_dev_ptr(h_stubs);
      door_bells = BufItemPool<uint32_t>::create_buf_item_pool(2 * srv_qlen, device_id);
      d_door_bells = BufItemPool<uint32_t>::get_dev_ptr(door_bells);


      gather_ctx = new CopyCtx;
      gather_ctx->launch_type = get_gpu_copy_type();
      gather_ctx->device_id = device_id;
      checkCudaErrors(cudaSetDevice(device_id));
      checkCudaErrors(cudaStreamCreateWithFlags(&gather_ctx->work_stream, cudaStreamNonBlocking));
      checkCudaErrors(cudaEventCreateWithFlags(&gather_ctx->work_complete, cudaEventDisableTiming));
      gather_ctx->h_stub = &h_stubs[0];
      gather_ctx->d_stub = &d_stubs[0];
      gather_ctx->door_bell = &door_bells[0];
      gather_ctx->d_door_bell = &d_door_bells[0];
      if(sg_on_gpu_entry(gather_ctx) != 1) {
        printf("Failed to start the GATHER kernel\n");
      }

      scatter_ctx = new CopyCtx;
      scatter_ctx->launch_type = get_gpu_copy_type();
      scatter_ctx->device_id = device_id;
      checkCudaErrors(cudaStreamCreateWithFlags(&scatter_ctx->work_stream, cudaStreamNonBlocking));
      checkCudaErrors(cudaEventCreateWithFlags(&scatter_ctx->work_complete, cudaEventDisableTiming));
      scatter_ctx->h_stub = &h_stubs[srv_qlen];
      scatter_ctx->d_stub = &d_stubs[srv_qlen];
      scatter_ctx->door_bell = &door_bells[srv_qlen];
      scatter_ctx->d_door_bell = &d_door_bells[srv_qlen];
      if(sg_on_gpu_entry(scatter_ctx) != 1) {
        printf("Failed to start the SCATTER kernel\n");
      }
    }

    ~P2pRpcSgGpuEngine()
    {
      *ACCESS_ONCE(gather_ctx->door_bell) = 3;
      *ACCESS_ONCE(scatter_ctx->door_bell) = 3;
      BufItemPool<g_copy_params>::delete_buf_item_pool(h_stubs, device_id);
      BufItemPool<uint32_t>::delete_buf_item_pool(door_bells, device_id);
    }
};

#endif

/************************** COPY APIs implemented in CPUs (DEPRECATED) ***************/
// Assumes mbufs have been allocated
// Returns number of bytes copied
inline int 
scatter_payload_on_cpu(struct p2p_bufs *payload_bufs, 
    uint8_t *payload, 
    size_t payload_size)
{
  uint8_t *payload_buf;
  size_t curr_pkt_size = 0;
  int byte_offset = 0;

  for(int i = 0; i < payload_bufs->num_items; i++) {
    curr_pkt_size = (payload_size <= RPC_MTU) ? payload_size : RPC_MTU;
    payload_size -= curr_pkt_size;

    // Scatter payload into payload_bufs
    payload_buf = (uint8_t*)payload_bufs->burst_items[i];
    memcpy(payload_buf, 
        &(payload[byte_offset]), 
        curr_pkt_size);
    payload_bufs->item_size[i] = curr_pkt_size;
    byte_offset += curr_pkt_size;
  }

  return byte_offset;
}

/*
 * Gathers a bunch of payload_bufs into a single payload
 * Takes offsets based on item_size in the payload_bufs
 * Returns the number of bytes copied.
 */
inline int 
gather_payload_on_cpu(struct p2p_bufs *payload_bufs,
    uint8_t *payload)
{
  uint8_t *payload_buf;
  int byte_offset = 0;

  for(int i = 0 ; i < payload_bufs->num_items; i++) {
    payload_buf = (uint8_t*)payload_bufs->burst_items[i];
    memcpy(&payload[byte_offset],
        payload_buf,
        payload_bufs->item_size[i]);
    byte_offset += payload_bufs->item_size[i];
  }

  return byte_offset;
}
/**********************************************************************************************/
