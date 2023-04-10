// Copyright (c) Adithya Kumar, The Pennsylvania State University. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include <queue>

#include "p2p_rpc_rr.h"
#include "p2p_rpc.h"
#include "config_utils.h"
#include "p2p_buf_pool.h"

//#include "transport/g_copy.cuh"

#include <cuda_runtime.h>
#include <helper_functions.h>
#include <helper_cuda.h>
#include <emmintrin.h>
#include <rte_ring.h>

#include "time_utils.h"
#include "stats_utils.h"

#include "p2p_rpc_app_ctx.h"

using AppInitCB = int (*)(AppCtx *);
using AppRunCB = int (*)(AppCtx *);
using AppCleanupCB = int(*)(AppCtx *);
using AppRunAsyncCB = int (*)(AppCtx*);
using AppRunWaitCB = int (*)(AppCtx *);

class P2pRpcAppServer
{
  AppInitCB app_init;
  AppRunCB app_run;
  AppCleanupCB app_cleanup;

  // Number of parallel instances of the server
  int num_instances;
  int device_id;

  // Maximum queue length for each instance
  uint16_t srv_qlen;

#ifdef PROFILE_MODE
  uint64_t dw_startNs;
  std::vector<uint64_t> DwDelay;
#endif

  // Shared across instances 
  std::vector<AppCtx> app_ctx;
  //std::vector<CopyCtx*> copy_ctx;
  // Current arch is 1 completion queue and 1 submission queues
  // Submission queue will be owned by the server
  // Completion queue will be owned by the transport entity
  struct rte_ring *sub_queue;
  struct rte_ring *completion_queue;

  uint32_t num_req;
public:
  uint32_t AppRx_payload_size, AppTx_payload_size;

  //inline void gather(struct p2p_rpc_rr *req, int instance = 0)
  //{
  //  TRACE_PRINTF("P2pRpcAppServer GATHER req_token: %ld, req: %p, req_payload: %p, resp_payload: %p\n", 
  //      req->req_token, (void*)req, (void*)req->req_payload, (void*)req->resp_payload);
  //  gather_skbs(req->payload_sk_bufs, req->payload_bufs, req->req_payload);
  //  //print_skb(req->payload_sk_bufs);
  //  sg_on_gpu(copy_ctx[instance], req->payload_sk_bufs, 0); 
  //  if (copy_ctx[instance]->launch_type == 1 || copy_ctx[instance]->launch_type == 2 || copy_ctx[instance]->launch_type == 4) {
  //    checkCudaErrors(cudaStreamSynchronize(copy_ctx[instance]->work_stream));
  //  }
  //  else {
  //    while (*ACCESS_ONCE(copy_ctx[instance]->door_bell) != 2)
  //      ;
  //  }
  //}

  //inline void scatter(struct p2p_rpc_rr *req, int instance = 0)
  //{
  //  TRACE_PRINTF("P2pRpcAppServer SCATTER req_token: %ld, req: %p, req_payload: %p, resp_payload: %p\n", 
  //      req->req_token, (void*)req, (void*)req->req_payload, (void*)req->resp_payload);
  //  scatter_skbs(req->payload_sk_bufs, req->payload_bufs, req->resp_payload);
  //  sg_on_gpu(copy_ctx[instance], req->payload_sk_bufs, 1);
  //  if (copy_ctx[instance]->launch_type == 1 || copy_ctx[instance]->launch_type == 2 || copy_ctx[instance]->launch_type == 4) {
  //    checkCudaErrors(cudaStreamSynchronize(copy_ctx[instance]->work_stream));
  //  }
  //  else {
  //    while (*ACCESS_ONCE(copy_ctx[instance]->door_bell) != 2)
  //      ;
  //  }
  //}

  inline void sync(int instance = 0)
  {
    if (app_ctx[instance].launch_type == 1 || app_ctx[instance].launch_type == 2 || app_ctx[instance].launch_type == 4) {
      checkCudaErrors(cudaStreamSynchronize(app_ctx[instance].work_stream));
    }
    else {
      while (*ACCESS_ONCE(app_ctx[instance].door_bell) != 2)
        ;
    }
  }

  //inline void copy_and_do_work(struct p2p_rpc_rr *req, int instance = 0)
  //{
  //  gather(req, instance);
  //  do_work(req, instance);
  //  scatter(req, instance);

  //  // Wait till everything is done
  //  sync(instance);
  //}

  inline void do_work(struct p2p_rpc_rr *req, int instance = 0)
  {
    app_ctx[instance].curr_rr = req;
    app_ctx[instance].h_stub->req = req->req_payload;
    app_ctx[instance].h_stub->resp = req->resp_payload;
    _mm_mfence();
    TRACE_PRINTF("P2pRpcAppServer do_work instance: %d, req: %p, req_payload: %p, resp_payload: %p\n", 
        instance, (void*)app_ctx[instance].curr_rr, app_ctx[instance].h_stub->req, app_ctx[instance].h_stub->resp);

    // Do work(req, resp)
    app_run(&app_ctx[instance]);
  }

  inline void do_work_sync(struct p2p_rpc_rr *req, int instance = 0)
  {
    do_work(req, instance);
    sync(instance);
  }

  // Call register_cq before enqueueing work
  void enqueue_work(struct p2p_rpc_rr *req)
  {
    while (rte_ring_sp_enqueue(sub_queue, (void *)req) != 0)
      ;
  }

  // Registers the queue to be notified (typically the IO queue) when work is complete
  void register_cq(struct rte_ring *ring_cq)
  {
    printf("P2pRpcAppServer, registering completion queue: %p\n", (void*)ring_cq);
    completion_queue = ring_cq;
  }

  // Worker function
  void HandleRpcs(volatile bool &force_quit)
  {
    struct p2p_rpc_rr *new_rr;
    int instance = 0;
    while (ACCESS_ONCE(force_quit) == 0)
    {
      // Dequeue req from ringbuffer
      while (rte_ring_sc_dequeue(sub_queue, (void **)&new_rr) != 0 && ACCESS_ONCE(force_quit) == 0)
        ;
      if (unlikely(ACCESS_ONCE(force_quit) != 0))
        break;
      TRACE_PRINTF("DO_WORK: req: %p\n", (void*)new_rr);
      //copy_and_do_work(new_rr, instance);
#ifdef PROFILE_MODE
  dw_startNs = getCurNs();
#endif
      do_work(new_rr, instance);
      sync(instance);
#ifdef PROFILE_MODE
  DwDelay.push_back(getCurNs() - dw_startNs);
#endif
      num_req++;

      // Round-Robin policy
      //instance = (instance + 1) % num_instances;
      TRACE_PRINTF("DO_WORK Complete, enqueueing into cq new_rr: %p\n", (void*)new_rr);
      // enqueue req backinto completion ringbuffer
      while (rte_ring_sp_enqueue(completion_queue, (void *)new_rr) != 0 && ACCESS_ONCE(force_quit) == 0)
        ;
      if (unlikely(ACCESS_ONCE(force_quit) != 0))
        break;
    }
    printf("P2pRpcAppServer, Reqs: %d serviced, Terminating rpc handler for instance %d\n", num_req, instance);
  }

  P2pRpcAppServer(AppInitCB _app_init,
                  AppRunCB _app_run,
                  AppCleanupCB _app_cleanup,
                  int _num_instances = 1,
                  int _device_id = get_cuda_device_id())
  {
    AppRx_payload_size = get_req_size();
    AppTx_payload_size = get_resp_size();
    app_init = _app_init;
    app_run = _app_run;
    app_cleanup = _app_cleanup;
    num_instances = _num_instances;
    device_id = _device_id;
    srv_qlen = 2 * MAX_WI_SIZE;
    num_req = 0;

    TRACE_PRINTF("Setting up P2pRpcAppServer, num_instances: %d, device_id: %d, queue_length: %d\n", num_instances, device_id, srv_qlen);

    // This is only needed in case we use a persistent thread and use
    // GDR mem to communicate with them. Else these can just be on host-memory
    // The g_params stub is used to communicate with the GPU App on the memory location for the request/response
    g_params *h_stubs = BufItemPool<g_params>::create_buf_item_pool(num_instances, device_id);
    g_params *d_stubs = BufItemPool<g_params>::get_dev_ptr(h_stubs);
    uint32_t *door_bells = BufItemPool<uint32_t>::create_buf_item_pool(num_instances, device_id);
    uint32_t *d_door_bells = BufItemPool<uint32_t>::get_dev_ptr(door_bells);

    for (int i = 0; i < num_instances; i++)
    {
      AppCtx new_app_ctx;
      new_app_ctx.launch_type = get_work_launch_type();
      new_app_ctx.device_id = device_id;
      checkCudaErrors(cudaStreamCreateWithFlags(&new_app_ctx.work_stream, cudaStreamNonBlocking));
      checkCudaErrors(cudaEventCreateWithFlags(&new_app_ctx.work_complete, cudaEventDisableTiming));

      new_app_ctx.h_stub = &h_stubs[i];
      new_app_ctx.d_stub = &d_stubs[i];
      new_app_ctx.door_bell = &door_bells[i];
      new_app_ctx.d_door_bell = &d_door_bells[i];

      if(app_init(&new_app_ctx) == 0) {
        printf("P2pRpcAppServer Warning!, app_init failed\n");
      }
      app_ctx.push_back(new_app_ctx);
    }

    std::string ring_name = std::string("APP_SERVER_QUEUE"); 
    sub_queue = rte_ring_create(ring_name.c_str(), srv_qlen, rte_socket_id(), RING_F_SP_ENQ | RING_F_SC_DEQ);
    if(sub_queue == NULL) {
      printf("P2pRpcAppServer Error!: Unable to create work queue\n");
    }
  }

  P2pRpcAppServer()
  {
    // Dummy application
    printf("P2pRpcAppServer Warning!, setting up a dummy application");
    AppRx_payload_size = get_req_size();
    AppTx_payload_size = get_resp_size();
    device_id = get_cuda_device_id();
    num_req = 0;

    AppCtx new_app_ctx;
    new_app_ctx.launch_type = get_work_launch_type();
    num_instances = 1;
    checkCudaErrors(cudaStreamCreateWithFlags(&new_app_ctx.work_stream, cudaStreamNonBlocking));
    checkCudaErrors(cudaEventCreateWithFlags(&new_app_ctx.work_complete, cudaEventDisableTiming));
    new_app_ctx.h_stub = BufItemPool<g_params>::create_buf_item_pool(1, device_id);
    new_app_ctx.door_bell = BufItemPool<uint32_t>::create_buf_item_pool(1, device_id);
    new_app_ctx.d_stub = BufItemPool<g_params>::get_dev_ptr(new_app_ctx.h_stub);
    new_app_ctx.d_door_bell = BufItemPool<uint32_t>::get_dev_ptr(new_app_ctx.door_bell);

    //CopyCtx *new_copy_ctx;
    //new_copy_ctx = init_copy_ctx();
    ////new_copy_ctx = init_copy_ctx_on_stream(new_app_ctx.work_stream);
    //copy_ctx.push_back(new_copy_ctx);

    new_app_ctx.device_id = device_id;
    if(app_init(&new_app_ctx) == 0) {
      printf("P2pRpcAppServer Warning!, app_init failed\n");
    }
    app_ctx.push_back(new_app_ctx);
    srv_qlen = MAX_WI_SIZE;
    std::string ring_name = std::string("APP_SERVER_QUEUE"); 
    sub_queue = rte_ring_create(ring_name.c_str(), srv_qlen, rte_socket_id(), RING_F_SP_ENQ | RING_F_SC_DEQ);
    if(sub_queue == NULL) {
      printf("P2pRpcAppServer Error!: Unable to create work queue\n");
    }
  }

  ~P2pRpcAppServer()
  {
#ifdef PROFILE_MODE
  printf("DoWork stats(ns) [N, Mean, p90, p95, p99]: %d, %0.2f, %0.2f, %0.2f, %0.2f\n", 
      DwDelay.size(), getMean(DwDelay), getPercentile(DwDelay, 0.90), 
      getPercentile(DwDelay, 0.95), getPercentile(DwDelay, 0.99)); 
#endif
    for (auto ctx : app_ctx)
    {
      app_cleanup(&ctx);
    }
  }
};
