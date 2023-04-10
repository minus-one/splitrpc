// Copyright (c) Adithya Kumar, The Pennsylvania State University. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include <queue>

//#include "p2p_rpc_rr.h"
//#include "p2p_rpc.h"

#include <emmintrin.h>
#include <rte_ring.h>

#include "time_utils.h"
#include "stats_utils.h"
#include "config_utils.h"

template <class T>
template <class R>
class P2pRpcGenericServer
{
  int (*)(T**, int, int) app_init; // Initializes specific number of instances of an app on a specific device  and returns all Ctx's 
  int (*)(T*, R*) app_run; // Given a context T*, and an input/output R*, it runs the app on the input R
  void (*)(T*) app_sync; // Waits for context T* to complete
  int (*)(T*) app_cleanup; // Cleans up the context

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
  std::vector<T> app_ctx;

  // Current arch is 1 completion queue and 1 submission queues
  // Submission queue will be owned by the server
  // Completion queue will be owned by the transport entity
  struct rte_ring *sub_queue;

  // FIXME: This should be a completion CB
  struct rte_ring *completion_queue;

  // Stats
  uint32_t num_req;

  volatile bool force_quit = 0;

  int nServerThreads;                         
  std::vector<std::thread> serverThreads;

public:

  inline void sync(int instance = 0)
  {
    app_sync(app_ctx[instance]);
  }

  inline void do_work(R *req, int instance = 0)
  {
#ifdef PROFILE_MODE
  dw_startNs = getCurNs();
#endif

    if(app_run(&app_ctx[instance], req)) {
      printf("P2pRpcGenericServer Warning, app_run failed\n");
    }

#ifdef PROFILE_MODE
  DwDelay.push_back(getCurNs() - dw_startNs);
#endif
  }

  inline void do_work_sync(T *req, int instance = 0)
  {
    do_work(req, instance);
    sync(instance);
  }

  // Call register_cq before enqueueing work
  void enqueue_work(T *req)
  {
    while (rte_ring_sp_enqueue(sub_queue, (void *)req) != 0)
      ;
  }

  void quit()
  {
    ACCESS_ONCE(force_quit) = 1;

    for(int i=0; i < nServerThreads; i++) {
      serverThreads[i].join();
    }
  }

  // Registers the queue to be notified (typically the IO queue) when work is complete
  void register_cq(struct rte_ring *ring_cq)
  {
    printf("P2pRpcGenericServer, registering completion queue: %p\n", (void*)ring_cq);
    completion_queue = ring_cq;
  }

  // Worker function
  void request_handler_worker(int instance = 0)
  {
    T *new_rr;
    while (ACCESS_ONCE(force_quit) == 0)
    {
      // Dequeue req from ringbuffer
      while (rte_ring_sc_dequeue(sub_queue, (void **)&new_rr) != 0 && ACCESS_ONCE(force_quit) == 0)
        ;
      if (unlikely(ACCESS_ONCE(force_quit) != 0))
        break;
      TRACE_PRINTF("DO_WORK: RR: %p\n", (void*)new_rr);
      do_work(new_rr, instance);
      sync(instance);

      num_req++;

      TRACE_PRINTF("DO_WORK Complete, enqueueing into cq new_rr: %p\n", (void*)new_rr);
      // enqueue req backinto completion ringbuffer
      while (rte_ring_sp_enqueue(completion_queue, (void *)new_rr) != 0 && ACCESS_ONCE(force_quit) == 0)
        ;
      if (unlikely(ACCESS_ONCE(force_quit) != 0))
        break;
    }
    printf("P2pRpcGenericServer, Reqs: %d serviced, Terminating rpc handler for instance %d\n", num_req, instance);
  }

  void StartWorkers()
  {
    // Proceed to the server's main loop.
    // Has to be single threaded if it is a M/M/1 type of a system
    for(int i=0; i< nServerThreads; i++) {
      serverThreads.push_back(std::thread(&P2pRpcGenericServer::request_handler_worker, this, i));
    }
  }

  P2pRpcGenericServer(
      int (*)(T**, int, int) _app_init, 
      int (*)(T*, R*) _app_run, 
      void (*)(T*) _app_sync, 
      int (*)(T*) _app_cleanup, 
      int _num_instances = 1,
      int _device_id = get_cuda_device_id())
  {
    app_init = _app_init;
    app_run = _app_run;
    app_sync = _app_sync;
    app_cleanup = _app_cleanup;
    num_instances = _num_instances;
    device_id = _device_id;
    srv_qlen = 2 * MAX_WI_SIZE;

    num_req = 0;
    TRACE_PRINTF("Setting up P2pRpcAppServer, num_instances: %d, device_id: %d, queue_length: %d\n", num_instances, device_id, srv_qlen);

    struct T** all_app_ctx;
    if(app_init(all_app_ctx, device_id, num_instances)) {
      printf("P2pRpcGenericServer Warning!, app_init failed\n");
    }
    for (int i = 0; i < num_instances; i++) { 
      app_ctx.push_back(all_app_ctx[i]);
    }

    std::string ring_name = std::string("APP_SERVER_QUEUE"); 
    // FIXME: If this is going to be truly multi-threaded, then this should not be a SPSC queue
    sub_queue = rte_ring_create(ring_name.c_str(), srv_qlen, rte_socket_id(), RING_F_SP_ENQ | RING_F_SC_DEQ);
    if(sub_queue == NULL) {
      printf("P2pRpcAppServer Error!: Unable to create work queue\n");
    }
    completion_queue = NULL;
  }

  ~P2pRpcAppServer()
  {
#ifdef PROFILE_MODE
  printf("DoWork stats(ns) [N, Mean, p90, p95, p99]: %d, %0.2f, %0.2f, %0.2f, %0.2f\n", 
      DwDelay.size(), getMean(DwDelay), getPercentile(DwDelay, 0.90), 
      getPercentile(DwDelay, 0.95), getPercentile(DwDelay, 0.99)); 
#endif
    for (auto ctx : app_ctx) {
      app_cleanup(&ctx);
    }
  }
};

#include "p2p_buf_pool.h"
#include "p2p_rpc_rr.h"
#include <emmintrin.h>
#include <helper_functions.h>
#include <helper_cuda.h>

static int default_init_all_app_ctx(AppCtx **all_ctx, int device_id, int num_instances) 
{
  *all_ctx = new AppCtx[num_instances];
  // This is only needed in case we use a persistent thread and use
  // GDR mem to communicate with them. Else these can just be on host-memory
  // The g_params stub is used to communicate with the GPU App on the memory location for the request/response
  g_params *h_stubs = BufItemPool<g_params>::create_buf_item_pool(num_instances, device_id);
  g_params *d_stubs = BufItemPool<g_params>::get_dev_ptr(h_stubs);
  uint32_t *door_bells = BufItemPool<uint32_t>::create_buf_item_pool(num_instances, device_id);
  uint32_t *d_door_bells = BufItemPool<uint32_t>::get_dev_ptr(door_bells);
  for (int i = 0; i < num_instances; i++)
  {
    all_ctx[i] = new AppCtx;
    AppCtx *new_app_ctx = all_ctx[i];

    new_app_ctx->launch_type = get_work_launch_type();
    new_app_ctx->device_id = device_id;
    if(device_id >= 0) {
      checkCudaErrors(cudaStreamCreateWithFlags(&new_app_ctx->work_stream, cudaStreamNonBlocking));
      checkCudaErrors(cudaEventCreateWithFlags(&new_app_ctx->work_complete, cudaEventDisableTiming));
    }
    new_app_ctx->h_stub = &h_stubs[i];
    new_app_ctx->d_stub = &d_stubs[i];
    new_app_ctx->door_bell = &door_bells[i];
    new_app_ctx->d_door_bell = &d_door_bells[i];
    // Here App specific initializations can happen
    // In the default App Init, we do nothing
  }
  return 0;
}

inline void default_app_sync(AppCtx *ctx)
{
  // If work is non-device
  if(ctx->device_id < 0)
    return ;

  if (ctx->launch_type == 1 || ctx->launch_type == 2 || ctx->launch_type == 4) {
    checkCudaErrors(cudaStreamSynchronize(ctx->work_stream));
  } else {
    while (*ACCESS_ONCE(ctx->door_bell) != 2)
      ;
  }
}

static int default_app_run(AppCtx *ctx, struct p2p_rpc_rr *req)
{
  ctx->curr_rr = req;
  ctx->h_stub->req = req->req_payload;
  ctx->h_stub->resp = req->resp_payload;
  _mm_mfence();

  TRACE_PRINTF("app_run ctx: %p, req_payload: %p, resp_payload: %p\n", 
      (void*)ctx, ctx->h_stub->req, ctx->h_stub->resp);

  // Internal call to the actual application code

  return 0;
}

static int default_app_cleanup(AppCtx *ctx)
{
  // Cleanup the ctx related things

  delete ctx;
  return 0;
}
