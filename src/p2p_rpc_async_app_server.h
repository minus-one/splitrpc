// Copyright (c) Adithya Kumar, The Pennsylvania State University. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include "debug_utils.h"
#include "time_utils.h"
#include "stats_utils.h"

#include "p2p_rpc_app_ctx.h"
#include "p2p_rpc_app_rr.h"

#define APP_RR_POOL_SIZE 4096

class P2pRpcAppInfo
{
  public:
    int appIdx; // Identifier to AsyncAppServer
    AppCtx* app_ctx;
    int device_id;

    AppInitCB app_init;
    AppRunCB app_run;
    AppCleanupCB app_cleanup;
    AppCompleteCB app_complete;

    P2pRpcAppRrPool *app_rr_pool;
    uint16_t rr_pool_size;
    size_t req_size, resp_size;
    g_params *h_stubs;
    g_params *d_stubs;
    uint32_t *door_bells;
    uint32_t *d_door_bells;

    P2pRpcAppInfo(AppInitCB _app_init,
        AppRunCB _app_run,
        AppCleanupCB _app_cleanup,
        AppCompleteCB _app_complete,
        int _device_id,
        size_t _req_size,
        size_t _resp_size,
        int _app_rr_pool_size = APP_RR_POOL_SIZE) 
    {
      app_init = _app_init;
      app_run = _app_run;
      app_cleanup = _app_cleanup;
      app_complete = _app_complete;
      device_id = _device_id;
      req_size = _req_size;
      resp_size = _resp_size;

      TRACE_PRINTF("Setting up P2pRpcAppRrPool\n");
      app_rr_pool = new P2pRpcAppRrPool(_app_rr_pool_size, device_id, req_size, resp_size);
      if(app_rr_pool == NULL) {
        printf("Failed to create App RR Pool...\n");
        exit(1);
      }
      rr_pool_size = _app_rr_pool_size;

      int num_instances = 1;

      TRACE_PRINTF("Creating P2pRpcAppInfo on device_id: %d\n", device_id);

      app_ctx = new AppCtx;
      app_ctx->curr_batch_size = get_ort_batch_size();
      app_ctx->next_batch_size = app_ctx->curr_batch_size;
      app_ctx->device_id = device_id;
      checkCudaErrors(cudaStreamCreateWithFlags(&app_ctx->work_stream, cudaStreamNonBlocking));
      checkCudaErrors(cudaEventCreateWithFlags(&app_ctx->work_complete, cudaEventDisableTiming));

      // This should be on GDR in case we use a persistent thread and use
      // GDR mem to communicate with them. Else these can just be on host-memory
      // The g_params stub is used to communicate with the GPU App on the memory location for the request/response
      app_ctx->launch_type = get_work_launch_type();
      if(app_ctx->launch_type == 3 || app_ctx->launch_type == 4) {
        h_stubs = BufItemPool<g_params>::create_buf_item_pool(num_instances, device_id);
        d_stubs = BufItemPool<g_params>::get_dev_ptr(h_stubs);
        door_bells = BufItemPool<uint32_t>::create_buf_item_pool(num_instances, device_id);
        d_door_bells = BufItemPool<uint32_t>::get_dev_ptr(door_bells);
      } else {
        h_stubs = new g_params[num_instances];
        d_stubs = h_stubs;
        checkCudaErrors(cudaHostAlloc(&door_bells, num_instances * sizeof(uint32_t), cudaHostAllocMapped));
        d_door_bells = door_bells;
      }
      app_ctx->h_stub = &h_stubs[0];
      app_ctx->d_stub = &d_stubs[0];
      app_ctx->door_bell = &door_bells[0];
      app_ctx->d_door_bell = &d_door_bells[0];

      app_ctx->ResetAppRunStatus();

      if(app_init(app_ctx) == 0) {
        printf("P2pRpcAsyncAppServer Warning!, app_init failed\n");
      }
    }

    ~P2pRpcAppInfo()
    {
      if(app_ctx->launch_type != 3 && app_ctx->launch_type != 4) {
        delete h_stubs;
        checkCudaErrors(cudaFreeHost(door_bells));
      }
      app_cleanup(app_ctx);
      delete app_ctx;
      printf("Cleaning up P2pRpcAppRrPool\n");
      delete app_rr_pool;
    }
};

class P2pRpcAsyncAppServer
{
#ifdef PROFILE_MODE
  uint64_t dw_startNs;
  std::vector<uint64_t> DwDelay;
#endif

  // Shared across instances 
  std::vector<P2pRpcAppInfo*> all_apps;
  uint32_t num_req;
  volatile int32_t force_quit_server;

public:

  inline void quit()
  {
    ACCESS_ONCE(force_quit_server) = 1;
    _mm_mfence();
    printf("Shutting down P2pRpcAsyncAppServer, force_quit_server value: %d\n", force_quit_server);
  }

  inline void update_batch_size(int appIdx)
  {
    //AppCtx *app_ctx = all_app_ctx[appIdx]; 
    AppCtx *app_ctx = all_apps[appIdx]->app_ctx; 
    // FIXME:Do a mutex lock
    app_ctx->curr_batch_size = app_ctx->next_batch_size;
  }

  inline void do_work(int appIdx, int rr_idx)
  {
    AppCtx *app_ctx = all_apps[appIdx]->app_ctx;
    P2pRpcAppRrPool *app_rr_pool = all_apps[appIdx]->app_rr_pool;
    TRACE_PRINTF("AppCtx: %p, AppRrIdx: %d\n", (void*)app_ctx, rr_idx);
    *app_ctx->h_stub = *app_rr_pool->get_app_rr(rr_idx)->h_stub;
    //_mm_mfence();
    all_apps[appIdx]->app_run(app_ctx);
  }

  inline void do_sync(int appIdx)
  {
    AppCtx *app_ctx = all_apps[appIdx]->app_ctx;
    app_ctx->WaitForAppRunComplete();
    all_apps[appIdx]->app_complete(app_ctx);
  }

  // This does not check if app_rr is actually ready.
  // It just runs
  // FIXME: Move this to private
  inline void do_work_sync(int appIdx, int rr_idx)
  {
    NVTX_R("AppRun");
    AppCtx *app_ctx = all_apps[appIdx]->app_ctx;
    P2pRpcAppRrPool *app_rr_pool = all_apps[appIdx]->app_rr_pool;
    TRACE_PRINTF("AppCtx: %p, AppRrIdx: %d\n", (void*)app_ctx, rr_idx);
    *app_ctx->h_stub = *app_rr_pool->get_app_rr(rr_idx)->h_stub;
    //_mm_mfence();
    all_apps[appIdx]->app_run(app_ctx);
    app_ctx->WaitForAppRunComplete();
    all_apps[appIdx]->app_complete(app_ctx);
    NVTX_P;
  }

  void sync_worker_loop(int appIdx)
  {
    AppCtx *app_ctx = all_apps[appIdx]->app_ctx;
    P2pRpcAppRrPool *app_rr_pool = all_apps[appIdx]->app_rr_pool;
    uint16_t app_rr_pool_size = all_apps[appIdx]->rr_pool_size;
    printf("Starting sync worker loop thread, app_ctx: %p\n", app_ctx);

    int rr_ci_idx = 0;
    int rr_si_idx = 0;
    int rr_win_size = 0;
    while(ACCESS_ONCE(force_quit_server) == 0) 
    {
      if(*ACCESS_ONCE(app_rr_pool->get_app_rr(rr_ci_idx)->h_state) == APP_RR_STATUS::RX_COMPLETE) {
        TRACE_PRINTF("Got RX_COMPLETE AppCtx: %p, rr_ci_idx: %d, AppRr: %p\n", 
            (void*)app_ctx, rr_ci_idx, (void*)app_rr_pool->get_app_rr(rr_ci_idx));
#ifdef PROFILE_MODE
        dw_startNs = getCurNs();
#endif
        rr_ci_idx = (rr_ci_idx + 1) % app_rr_pool_size;
        rr_win_size++;

        if(rr_win_size == app_ctx->curr_batch_size) {
          do_work_sync(appIdx, rr_si_idx);
          num_req += app_ctx->curr_batch_size;
          for(int i = rr_si_idx ; i != rr_ci_idx ; i = (i+1) % app_rr_pool_size) {
            *ACCESS_ONCE(app_rr_pool->get_app_rr(i)->h_state) = APP_RR_STATUS::WORK_COMPLETE;
          }
          rr_si_idx = rr_ci_idx;
          rr_win_size = 0;
          update_batch_size(appIdx);
        }
#ifdef PROFILE_MODE
        DwDelay.push_back(getCurNs() - dw_startNs);
#endif
      }
    }
    printf("Ending sync worker loop thread, app_ctx: %p, num_req: %d\n", app_ctx, num_req);
  }

  // The start_rr_idx will be used to pick up the starting address of the stubs
  inline void do_batch_work_sync(int appIdx, int start_rr_idx, int batch_size)
  {
    AppCtx *app_ctx = all_apps[appIdx]->app_ctx;
    app_ctx->curr_batch_size = batch_size;
    do_work_sync(appIdx, start_rr_idx);
  }

  void dynamic_batching_sync_worker_loop(int appIdx, const int max_batch_size)
  {
    AppCtx *app_ctx = all_apps[appIdx]->app_ctx;
    P2pRpcAppRrPool *app_rr_pool = all_apps[appIdx]->app_rr_pool;
    uint16_t app_rr_pool_size = all_apps[appIdx]->rr_pool_size;
    printf("Starting dynamic batching sync worker loop thread, app_ctx: %p, max_batch_size: %d\n", 
        app_ctx, max_batch_size);

    std::vector<uint64_t> winSizes;
    int rr_ci_idx = 0;
    int rr_si_idx = rr_ci_idx;
    int rr_win_size = 0;

    while(ACCESS_ONCE(force_quit_server) == 0)
    {
      while(rr_win_size < max_batch_size && 
          *ACCESS_ONCE(app_rr_pool->get_app_rr(rr_ci_idx)->h_state) == APP_RR_STATUS::RX_COMPLETE) {
        rr_win_size++;
        rr_ci_idx = (rr_ci_idx + 1) % app_rr_pool_size;
      }
      if(rr_win_size > 0) {
        TRACE_PRINTF("Got RX_COMPLETE rr_si_idx: %d, rr_ci_idx: %d, batch_size: %d\n", 
            rr_si_idx, rr_ci_idx, rr_win_size);
#ifdef PROFILE_MODE
        dw_startNs = getCurNs();
#endif
        do_batch_work_sync(appIdx, rr_si_idx, rr_win_size);
#ifdef PROFILE_MODE
        DwDelay.push_back(getCurNs() - dw_startNs);
#endif
        num_req += rr_win_size;
        for(int i = rr_si_idx ; i != rr_ci_idx; i = (i+1) % app_rr_pool_size) {
          *ACCESS_ONCE(app_rr_pool->get_app_rr(i)->h_state) = APP_RR_STATUS::WORK_COMPLETE;
        }
        winSizes.push_back(rr_win_size);
        rr_win_size = 0;
        rr_si_idx = rr_ci_idx;
        TRACE_PRINTF("Will next wait for rr_ci_idx: %d, h_state: %p\n", 
            rr_ci_idx, (void*)app_rr_pool->get_app_rr(rr_ci_idx)->h_state);
      }
    }
    printf("Ending dynamic batching sync worker loop thread, app_ctx: %p, num_req: %d\n", app_ctx, num_req);
    print_stat("WinSizes stats: ", winSizes);
  }

  // The start_rr_idx will be used to pick up the starting address of the stubs
  // The last_rr_idx will be used to pick up the notification
  inline void pre_launch_work(int appIdx, int start_rr_idx, int last_rr_idx)
  {
    AppCtx *app_ctx = all_apps[appIdx]->app_ctx;
    P2pRpcAppRrPool *app_rr_pool = all_apps[appIdx]->app_rr_pool;

    // The next set of rr's that will be expected
    // Copy the details of the buffers into the app_ctx'stub
    *app_ctx->h_stub = *app_rr_pool->get_app_rr(start_rr_idx)->h_stub;
    _mm_mfence();
    TRACE_PRINTF("PreLaunchWork, app_ctx: %p, h_stub: %p, start_rr_idx: %d, last_rr_idx: %d\n",
        (void*)app_ctx, (void*)app_ctx->h_stub, start_rr_idx, last_rr_idx);
    // Starting from rr_ci_idx to rr_ci_idx + app_ctx->curr_batch_size
    // is the next call
    app_ctx->AppRrRunNotify(app_rr_pool->get_app_rr(last_rr_idx)->d_state, APP_RR_STATUS::RX_COMPLETE);
    //app_ctx->AppRunNotify(app_ctx->curr_batch_size);
    // Do launch of the app
    all_apps[appIdx]->app_run(app_ctx);
    // Register completion notifier
    app_ctx->AppCompleteNotify(1U);
  }

  // Worker function 
  void async_worker_loop(int appIdx)
  {
    AppCtx *app_ctx = all_apps[appIdx]->app_ctx;
    P2pRpcAppRrPool *app_rr_pool = all_apps[appIdx]->app_rr_pool;
    uint16_t app_rr_pool_size = all_apps[appIdx]->rr_pool_size;

    // This will be local to one app_rr_pool
    int rr_ci_idx = 0;
    // Before we start, we reset the work-status and pre-launch work
    app_ctx->ResetAppRunStatus();
    pre_launch_work(appIdx, rr_ci_idx, (rr_ci_idx + app_ctx->curr_batch_size - 1) % app_rr_pool_size);

    printf("Pre-launch complete...async worker standing by...\n");

    while(ACCESS_ONCE(force_quit_server) == 0)
    {
      // Consumer action
      // Consumer will first notify the completion of requests
      // and then trigger the launch for the next set of requests
      // The producer will periodically notify as and when those
      // requests become ready
      if(app_ctx->IsAppRunComplete()) {
        TRACE_PRINTF("AppRunComplete rr_ci_idx: %d\n", rr_ci_idx);
        num_req += app_ctx->curr_batch_size;
        
        // Notify rr & move rr_ci_idx
        for(int i = 0 ; i < app_ctx->curr_batch_size; i++) {
          *ACCESS_ONCE(app_rr_pool->get_app_rr(rr_ci_idx)->h_state) = APP_RR_STATUS::WORK_COMPLETE;
          rr_ci_idx = (rr_ci_idx + 1) % app_rr_pool_size;
        }
        
        // Be ready for next work
        // First check if the batch-size has changed
        // Consumer action is the only place where we 
        // can actually change the batch-size
        update_batch_size(appIdx);
        TRACE_PRINTF("AppRunComplete, new batch_size: %d\n", app_ctx->curr_batch_size);

        // Reset and launch work 
        app_ctx->ResetAppRunStatus();
        all_apps[appIdx]->app_complete(app_ctx);
        pre_launch_work(appIdx, rr_ci_idx, (rr_ci_idx + app_ctx->curr_batch_size - 1) % app_rr_pool_size);
      }
    }

    // In case there is an outstanding run it will run-through and finish
    for(int i = 0 ; i < app_rr_pool_size; i++)
      *ACCESS_ONCE(app_rr_pool->get_app_rr(i)->h_state) = APP_RR_STATUS::RX_COMPLETE;
    app_ctx->NotifyAppRunStart();
    _mm_mfence();
    printf("Exiting async worker loop thread, waiting for all launches to complete...\n");
    app_ctx->WaitForAppRunComplete();
    printf("Ending async worker loop thread\n");
  }

  int register_app(P2pRpcAppInfo *app_info)
  {
    all_apps.push_back(app_info);
    return all_apps.size() - 1;
  }

  P2pRpcAsyncAppServer()
  {
    num_req = 0;
    TRACE_PRINTF("Setting up P2pRpcAppServer, ready to register apps\n");
    force_quit_server = 0;
  }

  ~P2pRpcAsyncAppServer()
  {
    PROF_PRINT("DoWork", DwDelay);
  }
};
