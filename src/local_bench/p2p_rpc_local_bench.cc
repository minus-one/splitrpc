// Copyright (c) Adithya Kumar, The Pennsylvania State University. All rights reserved.
// Licensed under the MIT License.

#include <cmath>
#include <algorithm>
#include <signal.h>

#include "p2p_rpc_rr_ng.h"

#include "time_utils.h"
#include "stats_utils.h"
#include "config_utils.h"

#include "g_utils.cuh"

#include "p2p_rpc_app_rr.h"
#include "p2p_rpc_async_app_server.h"
extern AppInitCB AppInit_cb;
extern AppRunCB AppRun_cb;
extern AppCleanupCB AppCleanup_cb;
extern AppRunAsyncCB AppRunAsync_cb;
extern AppRunWaitCB AppRunWait_cb;
extern AppCompleteCB AppComplete_cb;
P2pRpcAppInfo *app_info = NULL;
P2pRpcAsyncAppServer *app_server = NULL;

P2pRpcAppRrPool *app_rr_pool = NULL;

volatile bool force_quit = 0;
static void 
signal_handler(int signum) {                                                                          
  if (signum == SIGINT) {        
      printf("\n\nSignal %d received, preparing to exit...\n", signum);      
      ACCESS_ONCE(force_quit) = 1;
      if(app_server)
        app_server->quit();
    }
}

int
main() {
////////////////////////// Application initializations

  signal(SIGINT, signal_handler);

/////////////////////////// Context setup

  TRACE_PRINTF("Setting up P2pRpcAppInfo\n");
  app_info = new P2pRpcAppInfo(AppInit_cb, AppRun_cb, AppCleanup_cb, AppComplete_cb, 
      get_cuda_device_id(), get_req_size(), get_resp_size());
  if(app_info == NULL) {
    printf("Failed to create app info...\n");
    exit(1);
  }

  TRACE_PRINTF("Setting up P2pRpcAsyncAppServer\n");
  app_server = new P2pRpcAsyncAppServer(); 
  if(app_server == NULL) {
    printf("Failed to create app server...\n");
    exit(1);
  }
  app_info->appIdx = app_server->register_app(app_info);

  app_rr_pool= app_info->app_rr_pool;
  
////////////////////////// Run of app

  int batch_size = get_ort_batch_size();
  P2pRpcAppRr *app_rr[batch_size] = {NULL};

  for(int i = 0 ; i < batch_size; i++)
    app_rr[i] = app_rr_pool->get_next();

  float *h_in_addr;
  size_t req_size = get_req_size();
  size_t resp_size = get_resp_size();
  checkCudaErrors(cudaMallocHost((void**)&h_in_addr, req_size));
  for (int i = 0; i < req_size/sizeof(float); ++i) h_in_addr[i] = 5.0f;
  checkCudaErrors(cudaMemcpy(app_rr[0]->h_stub->req, h_in_addr, req_size, cudaMemcpyHostToDevice));
  cudaDeviceSynchronize();

  for(int i = 0 ; i < 2; i++) {
    printf("Running inference for rr_idx: %d\n", app_rr[0]->rr_idx);
    app_server->do_batch_work_sync(0, app_rr[0]->rr_idx, batch_size);
  }

  printf("Sample output: resp_size: %ld\n", resp_size);
  g_floatDump(app_rr[0]->h_stub->resp, resp_size);
  g_floatDump(app_rr[0]->h_stub->resp+(resp_size - sizeof(float)*5), 5);

////////////////////////// Bench run

  uint64_t startNs;
  int numMetrics = 50000;
  std::vector<uint64_t> metricValues;
  metricValues.reserve(numMetrics);
  for(int i = 0 ; i < numMetrics; i++) {
    startNs = getCurNs();
    app_server->do_batch_work_sync(0, app_rr[0]->rr_idx, batch_size);
    metricValues.push_back(getCurNs() - startNs);
  }

  printf("LocalBench stats(ns) [BatchSize: %d] [N, Mean, p90, p95, p99]: %ld, %0.2f, %0.2f, %0.2f, %0.2f\n", 
      batch_size, metricValues.size(), getMean(metricValues), getPercentile(metricValues, 0.90), 
      getPercentile(metricValues, 0.95), getPercentile(metricValues, 0.99)); 

//////////////////////// Cleanup

  printf("Stopping app...\n");
  delete app_server;
  printf("Cleaning up AppInfo\n");
  delete app_info;
  printf("Exiting app cleanly\n");
  return 0;
}
