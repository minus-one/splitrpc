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

#include "p2p_rpc_async_app_server.h"
extern AppInitCB AppInit_cb;
extern AppRunCB AppRun_cb;
extern AppCleanupCB AppCleanup_cb;
extern AppRunAsyncCB AppRunAsync_cb;
extern AppRunWaitCB AppRunWait_cb;
P2pRpcAsyncAppServer *app_server = NULL;

#include "p2p_rpc_app_rr.h"
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

  TRACE_PRINTF("Setting up P2pRpcAppRrPool\n");
  app_rr_pool = new P2pRpcAppRrPool(MAX_WI_SIZE, get_cuda_device_id(), get_req_size(), get_resp_size());
  if(app_rr_pool == NULL || app_rr_pool->get_app_rr_pool() == NULL) {
    printf("Failed to create App RR Pool...\n");
    exit(1);
  }
  size_t app_rr_pool_size = app_rr_pool->get_pool_size();

  TRACE_PRINTF("Setting up P2pRpcAsyncAppServer\n");
  app_server = new P2pRpcAsyncAppServer(AppInit_cb, AppRun_cb, AppCleanup_cb); 
  if(app_server == NULL) {
    printf("Failed to create app server...\n");
    exit(1);
  }
  app_server->register_app_rr_pool(app_rr_pool);

////////////////////////// Run of app

  int batch_size = 4;
  //P2pRpcAppRr *app_rr = NULL, *app_rr_2 = NULL;
  P2pRpcAppRr *app_rr[MAX_WI_SIZE] = {NULL};

  //for(int j = 0; j < MAX_WI_SIZE; j++) {
  //  app_rr[j] = app_rr_pool->get_next();
  //  app_rr[j]->set_dummy_output_on_gpu();
  //}
  app_rr[MAX_WI_SIZE-1] = app_rr_pool->get_next();

  //for(int j = 0; j < MAX_WI_SIZE; j++)
  //  app_rr[j]->print_app_rr_info();
  
  //for(int i = 0 ; i < 2; i++) {
  //  printf("Running inference for rr_idx: %d\n", app_rr[3]->rr_idx);
  //  app_server->do_work_sync(0, app_rr[3]->rr_idx);
  //}

////////////////////////// Bench run

  size_t app_rr_idx_to_run = 1;
  uint64_t startNs;
  int numMetrics = 1000;
  std::vector<uint64_t> metricValues;
  metricValues.reserve(numMetrics);
  for(int i = 0 ; i < numMetrics; i++) {
    startNs = getCurNs();
    for(int j = 0; j < batch_size; j++) {
      app_rr[j] = app_rr_pool->get_next();
      app_rr[j]->set_dummy_output_on_gpu();
    }

    //for(int j = 0; j < batch_size; j++)
    //  app_rr[j]->print_app_rr_info();

    //app_rr_idx_to_run = (((i * batch_size) + 3) % app_rr_pool_size);
    app_rr_idx_to_run = app_rr[0]->rr_idx;
    app_server->do_work_sync(0, app_rr_idx_to_run);
    for(int j = 0 ; j < batch_size; j++) 
      app_rr[j]->dump_output_on_gpu();
    metricValues.push_back(getCurNs() - startNs);
  }

  //for(int j = 0 ; j < MAX_WI_SIZE; j++) {
  //  app_rr[j]->dump_output_on_gpu();
  //}

  printf("LocalBench stats(ns) [N, Mean, p90, p95, p99]: %ld, %0.2f, %0.2f, %0.2f, %0.2f\n", 
      metricValues.size(), getMean(metricValues), getPercentile(metricValues, 0.90), 
      getPercentile(metricValues, 0.95), getPercentile(metricValues, 0.99)); 

  
//////////////////////// Cleanup

  printf("Stopping app...\n");
  delete app_server;
  printf("Cleaning up P2pRpcAppRrPool\n");
  delete app_rr_pool;
  printf("Exiting app cleanly\n");
  return 0;
}
