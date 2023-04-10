// Copyright (c) Adithya Kumar, The Pennsylvania State University. All rights reserved.
// Licensed under the MIT License.

#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <vector>
#include <iostream>
#include <signal.h>

#include "ort_app.h"
#include "p2p_rpc_app_ctx.h"
#include "debug_utils.h"
#include "time_utils.h"
#include "stats_utils.h"

#ifdef PROFILE_MODE
  uint64_t ar_startNs;
  std::vector<uint64_t> ArDelay;
#endif

inline int app_run(AppCtx *app_ctx)
{
#ifdef PROFILE_MODE
  ar_startNs = getCurNs();
#endif
  TRACE_PRINTF("Start AppCtx: %p, BS: %d, Req: %p, Resp: %p\n", 
      (void*)app_ctx, app_ctx->curr_batch_size, (void*)app_ctx->h_stub->req, (void*)app_ctx->h_stub->resp);

  OrtFacade *AppServer = (OrtFacade*)app_ctx->app_ctx_internal;
  AppServer->setup_run_with_data(app_ctx->h_stub->req, app_ctx->h_stub->resp, app_ctx->curr_batch_size);
  TRACE_PRINTF("Complete AppCtx: %p\n", (void*)app_ctx);
#ifdef PROFILE_MODE
  ArDelay.push_back(getCurNs() - ar_startNs);
#endif
  return 1;
}

int app_init(AppCtx *app_ctx)
{
  std::string model_path = getDatasetBasePath() + std::string("/models/") + get_ort_model_name();
  TRACE_PRINTF("OrtAppInit: Model: %s\n", model_path.c_str());

  OrtFacade *AppServer = new OrtFacade(app_ctx->device_id, app_ctx->work_stream);
  AppServer->loadModel(model_path);
  app_ctx->app_ctx_internal = (void*)AppServer;
  //AppServer->printModelInfo();

  return 1;
}

int app_cleanup(AppCtx* app_ctx)
{
#ifdef PROFILE_MODE
  printf("ArWork stats(ns) [N, Mean, p90, p95, p99]: %ld, %0.2f, %0.2f, %0.2f, %0.2f\n", 
      ArDelay.size(), getMean(ArDelay), getPercentile(ArDelay, 0.90), 
      getPercentile(ArDelay, 0.95), getPercentile(ArDelay, 0.99)); 
#endif
  OrtFacade *AppServer = (OrtFacade*)app_ctx->app_ctx_internal;
  delete AppServer;
  app_ctx->app_ctx_internal = NULL;
  return 1;
}

// Marks app execution to be complete
inline int app_complete(AppCtx *app_ctx)
{
  TRACE_PRINTF("OrtAppComplete, app_ctx: %p\n", (void*)app_ctx);
  OrtFacade *AppServer = (OrtFacade*)app_ctx->app_ctx_internal;
  AppServer->run_complete();
  return 1;
}

//int app_local_bench()
//{
//  printf("LocalBench...\n");
//  std::vector<void*>d_data;
//  OrtFacade *AppServer = new OrtFacade();
//  AppServer->setup_io_binding(d_data, d_data);
//  for(int i = 0 ; i < 10 ; i++)
//    AppServer->predict_with_io_binding();
//
//  uint64_t startNs, endNs;
//  int numMetrics = 20000;
//  std::vector<uint64_t> metricValues(numMetrics, 0);
//  int metricNum = 0;
//  for(int i = 0 ; i < numMetrics; i++) {
//    startNs = getCurNs();
//    AppServer->predict_with_io_binding();
//    endNs = getCurNs();
//    metricValues[metricNum] = endNs - startNs;
//    metricNum = (metricNum + 1) % numMetrics;
//  }
//  printf("LocalBench stats(ns) [N, Mean, p90, p95, p99]: %d, %0.2f, %0.2f, %0.2f, %0.2f\n", 
//      numMetrics, getMean(metricValues), getPercentile(metricValues, 0.90), 
//      getPercentile(metricValues, 0.95), getPercentile(metricValues, 0.99)); 
//  return 1;
//}

AppInitCB AppInit_cb = &app_init;
AppRunCB AppRun_cb = &app_run;
AppCleanupCB AppCleanup_cb = &app_cleanup;
AppCompleteCB AppComplete_cb = &app_complete;
