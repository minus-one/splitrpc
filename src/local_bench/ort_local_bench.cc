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
#include "p2p_buf_pool.h"

#include <helper_functions.h>
#include <helper_cuda.h>
#include <emmintrin.h>

//#include <cuda.h>
//#include <cuda_runtime.h>
//#include <cuda_runtime_api.h>

// Globally shared
OrtFacade *AppServer;

#ifdef PROFILE_MODE
  uint64_t ar_startNs;
  std::vector<uint64_t> ArDelay;
#endif

int app_run(AppCtx *app_ctx)
{
#ifdef PROFILE_MODE
  ar_startNs = getCurNs();
#endif
  TRACE_PRINTF("AppRun Start CTX: %p\n", (void*)app_ctx);
  //AppServer.copyInput(req, inputLen);
  if(app_ctx->launch_type == 1) {
    AppServer->load_data_and_predict(app_ctx->h_stub->req, app_ctx->h_stub->resp);
  } else if(app_ctx->launch_type == 4) {
    if(!app_ctx->graphCreated) {
      printf("Constructing CUDA graph\n");
      checkCudaErrors(cudaStreamBeginCapture(app_ctx->work_stream, cudaStreamCaptureModeGlobal));
      AppServer->load_data_and_predict(app_ctx->h_stub->req, app_ctx->h_stub->resp);
      checkCudaErrors(cudaStreamEndCapture(app_ctx->work_stream, &app_ctx->graph));
      checkCudaErrors(cudaGraphInstantiate(&app_ctx->instance, app_ctx->graph, NULL, NULL, 0));
      app_ctx->graphCreated = true;
    }
    checkCudaErrors(cudaGraphLaunch(app_ctx->instance, app_ctx->work_stream));
  }
  //AppServer.copyOutput(resp, outputLen);
  TRACE_PRINTF("AppRun complete CTX: %p\n", (void*)app_ctx);
#ifdef PROFILE_MODE
  ArDelay.push_back(getCurNs() - ar_startNs);
#endif
  return 1;
}

int app_init(AppCtx *app_ctx)
{
  std::string model_path = getDatasetBasePath() + std::string("/models/") + get_ort_model_name();
  TRACE_PRINTF("OrtAppInit: Model: %s\n", model_path.c_str());
  AppServer = new OrtFacade(app_ctx->device_id, app_ctx->work_stream);
  AppServer->loadModel(model_path);
  AppServer->printModelInfo();

  return 1;
}

int app_cleanup(AppCtx *app_ctx)
{
#ifdef PROFILE_MODE
  printf("ArWork stats(ns) [N, Mean, p90, p95, p99]: %d, %0.2f, %0.2f, %0.2f, %0.2f\n", 
      ArDelay.size(), getMean(ArDelay), getPercentile(ArDelay, 0.90), 
      getPercentile(ArDelay, 0.95), getPercentile(ArDelay, 0.99)); 
#endif
  delete AppServer;
  return 1;
}

int app_local_bench()
{
  printf("LocalBench...\n");
  std::vector<void*>d_data;
  AppServer->setup_io_binding(d_data, d_data);
  for(int i = 0 ; i < 10 ; i++)
    AppServer->predict_with_io_binding();

  uint64_t startNs, endNs;
  int numMetrics = 20000;
  std::vector<uint64_t> metricValues(numMetrics, 0);
  int metricNum = 0;
  for(int i = 0 ; i < numMetrics; i++) {
    startNs = getCurNs();
    AppServer->predict_with_io_binding();
    endNs = getCurNs();
    metricValues[metricNum] = endNs - startNs;
    metricNum = (metricNum + 1) % numMetrics;
  }
  printf("LocalBench stats(ns) [N, Mean, p90, p95, p99]: %d, %0.2f, %0.2f, %0.2f, %0.2f\n", 
      numMetrics, getMean(metricValues), getPercentile(metricValues, 0.90), 
      getPercentile(metricValues, 0.95), getPercentile(metricValues, 0.99)); 
  return 0;
}

int main()
{
  AppCtx *app_ctx = new AppCtx;
  app_ctx->launch_type = get_work_launch_type();
  app_ctx->device_id = get_cuda_device_id();
  checkCudaErrors(cudaStreamCreateWithFlags(&app_ctx->work_stream, cudaStreamNonBlocking));
  checkCudaErrors(cudaEventCreateWithFlags(&app_ctx->work_complete, cudaEventDisableTiming));

  app_ctx->h_stub = new g_params;

  //app_ctx->h_stub = BufItemPool<g_params>::create_buf_item_pool(1, app_ctx->device_id);
  //app_ctx->door_bell = BufItemPool<uint32_t>::create_buf_item_pool(1, app_ctx->device_id);
  //app_ctx->d_stub = BufItemPool<g_params>::get_dev_ptr(app_ctx->h_stub);
  //app_ctx->d_door_bell = BufItemPool<uint32_t>::get_dev_ptr(app_ctx->door_bell);

  size_t max_req_size = get_req_size();
  size_t max_resp_size = get_resp_size();

  void *d_in_addr, *d_out_addr;
  if (cudaMalloc((void **)&d_in_addr, max_req_size) != cudaSuccess) {
    printf("CudaMalloc failed, unable to create tmp_buffers\n");
    exit(1);
  }
  if (cudaMalloc((void **)&d_out_addr, max_resp_size) != cudaSuccess) {
    printf("CudaMalloc failed, unable to create tmp_buffers\n");
    exit(1);
  }

  float *h_in_addr;
  checkCudaErrors(cudaMallocHost((void**)&h_in_addr, max_req_size));
  for (int i = 0; i < max_req_size/sizeof(float); ++i) h_in_addr[i] = 1.0f;
  checkCudaErrors(cudaMemcpy(d_in_addr, h_in_addr, max_req_size, cudaMemcpyHostToDevice));
  cudaMemset(d_out_addr, 21, max_resp_size);
  cudaDeviceSynchronize();
  
  app_ctx->h_stub->req = (uint8_t*)d_in_addr;
  app_ctx->h_stub->resp = (uint8_t*)d_out_addr;
  //_mm_mfence();

  app_init(app_ctx);

  uint64_t startNs, endNs;
  int numMetrics = 20000;
  std::vector<uint64_t> metricValues(numMetrics, 0);
  int metricNum = 0;

  printf("Starting work...\n");

  for(int i = 0; i < numMetrics; i++) {
    startNs = getCurNs();
    app_run(app_ctx);
    //if (app_ctx->launch_type == 1 || app_ctx->launch_type == 2 || app_ctx->launch_type == 4) {
    //  checkCudaErrors(cudaStreamSynchronize(app_ctx->work_stream));
    //}
    //else {
    //  while (*ACCESS_ONCE(app_ctx->door_bell) != 2)
    //    ;
    //}
    endNs = getCurNs();
    metricNum = (metricNum + 1) % numMetrics;
    metricValues[metricNum] = endNs - startNs;
    if(i % 10000 == 0) {
      printf("Bench, progress: %d times\n", i);
    }
  }

  printf("ORT local-bench stats(ns) [N, Mean, p90, p95, p99]: %d, %0.2f, %0.2f, %0.2f, %0.2f\n", 
      numMetrics, getMean(metricValues), getPercentile(metricValues, 0.90), 
      getPercentile(metricValues, 0.95), getPercentile(metricValues, 0.99)); 

  app_cleanup(app_ctx);
  
  void *h_out_addr;
  cudaHostAlloc((void**)&h_out_addr, max_req_size, cudaHostAllocPortable);
  cudaMemcpy(h_out_addr, d_out_addr, max_resp_size, cudaMemcpyDefault);
  for(int i = 0 ; i < 5 ; i++) {
    printf("h_addr[%d] = %f\n", i, ((float*)h_out_addr)[i]);
  }

  return 0;
}
