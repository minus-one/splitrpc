// Copyright (c) Adithya Kumar, The Pennsylvania State University. All rights reserved.
// Licensed under the MIT License.

#include "nnfusion_rt.h"

#include <helper_functions.h>
#include <helper_cuda.h>

#include "config_utils.h"
#include "stats_utils.h"
#include "time_utils.h"
#include "p2p_buf_pool.h"
#include <emmintrin.h>

int app_cleanup(AppCtx *app_ctx)
{
  if(app_ctx->launch_type == 3) {
    ACCESS_ONCE(*(app_ctx->door_bell)) = 3;
    _mm_mfence();
  }
  return 1;
}

int app_init(AppCtx *app_ctx)
{
  printf("AppInit for LSTM\n");
  cudaSetDevice(get_cuda_device_id());
  cuda_init();
  
  printf("CudaInit for LSTM complete\n");
  ACCESS_ONCE(*(app_ctx->door_bell)) = 0;
  _mm_mfence();
  printf("LSTM Loaded model params...\n");

  if(app_ctx->launch_type == 3) {
    pt_entry(app_ctx);
    printf("Launched persistent LSTM dynamic parallelism kernel\n");
  } 
  size_t req_len = get_req_size();
  size_t resp_len = get_resp_size();

  printf("LSTM init complete, req-len = %ld, resp-len = %ld\n", req_len, resp_len);

  return 1;
}

// Do GPU work
int app_run(AppCtx *app_ctx) 
{
  TRACE_PRINTF("LSTM launch Type: %d\n", app_ctx->launch_type);

  if(app_ctx->launch_type == 1)
    kernel_entry(app_ctx);
  else if(app_ctx->launch_type == 2)
    cdp_entry(app_ctx);
  else if(app_ctx->launch_type == 3) {
    ACCESS_ONCE(*(app_ctx->door_bell)) = 1;
    _mm_mfence();
  }
  else if(app_ctx->launch_type == 4) {
    if(!app_ctx->graphCreated) {
      printf("Constructing CUDA graph\n");
      checkCudaErrors(cudaStreamBeginCapture(app_ctx->work_stream, cudaStreamCaptureModeGlobal));
      kernel_entry(app_ctx);
      checkCudaErrors(cudaStreamEndCapture(app_ctx->work_stream, &app_ctx->graph));
      checkCudaErrors(cudaGraphInstantiate(&app_ctx->instance, app_ctx->graph, NULL, NULL, 0));
      app_ctx->graphCreated = true;
    }
    checkCudaErrors(cudaGraphLaunch(app_ctx->instance, app_ctx->work_stream));
  }
  else
    kernel_entry(app_ctx);
  return 1;
}

int main()
{
  AppCtx *app_ctx = new AppCtx;
  app_ctx->launch_type = get_work_launch_type();
  app_ctx->device_id = get_cuda_device_id();
  checkCudaErrors(cudaStreamCreateWithFlags(&app_ctx->work_stream, cudaStreamNonBlocking));
  checkCudaErrors(cudaEventCreateWithFlags(&app_ctx->work_complete, cudaEventDisableTiming));

  app_ctx->h_stub = BufItemPool<g_params>::create_buf_item_pool(1, app_ctx->device_id);
  app_ctx->door_bell = BufItemPool<uint32_t>::create_buf_item_pool(1, app_ctx->device_id);
  app_ctx->d_stub = BufItemPool<g_params>::get_dev_ptr(app_ctx->h_stub);
  app_ctx->d_door_bell = BufItemPool<uint32_t>::get_dev_ptr(app_ctx->door_bell);

  void *d_in_addr, *d_out_addr;
  if (cudaMalloc((void **)&d_in_addr, 2048 * sizeof(float)) != cudaSuccess) {
    printf("CudaMalloc failed, unable to create tmp_buffers\n");
    exit(1);
  }
  if (cudaMalloc((void **)&d_out_addr, 256 * sizeof(float)) != cudaSuccess) {
    printf("CudaMalloc failed, unable to create tmp_buffers\n");
    exit(1);
  }

  float *h_in_addr;
  checkCudaErrors(cudaMallocHost((void**)&h_in_addr, 2048 * sizeof(float)));
  for (int i = 0; i < 2048; ++i) h_in_addr[i] = 1.0f;
  checkCudaErrors(cudaMemcpy(d_in_addr, h_in_addr, 2048 * sizeof(float), cudaMemcpyHostToDevice));
  cudaMemset(d_out_addr, 21, 256 * sizeof(float));
  cudaDeviceSynchronize();
  
  app_ctx->h_stub->req = (uint8_t*)d_in_addr;
  app_ctx->h_stub->resp = (uint8_t*)d_out_addr;
  _mm_mfence();

  app_init(app_ctx);

  uint64_t startNs, endNs;
  int numMetrics = 20000;
  std::vector<uint64_t> metricValues(numMetrics, 0);
  int metricNum = 0;

  printf("Starting work...\n");
  for(int i = 0; i < numMetrics; i++) {
    startNs = getCurNs();
    app_run(app_ctx);
    if (app_ctx->launch_type == 1 || app_ctx->launch_type == 2 || app_ctx->launch_type == 4) {
      checkCudaErrors(cudaStreamSynchronize(app_ctx->work_stream));
    }
    else {
      while (*ACCESS_ONCE(app_ctx->door_bell) != 2)
        ;
    }
    endNs = getCurNs();
    metricNum = (metricNum + 1) % numMetrics;
    metricValues[metricNum] = endNs - startNs;
    if(i % 10000 == 0) {
      printf("Bench, progress: %d times\n", i);
    }
  }

  printf("LSTM local-bench stats(ns) [N, Mean, p90, p95, p99]: %d, %0.2f, %0.2f, %0.2f, %0.2f\n", 
      numMetrics, getMean(metricValues), getPercentile(metricValues, 0.90), 
      getPercentile(metricValues, 0.95), getPercentile(metricValues, 0.99)); 

  app_cleanup(app_ctx);
  
  void *h_out_addr;
  cudaHostAlloc((void**)&h_out_addr, sizeof(float) * 256, cudaHostAllocPortable);
  cudaMemcpy(h_out_addr, d_out_addr, sizeof(float) * 5, cudaMemcpyDefault);
  for(int i = 0 ; i < 5 ; i++) {
    printf("h_addr[%d] = %f\n", i, ((float*)h_out_addr)[i]);
  }

  return 0;
}
