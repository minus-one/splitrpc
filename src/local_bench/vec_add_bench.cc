// Copyright (c) Adithya Kumar, The Pennsylvania State University. All rights reserved.
// Licensed under the MIT License.

#include "config_utils.h"
#include "stats_utils.h"
#include "time_utils.h"

#include "p2p_rpc_app_ctx.h"
#include "vector_add.cuh"
#include "p2p_buf_pool.h"

size_t inputLen = get_req_size();
size_t outputLen = get_resp_size();

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
  if (cudaMalloc((void **)&d_in_addr, inputLen) != cudaSuccess) {
    printf("CudaMalloc failed, unable to create tmp_buffers\n");
    exit(1);
  }
  if (cudaMalloc((void **)&d_out_addr, outputLen) != cudaSuccess) {
    printf("CudaMalloc failed, unable to create tmp_buffers\n");
    exit(1);
  }
  cudaMemset(d_in_addr, 1, inputLen);

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

  printf("VecAdd local-bench stats(ns) [N, Mean, p90, p95, p99]: %d, %0.2f, %0.2f, %0.2f, %0.2f\n", 
      numMetrics, getMean(metricValues), getPercentile(metricValues, 0.90), 
      getPercentile(metricValues, 0.95), getPercentile(metricValues, 0.99)); 

  app_cleanup(app_ctx);

  void *h_out_addr;
  cudaHostAlloc((void**)&h_out_addr, outputLen, cudaHostAllocPortable);
  cudaMemcpy(h_out_addr, d_out_addr, outputLen, cudaMemcpyDefault);
  for(int i = 0 ; i < 5 ; i++) {
    printf("Int: h_out_addr[%d] = %d\n", i, ((uint8_t*)h_out_addr)[i]);
  }
  for(int i = 0 ; i < 5 ; i++) {
    printf("Float: h_out_addr[%d] = %f\n", i, ((float*)h_out_addr)[i]);
  }

  return 0;
}
