// Copyright (c) Adithya Kumar, The Pennsylvania State University. All rights reserved.
// Licensed under the MIT License.

#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <vector>
#include <iostream>
#include <signal.h>

#include "time_utils.h"
#include "config_utils.h"
#include "stats_utils.h"

#include "grpc_handler.h"
#include "lenet_vanilla.cuh"

#include "p2p_buf_pool.h"

size_t inputLen = get_req_size(); 
size_t outputLen = get_resp_size();
#ifdef PROFILE_MODE
    uint64_t SStartNs, GStartNs;
    std::vector<uint64_t> SDelay, GDelay;
#endif

AppCtx *app_ctx = NULL;

volatile bool force_quit = 0;
static void 
signal_handler(int signum) {                                                                          
  if (signum == SIGINT) {        
      printf("\n\nSignal %d received, preparing to exit...\n", signum);      
      ACCESS_ONCE(force_quit) = 1;
    }
}

size_t appRun(void *req, void *resp)
{
#ifdef PROFILE_MODE
      GStartNs = getCurNs();
#endif
  cudaMemcpyAsync(app_ctx->h_stub->req, req, inputLen, cudaMemcpyHostToDevice, app_ctx->work_stream);
  cudaStreamSynchronize(app_ctx->work_stream);
#ifdef PROFILE_MODE
      GDelay.push_back(getCurNs() - GStartNs);
#endif
  app_run(app_ctx);
#ifdef PROFILE_MODE
  cudaStreamSynchronize(app_ctx->work_stream);
#endif
#ifdef PROFILE_MODE
      SStartNs = getCurNs();
#endif
  cudaMemcpyAsync(resp, app_ctx->h_stub->resp, outputLen, cudaMemcpyDeviceToHost, app_ctx->work_stream);
  cudaStreamSynchronize(app_ctx->work_stream);
#ifdef PROFILE_MODE
      SDelay.push_back(getCurNs() - SStartNs);
#endif

  return outputLen;
}

void run_grpc()
{
  app_ctx = new AppCtx;
  app_ctx->launch_type = get_work_launch_type();
  app_ctx->device_id = get_cuda_device_id();
  checkCudaErrors(cudaStreamCreateWithFlags(&app_ctx->work_stream, cudaStreamNonBlocking));
  checkCudaErrors(cudaEventCreateWithFlags(&app_ctx->work_complete, cudaEventDisableTiming));
  app_ctx->h_stub = BufItemPool<g_params>::create_buf_item_pool(1, app_ctx->device_id);
  app_ctx->d_stub = BufItemPool<g_params>::get_dev_ptr(app_ctx->h_stub);
  app_ctx->door_bell = BufItemPool<uint32_t>::create_buf_item_pool(1, app_ctx->device_id);
  app_ctx->d_door_bell = BufItemPool<uint32_t>::get_dev_ptr(app_ctx->door_bell);

  if (cudaMalloc((void **)&app_ctx->h_stub->req, inputLen) != cudaSuccess) {
    printf("CudaMalloc failed, unable to create tmp_buffers\n");
    exit(1);
  }
  if (cudaMalloc((void **)&app_ctx->h_stub->resp, outputLen) != cudaSuccess) {
    printf("CudaMalloc failed, unable to create tmp_buffers\n");
    exit(1);
  }

  app_init(app_ctx);

  printf("LenetAppCtx setup: device_id: %d, InputLen: %ld, OutputLen: %ld\n", app_ctx->device_id, inputLen, outputLen);

  std::string m_uri = get_server_ip() + std::string(":") + get_server_port();
  GrpcAsyncRequestHandler reqH(1, m_uri);
  reqH.BuildServer();
  reqH.Run();
  while(ACCESS_ONCE(force_quit) == 0);
  printf("Stopping Async Handler\n");
  reqH.quit();

  app_cleanup(app_ctx);
}

int main()
{
  // Install signal handlers to quit
  signal(SIGINT, signal_handler);

  printf("GrpcBench, Starting server (remember to start client)...\n");
  run_grpc();
  
#ifdef PROFILE_MODE
  printPbStats();
  PROF_PRINT("H2D", GDelay);
  PROF_PRINT("D2H", SDelay);
#endif
  return 0;
}

