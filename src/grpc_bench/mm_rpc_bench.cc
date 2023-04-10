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

#include "grpc_handler.h"
#include "p2p_buf_pool.h"

#include "mm.cuh"

size_t inputLen = get_req_size();
size_t outputLen = get_resp_size();
AppCtx *app_ctx = NULL;

volatile bool force_quit = 0;
static void 
signal_handler(int signum) {                                                                          
  if (signum == SIGINT) {        
      printf("\n\nSignal %d received, preparing to exit...\n", signum);      
      ACCESS_ONCE(force_quit) = 1;
    }
}

void *d_addr, *h_addr;
void *ctx;

size_t appRun(void *req, void *resp)
{
  cudaMemcpyAsync(app_ctx->h_stub->req, req, inputLen, cudaMemcpyHostToDevice, app_ctx->work_stream);
  cudaStreamSynchronize(app_ctx->work_stream);
  app_run(app_ctx);
  cudaStreamSynchronize(app_ctx->work_stream);
  cudaMemcpyAsync(resp, app_ctx->h_stub->resp, outputLen, cudaMemcpyDeviceToHost, app_ctx->work_stream);
  cudaStreamSynchronize(app_ctx->work_stream);

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

  printf("MatMul setup: device_id: %d, InputLen: %ld, OutputLen: %ld\n", app_ctx->device_id, inputLen, outputLen);

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
#endif
  return 0;
}

