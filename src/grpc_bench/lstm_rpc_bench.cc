// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.
// Copyright (c) Adithya Kumar, The Pennsylvania State University. All rights reserved.
// Licensed under the MIT License.

#include "nnfusion_rt.h"
#include <cuda_profiler_api.h>
#include <limits>
#include <stdexcept>
#include <sstream>
#include <stdio.h>
#include <stdlib.h>
#include <signal.h>
#include <emmintrin.h>

#define CUDA_SAFE_CALL(x)                                                                          \
    do                                                                                             \
    {                                                                                              \
        cudaError_t result = (x);                                                                  \
        if (result != cudaSuccess)                                                                 \
        {                                                                                          \
            const char* msg = cudaGetErrorString(result);                                          \
            std::stringstream safe_call_ss;                                                        \
            safe_call_ss << "\nerror: " #x " failed with error"                                    \
                         << "\nfile: " << __FILE__ << "\nline: " << __LINE__ << "\nmsg: " << msg;  \
            throw std::runtime_error(safe_call_ss.str());                                          \
        }                                                                                          \
    } while (0)

#include <helper_functions.h>
#include <helper_cuda.h>

#include "config_utils.h"
#include "stats_utils.h"
#include "grpc_handler.h"
#include "p2p_buf_pool.h"

volatile bool force_quit = 0;
static void 
signal_handler(int signum) {                                                                          
  if (signum == SIGINT) {        
      printf("\n\nSignal %d received, preparing to exit...\n", signum);      
      ACCESS_ONCE(force_quit) = 1;
    }
}

AppCtx new_app_ctx;
size_t inputLen = 8192;
size_t outputLen = 1024;
#ifdef PROFILE_MODE
    uint64_t SStartNs, GStartNs;
    std::vector<uint64_t> SDelay, GDelay;
#endif

inline int app_run(AppCtx *app_ctx)
{
  if(app_ctx->launch_type == 1)
    kernel_entry(app_ctx);
  else if(app_ctx->launch_type == 2)
    cdp_entry(app_ctx);
  else if(app_ctx->launch_type == 3) {
    ACCESS_ONCE(*(app_ctx->door_bell)) = 1;
    _mm_mfence();
  }
  else if(app_ctx->launch_type == 4)
    cuda_graph_entry(app_ctx);
  else
    kernel_entry(app_ctx);
  return 1;
}


// GRPC Callback
size_t appRun(void *req, void *resp)
{
#ifdef PROFILE_MODE
      GStartNs = getCurNs();
#endif

  checkCudaErrors(cudaMemcpyAsync(new_app_ctx.h_stub->req, req, inputLen, cudaMemcpyHostToDevice, new_app_ctx.work_stream));
  checkCudaErrors(cudaStreamSynchronize(new_app_ctx.work_stream));
#ifdef PROFILE_MODE
      GDelay.push_back(getCurNs() - GStartNs);
#endif

  app_run(&new_app_ctx);
#ifdef PROFILE_MODE
  cudaStreamSynchronize(new_app_ctx.work_stream);
#endif

#ifdef PROFILE_MODE
      SStartNs = getCurNs();
#endif
  checkCudaErrors(cudaMemcpyAsync(resp, new_app_ctx.h_stub->resp, outputLen, cudaMemcpyDeviceToHost, new_app_ctx.work_stream));
  checkCudaErrors(cudaStreamSynchronize(new_app_ctx.work_stream));
#ifdef PROFILE_MODE
      SDelay.push_back(getCurNs() - SStartNs);
#endif
  return outputLen;
}

void setup_ctx()
{
  new_app_ctx.launch_type = get_work_launch_type();
  new_app_ctx.device_id = get_cuda_device_id();
  checkCudaErrors(cudaStreamCreateWithFlags(&new_app_ctx.work_stream, cudaStreamNonBlocking));
  checkCudaErrors(cudaEventCreateWithFlags(&new_app_ctx.work_complete, cudaEventDisableTiming));

  new_app_ctx.h_stub = BufItemPool<g_params>::create_buf_item_pool(1, new_app_ctx.device_id);
  new_app_ctx.door_bell = BufItemPool<uint32_t>::create_buf_item_pool(1, new_app_ctx.device_id);
  new_app_ctx.d_stub = BufItemPool<g_params>::get_dev_ptr(new_app_ctx.h_stub);
  new_app_ctx.d_door_bell = BufItemPool<uint32_t>::get_dev_ptr(new_app_ctx.door_bell);

  checkCudaErrors(cudaMalloc((void**)&new_app_ctx.h_stub->req, sizeof(float) * 2048));
  checkCudaErrors(cudaMalloc((void**)&new_app_ctx.h_stub->resp, sizeof(float) * 256));

  cudaSetDevice(new_app_ctx.device_id);
  cuda_init();

  //ACCESS_ONCE(*(app_ctx->door_bell)) = 0;
  //_mm_mfence();
  printf("LSTM Loaded model params...\n");
}

void run_grpc()
{
  //if(app_ctx->launch_type == 3) {
  //  pt_entry(app_ctx);
  //  printf("Launched persistent LSTM dynamic parallelism kernel\n");
  //} 
  std::string m_uri = get_server_ip() + std::string(":") + get_server_port();
  GrpcAsyncRequestHandler reqH(1, m_uri);
  reqH.BuildServer();
  reqH.Run();
  while(ACCESS_ONCE(force_quit) == 0);
  printf("Stopping Async Handler\n");
  reqH.quit();
}

int main()
{
  // Install signal handlers to quit
  signal(SIGINT, signal_handler);

  setup_ctx();
  printf("Doing test runs locally...\n");
  
  void *tmp_h_in = aligned_alloc(sizeof(float), 2048 * sizeof(float));
  void *tmp_h_out = aligned_alloc(sizeof(float), 256 * sizeof(float));

  for(int i = 0 ; i < 10 ; i++) {
    printf("Test run %d\n", i);
    for(int k = 0 ; k < 2048 ; k++) {
     ((float*)tmp_h_in)[i] = 1.0f;
    }
    //std::memset(tmp_h_in, i, inputLen);
    std::memset(tmp_h_out, 0, outputLen);

    appRun(tmp_h_in, tmp_h_out);
    for(int j = 0 ; j < 5 ; j++) {                                      
        printf("Int: tmp_h_out[%d] = %d\n", j, ((uint8_t*)tmp_h_out)[j]);         
    }                                                                   
    printf("Int: tmp_h_out[-2] = %d\n", ((uint8_t*)tmp_h_out)[outputLen-2]);
    printf("Int: tmp_h_out[-1] = %d\n", ((uint8_t*)tmp_h_out)[outputLen-1]);
    printf("================================================================\n");
  }

  free(tmp_h_in);
  free(tmp_h_out);

  printf("GrpcBench, Starting server (remember to start client)...\n");
  run_grpc();
  
#ifdef PROFILE_MODE
  printPbStats();
  PROF_PRINT("H2D", GDelay);
  PROF_PRINT("D2H", SDelay);
#endif
  CUDA_SAFE_CALL(cudaFree(new_app_ctx.h_stub->req));
  CUDA_SAFE_CALL(cudaFree(new_app_ctx.h_stub->resp));

  return 0;
}
