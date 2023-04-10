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

#include "config_utils.h"
#include "stats_utils.h"
#include "udp_common.h"

int app_run(AppCtx *app_ctx)
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

void run_local(void){

  AppCtx new_app_ctx;
  new_app_ctx.launch_type = get_work_launch_type();
  new_app_ctx.device_id = get_cuda_device_id();
  checkCudaErrors(cudaStreamCreateWithFlags(&new_app_ctx.work_stream, cudaStreamNonBlocking));
  checkCudaErrors(cudaEventCreateWithFlags(&new_app_ctx.work_complete, cudaEventDisableTiming));

  new_app_ctx.h_stub = BufItemPool<g_params>::create_buf_item_pool(1, device_id);
  new_app_ctx.door_bell = BufItemPool<uint32_t>::create_buf_item_pool(1, device_id);
  new_app_ctx.d_stub = BufItemPool<g_params>::get_dev_ptr(new_app_ctx.h_stub);
  new_app_ctx.d_door_bell = BufItemPool<uint32_t>::get_dev_ptr(new_app_ctx.door_bell);

  CUDA_SAFE_CALL(cudaMalloc((void**)&new_app_ctx.h_stub->req, sizeof(float) * 2048));
  CUDA_SAFE_CALL(cudaMalloc((void**)&new_app_ctx.h_stub->resp, sizeof(float) * 256));
  
  cudaSetDevice(new_app_ctx.device_id);
  cuda_init();

  ACCESS_ONCE(*(app_ctx->door_bell)) = 0;
  _mm_mfence();
  printf("LSTM Loaded model params...\n");

  if(app_ctx->launch_type == 3) {
    pt_entry(app_ctx);
    printf("Launched persistent LSTM dynamic parallelism kernel\n");
  } 

  //input argument
  float *Parameter_96_0_host;
  CUDA_SAFE_CALL(cudaMallocHost((void**)&Parameter_96_0_host, sizeof(float)* 2048));
  //Parameter_96_0_host = (float*)malloc(sizeof(float) * 2048);

  //output arguments
  float* Result_2261_0_host;
  CUDA_SAFE_CALL(cudaMallocHost((void**)&Result_2261_0_host, sizeof(float) * 256));
  //Result_2261_0_host = (float*)malloc(sizeof(float) * 256);

  // fill input values
  for (int i = 0; i < 2048; ++i) Parameter_96_0_host[i] = 1.0f;
  CUDA_SAFE_CALL(cudaMemcpy(new_app_ctx.h_stub->req, Parameter_96_0_host, sizeof(float) * 2048, cudaMemcpyHostToDevice));


  //warm up for 5 iters:
  for(int i_=0; i_< 5; i_++)
  {
    CUDA_SAFE_CALL(cudaMemcpy(new_app_ctx.h_stub->req, Parameter_96_0_host, sizeof(float) * 2048, cudaMemcpyHostToDevice));

    app_run(&new_app_ctx);

    CUDA_SAFE_CALL(cudaMemcpy(Result_2261_0_host, new_app_ctx.h_stub->resp,  sizeof(float) * 256, cudaMemcpyDeviceToHost));
    //CUDA_SAFE_CALL(cudaDeviceSynchronize()); 
    printf("%s \n", "Result_2261_0:");
    for (int i = 0; i < 10; ++i) printf("%e ", (float)Result_2261_0_host[i]); 
    printf(" .. (size = 256, ends with %e);\n", (float)Result_2261_0_host[255]);
  }

  //GPU time measurement
  float ms_max = std::numeric_limits<float>::min();
  float ms_min = std::numeric_limits<float>::max();
  float ms_total, ms_i;
  cudaEvent_t start, stop, start_i, stop_i;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventCreate(&start_i);
  cudaEventCreate(&stop_i);

  //kernel call
  int steps = 20000;
  int numMetrics = steps;
  std::vector<uint64_t> metricValues(numMetrics, 0);
  int metricNum = 0;

  for (int i_=0; i_<steps; i_++)
  {
    cudaEventRecord(start_i, new_app_ctx.work_stream);
    CUDA_SAFE_CALL(cudaMemcpy(new_app_Ctx.h_stub->req, Parameter_96_0_host, sizeof(float) * 2048, cudaMemcpyHostToDevice));

    app_run(&new_app_ctx);

    CUDA_SAFE_CALL(cudaMemcpy(Result_2261_0_host, new_app_ctx.h_stub->resp,  sizeof(float) * 256, cudaMemcpyDeviceToHost));
    cudaEventRecord(stop_i, new_app_ctx.work_stream);
    cudaEventSynchronize(stop_i);
    cudaEventElapsedTime(&ms_i, start_i, stop_i);
    //printf("Iteration time %f ms\n", ms_i);
    if (ms_i > ms_max)  ms_max = ms_i;
    if (ms_i < ms_min) ms_min = ms_i;

    metricValues[metricNum] = (uint64_t)(ms_i * 1E6);
    metricNum = (metricNum + 1) % numMetrics;
  }

  cudaEventRecord(stop, new_app_ctx.work_stream);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&ms_total, start, stop);
  printf("Summary: [min, max, mean] = [%f, %f, %f] ms\n",  ms_min, ms_max, ms_total/steps);

  printf("Exec time stats(ns) [N, Mean, p90, p95, p99]: %d, %0.2f, %0.2f, %0.2f, %0.2f\n", 
      numMetrics, getMean(metricValues), getPercentile(metricValues, 0.90), 
      getPercentile(metricValues, 0.95), getPercentile(metricValues, 0.99)); 

  //free context
  CUDA_SAFE_CALL(cudaFree(new_app_ctx.h_stub->req));
  CUDA_SAFE_CALL(cudaFree(new_app_ctx.h_stub->resp));
  //CUDA_SAFE_CALL(cudaFree(Parameter_96_0));
  cuda_free();

  cudaFreeHost(Parameter_96_0_host);
  cudaFreeHost(Result_2261_0_host);
}

void run_network(void){
  int udp_sock = initUdpSock(50051, false);
  struct sockaddr_in si_other;

  AppCtx new_app_ctx;
  new_app_ctx.launch_type = get_work_launch_type();
  new_app_ctx.device_id = get_cuda_device_id();
  checkCudaErrors(cudaStreamCreateWithFlags(&new_app_ctx.work_stream, cudaStreamNonBlocking));
  checkCudaErrors(cudaEventCreateWithFlags(&new_app_ctx.work_complete, cudaEventDisableTiming));

  new_app_ctx.h_stub = BufItemPool<g_params>::create_buf_item_pool(1, device_id);
  new_app_ctx.door_bell = BufItemPool<uint32_t>::create_buf_item_pool(1, device_id);
  new_app_ctx.d_stub = BufItemPool<g_params>::get_dev_ptr(new_app_ctx.h_stub);
  new_app_ctx.d_door_bell = BufItemPool<uint32_t>::get_dev_ptr(new_app_ctx.door_bell);

  CUDA_SAFE_CALL(cudaMalloc((void**)&new_app_ctx.h_stub->req, sizeof(float) * 2048));
  CUDA_SAFE_CALL(cudaMalloc((void**)&new_app_ctx.h_stub->resp, sizeof(float) * 256));
  
  cudaSetDevice(new_app_ctx.device_id);
  cuda_init();

  ACCESS_ONCE(*(app_ctx->door_bell)) = 0;
  _mm_mfence();
  printf("LSTM Loaded model params...\n");

  if(app_ctx->launch_type == 3) {
    pt_entry(app_ctx);
    printf("Launched persistent LSTM dynamic parallelism kernel\n");
  } 

  //input argument
  float *Parameter_96_0_host;
  //CUDA_SAFE_CALL(cudaMallocHost((void**)&Parameter_96_0_host, sizeof(float)* 2048));
  Parameter_96_0_host = (float*)malloc(sizeof(float) * 2048);

  //output arguments
  float* Result_2261_0_host;
  //CUDA_SAFE_CALL(cudaMallocHost((void**)&Result_2261_0_host, sizeof(float) * 256));
  Result_2261_0_host = (float*)malloc(sizeof(float) * 256);

  // fill input values
  for (int i = 0; i < 2048; ++i) Parameter_96_0_host[i] = 1.0f;
  CUDA_SAFE_CALL(cudaMemcpy(new_app_ctx.h_stub->req, Parameter_96_0_host, sizeof(float) * 2048, cudaMemcpyHostToDevice));

  //warm up for 5 iters:
  for(int i_=0; i_< 5; i_++)
  {
    CUDA_SAFE_CALL(cudaMemcpy(new_app_ctx.h_stub->req, Parameter_96_0_host, sizeof(float) * 2048, cudaMemcpyHostToDevice));

    app_run(&new_app_ctx);

    CUDA_SAFE_CALL(cudaMemcpy(Result_2261_0_host, new_app_ctx.h_stub->resp,  sizeof(float) * 256, cudaMemcpyDeviceToHost));
    //CUDA_SAFE_CALL(cudaDeviceSynchronize()); 
    printf("%s \n", "Result_2261_0:");
    for (int i = 0; i < 10; ++i) printf("%e ", (float)Result_2261_0_host[i]); 
    printf(" .. (size = 256, ends with %e);\n", (float)Result_2261_0_host[255]);
  }

  while(1)
  {
    getData(udp_sock, &si_other, (void*)Parameter_96_0_host, sizeof(float) * 2048);
    //CUDA_SAFE_CALL(cudaMemcpy(new_app_ctx.h_stub->req, Parameter_96_0_host, sizeof(float) * 2048, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpyAsync(new_app_ctx.h_stub->req, Parameter_96_0_host, sizeof(float) * 2048, cudaMemcpyHostToDevice, new_app_ctx.work_stream));
    cudaStreamSynchronize(new_cpp_ctx.work_stream);

    app_run(&new_app_ctx);

    //CUDA_SAFE_CALL(cudaMemcpy(Result_2261_0_host, new_app_ctx.h_stub->resp,  sizeof(float) * 256, cudaMemcpyDeviceToHost));
    CUDA_SAFE_CALL(cudaMemcpyAsync(Result_2261_0_host, new_app_ctx.h_stub->resp,  sizeof(float) * 256, cudaMemcpyDeviceToHost, new_app_ctx.work_stream));
    cudaStreamSynchronize(new_app_ctx.work_stream);

    sendData(udp_sock, &si_other, 50052, (void*)Result_2261_0_host, sizeof(float) * 256);
  }

  //free context
  //CUDA_SAFE_CALL(cudaFree(Parameter_96_0));
  CUDA_SAFE_CALL(cudaFree(new_app_ctx.h_stub->req));
  CUDA_SAFE_CALL(cudaFree(new_app_ctx.h_stub->resp));
  cuda_free();
  //cudaFreeHost(Parameter_96_0_host);
  //cudaFreeHost(Result_2261_0_host);
}

// 0 = RunLocal, 1 = RunNetwork
static inline int8_t get_benchmark_type() {
  return readEnvInfo<int16_t>("P2P_RPC_BENCHMARK_TYPE", 0);
}

int main()
{
  int8_t bench_type = get_benchmark_type();
  if(bench_type == 0) {
    printf("Running localbench.....\n");
    run_local();
  } else if(bench_type == 1) {
    printf("Networkbench, starting UDP server (remember to start client).....\n");
    run_network();
  }
  return 0;
}
