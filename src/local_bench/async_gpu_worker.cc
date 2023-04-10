// Copyright (c) Adithya Kumar, The Pennsylvania State University. All rights reserved.
// Licensed under the MIT License.

#include <assert.h>
#include <stdlib.h>
#include <unistd.h>
#include <stdio.h>
#include <vector>
#include <iostream>
#include <signal.h>

#include "ort_app.h"

#include "debug_utils.h"
#include "time_utils.h"
#include "stats_utils.h"
#include "p2p_buf_pool.h"
#include "p2p_rpc_app_ctx.h"

#include <helper_functions.h>
#include <helper_cuda.h>
#include <emmintrin.h>

#include "concurrentqueue.h"

#include <nvToolsExt.h>

// Globally shared
OrtFacade *AppServer;

extern void stream_sync(AppCtx*);
extern void app_run_notifier(AppCtx*, int);
extern void app_complete_notifier(AppCtx*, int);

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

moodycamel::ConcurrentQueue<void*> launch_queue; 
moodycamel::ConcurrentQueue<void*> ready_queue; 

uint64_t startNs;
int numMetrics = 100;
//int numMetrics = 20;
//std::vector<uint64_t> metricValues(numMetrics, 0);
std::vector<uint64_t> metricValues;
int metricNum = 0;

void CUDART_CB app_completion_cb(void *data) {
  nvtxMark("AppRun complete");
  metricValues.push_back(getCurNs() - startNs);
  metricNum = (metricNum + 1);
  AppCtx *app_ctx = ((AppCtx::AppCallInfo*)data)->app_ctx;
  launch_queue.enqueue(data);
  nvtxMark("Enqueued into launch queue");
  // Do networking related things here
  ready_queue.enqueue((void*)app_ctx);
  nvtxMark("Enqueued into readyQueue");
}

#ifdef PROFILE_MODE
  uint64_t ar_startNs;
  std::vector<uint64_t> ArDelay;
#endif

int app_run(AppCtx *app_ctx, int i_idx=0)
{
#ifdef PROFILE_MODE
  ar_startNs = getCurNs();
#endif
  TRACE_PRINTF("AppRun Start CTX: %p\n", (void*)app_ctx);

  if(app_ctx->launch_type == 1 || app_ctx->launch_type == 5) {
    AppServer->load_data_and_predict(app_ctx->h_stub[i_idx].req, app_ctx->h_stub[i_idx].resp);
  } 
  else if(app_ctx->launch_type == 4) {
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

  TRACE_PRINTF("AppRun complete CTX: %p\n", (void*)app_ctx);
#ifdef PROFILE_MODE
  ArDelay.push_back(getCurNs() - ar_startNs);
#endif
  return 1;
}

void launch_work(AppCtx *app_ctx, int i_idx)
{
  app_run_notifier(app_ctx, i_idx);
  app_run(app_ctx, i_idx);
  app_complete_notifier(app_ctx, i_idx);
  //cudaLaunchHostFunc(app_ctx->work_stream, &app_completion_cb, tag); 
}

void worker_func() 
{
  printf("Starting worker thread\n");
  void *tag = NULL;
  bool itemsLeft = false;

  do {
    itemsLeft = launch_queue.try_dequeue(tag);
    if(itemsLeft) {
      launch_work(((AppCtx::AppCallInfo*)tag)->app_ctx, ((AppCtx::AppCallInfo*)tag)->i_idx);
      if(metricNum >= numMetrics) {
        break;
      }
    }
  } while (1);
  printf("Ending worker thread\n");
}

void completion_listener(AppCtx *app_ctx)
{
  printf("Starting completion listener thread\n");
  int ci_idx = 0;

  do {
    if(ACCESS_ONCE(app_ctx->door_bell[ci_idx]) == 3) {
      nvtxMark("AppRun complete");
      metricValues.push_back(getCurNs() - startNs);
      metricNum = (metricNum + 1);
      launch_queue.enqueue((void*)&app_ctx->call_info[ci_idx]);
      ci_idx = (ci_idx + 1) % app_ctx->max_instances;
      nvtxMark("Enqueued into launch queue");
      // Do networking related things here
      ready_queue.enqueue((void*)app_ctx);
      nvtxMark("Enqueued into readyQueue");
    }
    if(metricNum >= numMetrics + 1) {
      break;
    }
  } while(1);
  printf("Ending completion listener thread\n");
}

void trigger_func()
{
  printf("Starting launcher thread\n");
  void *data = NULL;
  AppCtx *app_ctx;
  int pi_idx = 0;

  bool itemsLeft = false;

  do {
    itemsLeft = ready_queue.try_dequeue(data);
    if(itemsLeft) {
      app_ctx = (AppCtx*)data;
      startNs = getCurNs();
      ACCESS_ONCE(app_ctx->door_bell[pi_idx]) = 1;
      _mm_mfence();
      pi_idx = (pi_idx + 1) % app_ctx->max_instances;
      if(metricNum % 1000 == 0) {
        printf("Progress... %d launched\n", metricNum);
      }
      if(metricNum >= numMetrics + 1) {
        break;
      }
    }
  } while(1);
  printf("Ending launcher thread\n");
}

////// Single threaded version of above 3 functions
void async_worker_loop(AppCtx *app_ctx)
{
  printf("Starting async worker loop thread\n");
  int ci_idx = 0;
  int pi_idx = 0;
  void *data = NULL;
  metricNum = 0;
  do {

    // Consumer action
    if(ACCESS_ONCE(app_ctx->door_bell[ci_idx]) == 3) {
      metricValues.push_back(getCurNs() - startNs);
      nvtxMark("AppRun complete");
      metricNum++;

      // Exit condition
      if(metricNum + app_ctx->max_instances <= numMetrics) {
        launch_work(app_ctx, ci_idx);
      }
      ci_idx = (ci_idx + 1) % app_ctx->max_instances;
      // Do networking related things here
      ready_queue.enqueue((void*)app_ctx);
      nvtxMark("Enqueued into readyQueue");
      //printf("PI: %d, CI: %d\n", pi_idx, ci_idx);
    }

    // Producer action
    if(ready_queue.try_dequeue(data)) {
      //app_ctx = (AppCtx*)data;
      nvtxMark("AppRun start");
      startNs = getCurNs();
      ACCESS_ONCE(app_ctx->door_bell[pi_idx]) = 1;
      _mm_mfence();
      pi_idx = (pi_idx + 1) % app_ctx->max_instances;
      //printf("PI: %d, CI: %d\n", pi_idx, ci_idx);
    }

    // Exit action
    if(metricNum >= numMetrics) {
      break;
    }

  } while(1);

  printf("Ending async worker loop thread\n");
}

int main()
{
  size_t max_req_size = get_req_size();
  size_t max_resp_size = get_resp_size();
  int srv_qlen = 2;

//////////////////////// SETUP device side buffers
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


/////////////////////// SETUP AppCtx
  AppCtx *app_ctx = new AppCtx;
  app_ctx->launch_type = get_work_launch_type();
  app_ctx->device_id = get_cuda_device_id();
  checkCudaErrors(cudaSetDevice(app_ctx->device_id));
  checkCudaErrors(cudaStreamCreateWithFlags(&app_ctx->work_stream, cudaStreamNonBlocking));
  checkCudaErrors(cudaEventCreateWithFlags(&app_ctx->work_complete, cudaEventDisableTiming));

  app_ctx->max_instances = srv_qlen;
  app_ctx->h_stub = new g_params[app_ctx->max_instances];
  //app_ctx->h_stub = BufItemPool<g_params>::create_buf_item_pool(1, app_ctx->device_id);
  //app_ctx->d_stub = BufItemPool<g_params>::get_dev_ptr(app_ctx->h_stub);
  app_ctx->door_bell = BufItemPool<uint32_t>::create_buf_item_pool(app_ctx->max_instances, app_ctx->device_id);
  app_ctx->d_door_bell = BufItemPool<uint32_t>::get_dev_ptr(app_ctx->door_bell);
  checkCudaErrors(cudaMemset(app_ctx->d_door_bell, 0, app_ctx->max_instances * sizeof(uint32_t)));
  
  app_ctx->call_info = new AppCtx::AppCallInfo[app_ctx->max_instances];
  for(int i = 0 ; i < app_ctx->max_instances; i++) {
    app_ctx->h_stub[i].req = (uint8_t*)d_in_addr;
    app_ctx->h_stub[i].resp = (uint8_t*)d_out_addr;
    app_ctx->call_info[i].app_ctx = app_ctx;
    app_ctx->call_info[i].i_idx = i;
  }

  app_init(app_ctx);

  printf("Warming up...\n");
  for(int i = 0; i < 5; i++) {
    app_run(app_ctx);
    stream_sync(app_ctx);
  }

  printf("Starting work... launching first job\n");
  if(app_ctx->launch_type == 1) {
    for(int i = 0 ; i < numMetrics; i++) {
      startNs = getCurNs();
      app_run(app_ctx);
      stream_sync(app_ctx);
      metricValues.push_back(getCurNs() - startNs);
    }
  }
  else if(app_ctx->launch_type == 5) {
    //std::thread t_worker(worker_func);
    //std::thread t_loadgen(trigger_func);
    //std::thread t_listener(completion_listener, app_ctx);

    for(int i = 0 ; i < app_ctx->max_instances; i++) {
      launch_work(app_ctx, i);
      //launch_queue.enqueue((void*)&app_ctx->call_info[i]);
    }

    ready_queue.enqueue((void*)app_ctx);
    ready_queue.enqueue((void*)app_ctx);

    std::thread t_worker(async_worker_loop, app_ctx);

    printf("Waiting for all work to be done...\n");
    stream_sync(app_ctx);
    t_worker.join();
    //t_loadgen.join();
    //t_listener.join();
  }

  printf("ORT local-bench stats(ns) [N, Mean, p90, p95, p99]: %d, %0.2f, %0.2f, %0.2f, %0.2f\n", 
      metricValues.size(), getMean(metricValues), getPercentile(metricValues, 0.90), 
      getPercentile(metricValues, 0.95), getPercentile(metricValues, 0.99)); 

  app_cleanup(app_ctx);
  
  void *h_out_addr;
  checkCudaErrors(cudaHostAlloc((void**)&h_out_addr, max_req_size, cudaHostAllocPortable));
  checkCudaErrors(cudaMemcpy(h_out_addr, d_out_addr, max_resp_size, cudaMemcpyDefault));
  for(int i = 0 ; i < 5 ; i++) {
    //printf("h_addr[%d] = %f\n", i, ((float*)h_out_addr)[i]);
    printf("h_addr[%d] = %d\n", i, ((uint8_t*)h_out_addr)[i]);
  }

  return 0;
}
