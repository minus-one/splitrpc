// Copyright (c) Adithya Kumar, The Pennsylvania State University. All rights reserved.
// Licensed under the MIT License.

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#define CUDA_DRIVER_API
#include <helper_cuda.h>
#include <helper_functions.h>
#include <emmintrin.h>

#include "p2p_rpc_app_ctx.h"

__launch_bounds__(1) __global__ void work_notifier(volatile uint32_t *door_bell) 
{
  // Get thread ID.
  uint32_t wait_status;
  while (1) {
    wait_status = ACCESS_ONCE(*(door_bell));
    if(wait_status == 1) {
      // Signal work has started
      *door_bell = 2;
      __threadfence_system();
      break;
    }
  }
}

__launch_bounds__(1) __global__ void completion_notifier(volatile uint32_t *door_bell)
{
  // Signal work to be complete
  ACCESS_ONCE(*(door_bell)) = 3;
  __threadfence_system();
}

void stream_sync(AppCtx *app_ctx)
{
  if (app_ctx->launch_type == 1 || app_ctx->launch_type == 2 || 
      app_ctx->launch_type == 4 || app_ctx->launch_type == 5) {
    checkCudaErrors(cudaStreamSynchronize(app_ctx->work_stream));
  }
  else if(app_ctx->launch_type == 3) {
    while (*ACCESS_ONCE(app_ctx->door_bell) != 2)
      ;
  }
}

void app_run_notifier(AppCtx *app_ctx, int i_idx = 0)
{
  work_notifier<<<1, 1, 0, app_ctx->work_stream>>>(&app_ctx->d_door_bell[i_idx]);
  //checkCudaErrors(cuStreamWaitValue32(app_ctx->work_stream, (CUdeviceptr)&app_ctx->d_door_bell[i_idx], 1, 0));
  //checkCudaErrors(cuStreamWriteValue32(app_ctx->work_stream, (CUdeviceptr)&app_ctx->d_door_bell[i_idx], 2, 0));
}

void app_complete_notifier(AppCtx *app_ctx, int i_idx = 0)
{
  completion_notifier<<<1, 1, 0, app_ctx->work_stream>>>(&app_ctx->d_door_bell[i_idx]);
  //checkCudaErrors(cuStreamWriteValue32(app_ctx->work_stream, (CUdeviceptr)&app_ctx->d_door_bell[i_idx], 3, 0));
}
