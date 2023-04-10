// Copyright (c) Adithya Kumar, The Pennsylvania State University. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include <stdio.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/fcntl.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <nvToolsExt.h>

#define CUDA_DRIVER_API
#include <helper_functions.h>
#include <helper_cuda.h>
#include <emmintrin.h> 

#include "config_utils.h"
#include "time_utils.h"
#include "stats_utils.h"

#include "p2p_rpc_app_ctx.h"
extern "C" void kernel_entry(AppCtx *app_ctx);
extern "C" void cdp_entry(AppCtx *app_ctx);
extern "C" void pt_entry(AppCtx *app_ctx);
extern "C" void cuda_graph_entry(AppCtx *app_ctx);

void cuda_init();

static int app_cleanup(AppCtx *app_ctx)
{
  if(app_ctx->launch_type == 3) {
    ACCESS_ONCE(*(app_ctx->door_bell)) = 3;
    _mm_mfence();
  }
  return 1;
}

static int app_init(AppCtx *app_ctx)
{
  cudaSetDevice(app_ctx->device_id);
  cuda_init();
  if(app_ctx->launch_type == 1) {
    printf("Launching kernels from host\n");
  } else if(app_ctx->launch_type == 2) {
    printf("Launching kernels using dynamic parallelism\n");
  } else if(app_ctx->launch_type == 3) {
    pt_entry(app_ctx);
    printf("Launched persistent LeNet dynamic parallelism kernel\n");
  } else if(app_ctx->launch_type == 4) {
    printf("Launching cuda graphs\n");
  }

  printf("LeNet: Loaded all data, app_init complete\n");
  return 1;
}

static int app_run(AppCtx *app_ctx) 
{
  if(app_ctx->launch_type == 1) {
    kernel_entry(app_ctx);
  } else if(app_ctx->launch_type == 2) {
    cdp_entry(app_ctx);
  } else if(app_ctx->launch_type == 3) {
    ACCESS_ONCE(*(app_ctx->door_bell)) = 1;
    _mm_mfence();
  } else if(app_ctx->launch_type == 4) {
    cuda_graph_entry(app_ctx);
  } else {
    kernel_entry(app_ctx);
  }

  return 1;
}
