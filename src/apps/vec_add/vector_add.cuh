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

#include "p2p_rpc_app_ctx.h"

#include "config_utils.h"

extern "C" void kernel_entry(AppCtx *app_ctx);
extern "C" void cdp_entry(AppCtx *app_ctx);
extern "C" void pt_entry(AppCtx *app_ctx);
extern "C" void cuda_graph_entry(AppCtx *app_ctx);

extern size_t VEC_ADD_LEN;

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
  VEC_ADD_LEN = get_req_size();
  ACCESS_ONCE(*(app_ctx->door_bell)) = 0;
  _mm_mfence();

  printf("VECADD init on device: %d, stream: %p\n", app_ctx->device_id, app_ctx->work_stream);
  cudaSetDevice(app_ctx->device_id);

  if(app_ctx->launch_type == 3) {
    pt_entry(app_ctx);
    printf("Launched persistent vec_add kernel\n");
  } 

  printf("VECADD init complete, vec-len = %ld\n", VEC_ADD_LEN);
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
