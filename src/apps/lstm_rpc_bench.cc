// Copyright (c) Adithya Kumar, The Pennsylvania State University. All rights reserved.
// Licensed under the MIT License.
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>

#include "nnfusion_rt.h"
#include "p2p_rpc_async_app_server.h"

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
  else if(app_ctx->launch_type == 4)
    cuda_graph_entry(app_ctx);
  else
    kernel_entry(app_ctx);
  return 1;
}

int app_complete(AppCtx *)
{return 1;}

AppInitCB AppInit_cb = &app_init;
AppRunCB AppRun_cb = &app_run;
AppCleanupCB AppCleanup_cb = &app_cleanup;
AppCompleteCB AppComplete_cb = &app_complete;
