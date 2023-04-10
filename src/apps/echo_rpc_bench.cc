// Copyright (c) Adithya Kumar, The Pennsylvania State University. All rights reserved.
// Licensed under the MIT License.

#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <vector>
#include <iostream>
#include <signal.h>

#include "p2p_rpc_async_app_server.h"
#include "debug_utils.h"

//#ifdef PROFILE_MODE
//#include <nvToolsExt.h>
//#endif
//#include <cuda.h>
//#include <cuda_runtime.h>
//#include <cuda_runtime_api.h>

// Globally shared

//size_t appRun(void *req, void *resp)
size_t app_resp_size;
int app_run(AppCtx *app_ctx)
{
  TRACE_PRINTF("AppRun Start CTX: %p\n", (void*)app_ctx);
  checkCudaErrors(
      cudaMemcpyAsync(
        (void*)app_ctx->h_stub->resp, 
        (void*)app_ctx->h_stub->req, 
        app_resp_size, 
        cudaMemcpyDeviceToDevice, app_ctx->work_stream));
  TRACE_PRINTF("AppRun complete CTX: %p\n", (void*)app_ctx);
  return 1;
}

int app_init(AppCtx *)
{
  TRACE_PRINTF("EchoAppInit\n"); 
  app_resp_size = get_resp_size();
  return 1;
}

int app_cleanup(AppCtx *)
{
  return 1;
}

int app_complete(AppCtx *)
{return 1;}

AppInitCB AppInit_cb = &app_init;
AppRunCB AppRun_cb = &app_run;
AppCleanupCB AppCleanup_cb = &app_cleanup;
AppCompleteCB AppComplete_cb = &app_complete;
