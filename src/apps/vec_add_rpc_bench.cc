// Copyright (c) Adithya Kumar, The Pennsylvania State University. All rights reserved.
// Licensed under the MIT License.
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>

#include "vector_add.cuh"
#include "p2p_rpc_async_app_server.h"

int app_complete(AppCtx *)
{return 1;}

AppInitCB AppInit_cb = &app_init;
AppRunCB AppRun_cb = &app_run;
AppCleanupCB AppCleanup_cb = &app_cleanup;
AppCompleteCB AppComplete_cb = &app_complete;
