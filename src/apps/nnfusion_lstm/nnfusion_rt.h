// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.
// Copyright (c) Adithya Kumar, The Pennsylvania State University. All rights reserved.
// Licensed under the MIT License.

#pragma once

typedef signed char int8_t;
typedef signed short int16_t;
typedef signed int int32_t;
typedef signed long int int64_t;
typedef unsigned char uint8_t;
typedef unsigned short uint16_t;
typedef unsigned int uint32_t;
typedef unsigned long int uint64_t;

#include <cuda.h>
#include <cuda_runtime.h>

#define CUDA_DRIVER_API
#include <helper_functions.h>
#include <helper_cuda.h>

extern "C" void cuda_init();
extern "C" void cuda_free();

#include "p2p_rpc_app_ctx.h"

extern "C" void kernel_entry(AppCtx *app_ctx);
extern "C" void cdp_entry(AppCtx *app_ctx);
extern "C" void pt_entry(AppCtx *app_ctx);
extern "C" void cuda_graph_entry(AppCtx *app_ctx);
