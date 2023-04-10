// Copyright (c) Adithya Kumar, The Pennsylvania State University. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cuda_runtime.h>
#include <helper_functions.h>
#include <helper_cuda.h>
#include <emmintrin.h> 
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <stdint.h>
#include <emmintrin.h> 

#include "p2p_bufs.h"
#include "p2p_buf_pool.h"
#include "config_utils.h"

#include "p2p_rpc_app_ctx.h"

#ifdef __cplusplus                                                             
extern "C" {                                                                   
#endif                                                                         

  typedef p2p_sk_buf g_copy_params;
  typedef MetaAppCtx<g_copy_params> CopyCtx;
  
  int
    sg_on_gpu_entry(CopyCtx *ctx);

  int
    sg_on_gpu(CopyCtx *ctx,
        p2p_sk_buf *skb);

#ifdef __cplusplus                                                             
}                                                                              
#endif                                                                         
