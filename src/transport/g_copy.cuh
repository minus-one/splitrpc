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

typedef MetaAppCtx<g_copy_params> CopyCtx;
                                                                               
CopyCtx* 
init_copy_ctx();

CopyCtx*
init_copy_ctx_on_stream(cudaStream_t);

int
sg_on_gpu(CopyCtx *ctx,
    p2p_sk_buf *skb,
    int instance=0);

int
gather_on_gpu(CopyCtx *ctx,
    p2p_bufs *buf_ptrs,
    uint8_t *io_buf);

int 
scatter_on_gpu(CopyCtx *ctx,
    p2p_bufs *buf_ptrs,
    uint8_t *io_buf,
    int io_buf_size);

int
gather_on_gpu_sync(CopyCtx *ctx,
    p2p_bufs *buf_ptrs,
    uint8_t *io_buf);

//cudaError_t                                                                    
//RunGatherKernel(                                                               
//    p2p_bufs *input_bufs,
//    size_t *byte_offsets, 
//    size_t request_count,
//    uint8_t *output_buffer,              
//    cudaStream_t stream); 
//
//cudaError_t                                                                    
//RunScatterKernel(                                                               
//    uint8_t *input_buffer,
//    size_t *byte_offsets, 
//    p2p_bufs *input_bufs,
//    size_t request_count,
//    cudaStream_t stream);

void SetDummyData(void *start_addr, int len, uint8_t dummy_value);

#ifdef __cplusplus                                                             
}                                                                              
#endif                                                                         
