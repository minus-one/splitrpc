// Copyright (c) Adithya Kumar, The Pennsylvania State University. All rights reserved.
// Licensed under the MIT License.

#include "g_copy_ng.cuh"
#include "p2p_rpc.h"
#include "config_utils.h"
#ifdef PROFILE_MODE
#include <nvToolsExt.h>
#endif

#include "g_utils.cuh"

#define MAX_QUEUE_SIZE 32

#define SCATTER_GATHER_TB_SZ 1024

__launch_bounds__(SCATTER_GATHER_TB_SZ) __global__ void CopyKernelSingleTB(        
    g_copy_params *_stub,
    volatile uint32_t *door_bell)
{
  // Get thread ID.
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  uint32_t wait_status;
  __shared__ uint32_t wait_status_shared[1];
  //__shared__ struct g_copy_params call_params_shared;

  __syncthreads();
  while(1) {
    // Wait for work to be ready
    if (tid == 0) {
      while (1) {
        wait_status = ACCESS_ONCE(*(door_bell));
        if(wait_status == 1 || wait_status == 3) {
          wait_status_shared[0] = wait_status;
          //call_params_shared.req = call_params->req;
          //call_params_shared.resp = call_params->resp;
          __threadfence_block();
          break;
        }
      }
    } 
    __syncthreads();

    if (wait_status_shared[0] != 1 && wait_status_shared[0] != 2)
      break;

    const uintptr_t* __restrict i_buf = _stub->i_buf; 
    uintptr_t* __restrict o_buf = _stub->o_buf;
    const size_t* __restrict len = _stub->len;

    int lane_id = threadIdx.x;
    uint8_t* __restrict input_buffer;
    size_t byte_size;
    uint8_t* __restrict output_buffer;

    for(int buf_idx = 0; buf_idx < _stub->num_items ; buf_idx++) {
      input_buffer = (uint8_t*)i_buf[buf_idx];
      byte_size = len[buf_idx];
      output_buffer = (uint8_t*)o_buf[buf_idx];

      if (((byte_size % 4) == 0) && (((uint64_t)input_buffer % 4) == 0) &&
          (((uint64_t)output_buffer % 4) == 0)) {
        int32_t* input_4 = (int32_t*)input_buffer;
        int32_t* output_4 = (int32_t*)output_buffer;
        int element_count = byte_size / 4;
        for (int elem_id = lane_id; elem_id < element_count;
            elem_id += SCATTER_GATHER_TB_SZ) {
          output_4[elem_id] = input_4[elem_id];
        }
      } else {
        for (int elem_id = lane_id; elem_id < byte_size;                           
            elem_id += SCATTER_GATHER_TB_SZ) {                                        
          output_buffer[elem_id] =                                     
            __ldg(input_buffer + elem_id);                               
        }
      }
    }
    __threadfence();
    __syncthreads();

    // Signal work to be complete
    if (tid == 0) {
      ACCESS_ONCE(*(door_bell)) = 2;
      __threadfence_system();
    }
  }

/*
  for(int buf_idx = 0 ; buf_idx < _stub->num_items ; buf_idx++) {
    const uint8_t* input_buffer = (uint8_t*)i_buf[buf_idx];
    int byte_size = len[buf_idx];
    uint8_t* output_buffer = (uint8_t*)o_buf[buf_idx];

     if (((byte_size % 4) == 0) && (((uint64_t)input_buffer % 4) == 0) &&
        (((uint64_t)output_buffer % 4) == 0)) {
      int32_t* input_4 = (int32_t*)input_buffer;
      int32_t* output_4 = (int32_t*)output_buffer;
      int element_count = byte_size / 4;
      for (int elem_id = lane_id; elem_id < element_count;
          elem_id += SCATTER_GATHER_TB_SZ) {
        output_4[elem_id] = input_4[elem_id];
      }
    } else {
      for (int elem_id = lane_id; elem_id < byte_size;                           
          elem_id += SCATTER_GATHER_TB_SZ) {                                        
        output_buffer[elem_id] =                                     
          __ldg(input_buffer + elem_id);                               
      }
    }
  }
  */
}

__launch_bounds__(SCATTER_GATHER_TB_SZ) __global__ void CopyKernel(        
    g_copy_params *_stub,
    volatile uint32_t *door_bell)
{
  const uintptr_t* __restrict i_buf = _stub->i_buf; 
  uintptr_t* __restrict o_buf = _stub->o_buf;
  const size_t* __restrict len = _stub->len;

  int buf_idx = blockIdx.x;
  const uint8_t* input_buffer = (uint8_t*)i_buf[buf_idx];
  size_t byte_size = len[buf_idx];
  uint8_t* output_buffer = (uint8_t*)o_buf[buf_idx];
   
  int lane_id = threadIdx.x;
  if (((byte_size % 4) == 0) && (((uint64_t)input_buffer % 4) == 0) &&
      (((uint64_t)output_buffer % 4) == 0)) {
    int32_t* input_4 = (int32_t*)input_buffer;
    int32_t* output_4 = (int32_t*)output_buffer;
    int element_count = byte_size / 4;
    for (int elem_id = lane_id; elem_id < element_count;
         elem_id += SCATTER_GATHER_TB_SZ) {
      output_4[elem_id] = input_4[elem_id];
    }
  } else {
    for (int elem_id = lane_id; elem_id < byte_size;                           
         elem_id += SCATTER_GATHER_TB_SZ) {                                        
      output_buffer[elem_id] =                                     
          __ldg(input_buffer + elem_id);                               
    }
  }
}

// Launches a kernel from the device side to do cudamemcpyasync
__launch_bounds__(SCATTER_GATHER_TB_SZ) __global__ void CopyKernelAsync(        
    g_copy_params *_stub)
{
  const uintptr_t* __restrict i_buf = _stub->i_buf; 
  const size_t* __restrict len = _stub->len;
  uintptr_t* __restrict o_buf = _stub->o_buf;

  int buf_idx = threadIdx.x;
  //int buf_idx = blockIdx.x;
  const uint8_t* input_buffer = (uint8_t*)i_buf[buf_idx];
  size_t byte_size = len[buf_idx];
  uint8_t* output_buffer = (uint8_t*)o_buf[buf_idx];

  cudaStream_t s;
  cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
  cudaMemcpyAsync((void*)output_buffer, (void*)input_buffer, byte_size, cudaMemcpyDeviceToDevice, s);
}

#ifdef __cplusplus                                                             
extern "C" {                                                                   
#endif                                                                         

int
sg_on_gpu_entry(CopyCtx *ctx)
{
  ACCESS_ONCE(*(ctx->door_bell)) = 0;
  _mm_mfence();
  if(ctx->launch_type == 1) {
    TRACE_PRINTF("Init SG: CopyCtx: %p, Launching PT\n", (void*)ctx);
    CopyKernelSingleTB<<<1, SCATTER_GATHER_TB_SZ, 0, ctx->work_stream>>>(
        (g_copy_params*)(ctx->d_stub), ctx->d_door_bell);
  }
  cudaError_t err = cudaGetLastError();
  if ( err != cudaSuccess ) {
    printf("CUDA Error: %s\n", cudaGetErrorString(err));
    return 0;
  }
  return 1;
}

int
sg_on_gpu(CopyCtx *ctx, p2p_sk_buf *skb)
{
  TRACE_PRINTF("SG: CopyCtx: %p, SKB: %p\n", (void*)ctx, (void*)skb);
  if(ctx == NULL)
    return 0;

  if(ctx->launch_type == 0) {
    for(int i = 0; i < skb->num_items; i++) {
      checkCudaErrors(
          cudaMemcpyAsync((void*)skb->o_buf[i], (void*)skb->i_buf[i], 
            skb->len[i], cudaMemcpyDeviceToDevice, ctx->work_stream));
    }
  } else if(ctx->launch_type == 1) {
    std::memcpy((void*)(ctx->h_stub), skb, sizeof(g_copy_params));
    _mm_mfence();
    ACCESS_ONCE(*(ctx->door_bell)) = 1;
    _mm_mfence();
    //CopyKernelSingleTB<<< 1, SCATTER_GATHER_TB_SZ, 0, ctx->work_stream>>>(
    //    (g_copy_params*)(ctx->d_stub), ctx->d_door_bell);
  } else if(ctx->launch_type == 2) {
    std::memcpy(ctx->h_stub, skb, sizeof(g_copy_params));
    _mm_mfence();
    CopyKernel<<<skb->num_items, SCATTER_GATHER_TB_SZ, 0, ctx->work_stream>>>(
        ctx->d_stub, ctx->d_door_bell);
  } else if(ctx->launch_type == 3) {
    std::memcpy(ctx->h_stub, skb, sizeof(g_copy_params));
    _mm_mfence();
    CopyKernelAsync<<<1, skb->num_items, 0, ctx->work_stream>>>(ctx->d_stub);
  }
  cudaError_t err = cudaGetLastError();
  if ( err != cudaSuccess ) {
    printf("CUDA Error: %s\n", cudaGetErrorString(err));
    return 0;
  }
  return 1;
}

#ifdef __cplusplus                                                             
}                                                                              
#endif                                                                         
