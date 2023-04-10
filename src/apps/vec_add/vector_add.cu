// Copyright (c) Adithya Kumar, The Pennsylvania State University. All rights reserved.
// Licensed under the MIT License.
#include <stdio.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/fcntl.h>

#include "vector_add.cuh"

// Globals
size_t VEC_ADD_LEN = 1024;
#define ADD_CONST 1
#define VECTOR_ADD_TB_SZ 1024

__device__ __forceinline__ unsigned long long __globaltimer()
{
  unsigned long long globaltimer;
  asm volatile("mov.u64 %0, %globaltimer;" : "=l"(globaltimer));
  return globaltimer;
}

__device__ __forceinline__ void 
vector_inc_internal(uint8_t __restrict *req, uint8_t __restrict *resp, size_t len)
{
  int lane_id = threadIdx.x;
  for (int idx = lane_id; idx < len; idx += VECTOR_ADD_TB_SZ)
    resp[idx] = req[idx] + ADD_CONST;
}

__launch_bounds__(VECTOR_ADD_TB_SZ) 
__global__ void 
vector_add(g_params call_params, size_t len)
{
  vector_inc_internal(call_params.req, call_params.resp, len);
}

__launch_bounds__(1) __global__ void work_notifier(volatile uint32_t *door_bell) 
{
  // Get thread ID.
  uint32_t wait_status;
  while (1) {
    wait_status = ACCESS_ONCE(*(door_bell));
    if(wait_status == 1 || wait_status == 3) {
      break;
    }
  }
}

__launch_bounds__(1) __global__ void completion_notifier(volatile uint32_t *door_bell)
{
  // Signal work to be complete
  ACCESS_ONCE(*(door_bell)) = 2;
  __threadfence_system();
}

__launch_bounds__(VECTOR_ADD_TB_SZ) 
__global__ void 
vector_add_cg(volatile g_params *d_stub, size_t len)
{
  vector_inc_internal(d_stub->req, d_stub->resp, len);
}


__global__ void 
vec_add_dyn_kernel_launch(volatile g_params *d_stub, size_t len) {
  dim3 blockSize(VECTOR_ADD_TB_SZ, 1, 1);
  dim3 gridSize(1, 1);

  // Do the work
  vector_add
    <<<gridSize, blockSize>>>(*d_stub, len);
  cudaDeviceSynchronize();
}

__global__ void 
vec_add_pt(g_params *call_params, size_t len, volatile uint32_t *door_bell)
{
  // Get thread ID.
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  uint32_t wait_status;
  __shared__ uint32_t wait_status_shared[1];
  __shared__ struct g_params call_params_shared;

  __syncthreads();

  while(1) {
    // Wait for work to be ready
    if (tid == 0) {
      while (1) {
        wait_status = ACCESS_ONCE(*(door_bell));
        if(wait_status == 1 || wait_status == 3) {
          wait_status_shared[0] = wait_status;
          call_params_shared.req = call_params->req;
          call_params_shared.resp = call_params->resp;
          __threadfence_block();
          break;
        }
      }
    } 
    __syncthreads();

    if (wait_status_shared[0] != 1 && wait_status_shared[0] != 2)
      break;

    //Do Work
    vector_inc_internal(call_params_shared.req, call_params_shared.resp, len);

    __threadfence();
    __syncthreads();

    // Signal work to be complete
    if (tid == 0) {
      ACCESS_ONCE(*(door_bell)) = 2;
      __threadfence_system();
    }
  }
}

dim3 blockSize(VECTOR_ADD_TB_SZ, 1, 1);
dim3 gridSize(1, 1);

void pt_entry(AppCtx *app_ctx)
{
  vec_add_pt<<<gridSize, blockSize, 0, app_ctx->work_stream>>>(app_ctx->d_stub, VEC_ADD_LEN, app_ctx->d_door_bell);
}

void kernel_entry(AppCtx *app_ctx)
{
  vector_add
    <<<gridSize, blockSize, 0, app_ctx->work_stream>>>(*app_ctx->h_stub, (VEC_ADD_LEN * app_ctx->curr_batch_size));
}

void cdp_entry(AppCtx *app_ctx)
{
  vec_add_dyn_kernel_launch<<<1, 1, 0, app_ctx->work_stream>>>(app_ctx->d_stub, VEC_ADD_LEN);
}

//bool graphCreated = false;
//cudaGraph_t graph;
//cudaGraphExec_t instance;
//void vec_add_graph_launch(g_params call_params, 
//    size_t len, cudaStream_t stream) {
//  if(!graphCreated) {
//    printf("Constructing CUDA graph\n");
//    checkCudaErrors(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
//
//    // Your kernels here
//    dim3 blockSize(VECTOR_ADD_TB_SZ, 1, 1);
//    dim3 gridSize(1, 1);
//    // Do the work
//    vector_add
//      <<<gridSize, blockSize, 0, stream>>>(call_params, len);
//
//    checkCudaErrors(cudaStreamEndCapture(stream, &graph));
//    
//    graphCreated=true;
//    
//    printf("Graph created\n");
//    checkCudaErrors(cudaGraphInstantiate(&instance, graph, NULL, NULL, 0));
//    printf("Instantiated the graph\n");
//  }     
//  checkCudaErrors(cudaGraphLaunch(instance, stream));
//  checkCudaErrors(cudaStreamSynchronize(stream));
//}

void cuda_graph_entry(AppCtx *app_ctx)
{
  if(!app_ctx->graphCreated) {
    printf("Constructing CUDA graph for vec-add, LEN: %ld, batch_size: %d\n", VEC_ADD_LEN, app_ctx->curr_batch_size);
    checkCudaErrors(cudaStreamBeginCapture(app_ctx->work_stream, cudaStreamCaptureModeGlobal));
    dim3 blockSize(VECTOR_ADD_TB_SZ, 1, 1);
    dim3 gridSize(1, 1);
    vector_add_cg
      <<<gridSize, blockSize, 0, app_ctx->work_stream>>>(app_ctx->d_stub, VEC_ADD_LEN * app_ctx->curr_batch_size);
    checkCudaErrors(cudaStreamEndCapture(app_ctx->work_stream, &app_ctx->graph));
    checkCudaErrors(cudaGraphInstantiate(&app_ctx->instance, app_ctx->graph, NULL, NULL, 0));
    app_ctx->graphCreated = true;
  }
  // Ensure the stub info gets written so that device can pick it up
  _mm_mfence();
  checkCudaErrors(cudaGraphLaunch(app_ctx->instance, app_ctx->work_stream));
  //vec_add_graph_launch(*(app_ctx->h_stub), VEC_ADD_LEN, app_ctx->work_stream); 
}
