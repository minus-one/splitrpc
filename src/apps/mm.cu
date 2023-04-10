// Copyright (c) Adithya Kumar, The Pennsylvania State University. All rights reserved.
// Licensed under the MIT License.

// Assumes a square matrix of dim m_dim
// Can only do 16x16 or 32x32 matrices

#define MM_TB_SZ 32

#include "mm.cuh"
int M_DIM = 32;

__device__ __forceinline__ void simple_mm_internal(uint8_t __restrict *req, uint8_t __restrict *resp, int m_dim)
{
  int lane_x = threadIdx.x;
  int lane_y = threadIdx.y;
  
  float *A = (float*)req;
  float *B = (float*)(req + (m_dim * m_dim * sizeof(float)));
  float *C = (float*)resp;

  for(int x_dim = lane_x ; x_dim < m_dim ; x_dim += MM_TB_SZ) {
    for(int y_dim = lane_y ; y_dim < m_dim ; y_dim += MM_TB_SZ) {
      float val = 0;
      for(int k = 0 ; k < m_dim; k++) {
        val += A[y_dim * m_dim + k] * B[k * m_dim + x_dim]; 
      }
      C[y_dim * m_dim + x_dim] = val;
    }
  }
}

__global__ void
simple_mm(g_params stub, int m_dim)
{
  simple_mm_internal(stub.req, stub.resp, m_dim);
}

__global__ void
simple_mm_cg(volatile g_params *d_stub, int m_dim)
{
  simple_mm_internal(d_stub->req, d_stub->resp, m_dim);
}

__global__ void 
mm_pt(g_params *call_params, int m_dim, volatile uint32_t *door_bell)
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
    simple_mm_internal(call_params_shared.req, call_params_shared.resp, m_dim);

    __threadfence();
    __syncthreads();

    // Signal work to be complete
    if (tid == 0) {
      ACCESS_ONCE(*(door_bell)) = 2;
      __threadfence_system();
    }
  }
}

void kernel_entry(AppCtx *app_ctx)
{
  dim3 blockSize(MM_TB_SZ, MM_TB_SZ);
  dim3 gridSize(1, 1);
  simple_mm<<<gridSize, blockSize, 0, app_ctx->work_stream>>>(*app_ctx->h_stub, M_DIM);
}

void pt_entry(AppCtx *app_ctx)
{
  dim3 blockSize(MM_TB_SZ, MM_TB_SZ);
  dim3 gridSize(1, 1);
  mm_pt<<<gridSize, blockSize, 0, app_ctx->work_stream>>>(app_ctx->d_stub, M_DIM, app_ctx->d_door_bell);
}

void cdp_entry(AppCtx *app_ctx)
{

}

void cuda_graph_entry(AppCtx *app_ctx)
{
  if(!app_ctx->graphCreated) {
    printf("Constructing CUDA graph for mat-mul, m_dim: %d\n", M_DIM);
    checkCudaErrors(cudaStreamBeginCapture(app_ctx->work_stream, cudaStreamCaptureModeGlobal));
    dim3 blockSize(MM_TB_SZ, MM_TB_SZ);
    dim3 gridSize(1, 1);
    simple_mm_cg
      <<<gridSize, blockSize, 0, app_ctx->work_stream>>>(app_ctx->d_stub, M_DIM);
    checkCudaErrors(cudaStreamEndCapture(app_ctx->work_stream, &app_ctx->graph));
    checkCudaErrors(cudaGraphInstantiate(&app_ctx->instance, app_ctx->graph, NULL, NULL, 0));
    app_ctx->graphCreated = true;
  }
  // Ensure the stub info gets written so that device can pick it up
  _mm_mfence();
  checkCudaErrors(cudaGraphLaunch(app_ctx->instance, app_ctx->work_stream));
}
