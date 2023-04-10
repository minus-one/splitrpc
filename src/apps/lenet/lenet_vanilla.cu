// Copyright (c) Adithya Kumar, The Pennsylvania State University. All rights reserved.
// Licensed under the MIT License.

#include "lenet_vanilla.cuh"

//static inline int8_t get_work_launch_type() {
//  return readEnvInfo<int16_t>("P2P_RPC_WORK_LAUNCH_TYPE", 1);
//}

__device__ __forceinline__ unsigned long long __globaltimer()
{
  unsigned long long globaltimer;
  asm volatile("mov.u64 %0, %globaltimer;" : "=l"(globaltimer));
  return globaltimer;
}

__global__ void fuse_conv2d_kernel0( float* __restrict__ input0,  float* __restrict__ input1,  float* __restrict__ compute) {
   float compute_local[2];
  __shared__ float pad_temp_shared[180];
  __shared__ float input1_shared[500];
  for (int yy_c_init = 0; yy_c_init < 2; ++yy_c_init) {
    compute_local[yy_c_init] = 0.000000e+00f;
  }
  if ((((int)threadIdx.z) * 9) < ((180 - ((int)threadIdx.y)) - ((int)threadIdx.x))) {
    pad_temp_shared[(((((((((int)threadIdx.z) * 9) + ((int)threadIdx.y)) + ((int)threadIdx.x)) / 180) * 180) + ((((((int)threadIdx.z) * 3) + ((int)threadIdx.y)) + ((int)threadIdx.x)) % 6)) + ((((((((int)threadIdx.z) * 9) + ((int)threadIdx.y)) + ((int)threadIdx.x)) / 6) % 30) * 6))] = (((((1 <= (((((((int)threadIdx.z) * 9) + ((int)threadIdx.y)) + ((int)threadIdx.x)) / 6) % 30)) && ((((((((int)threadIdx.z) * 9) + ((int)threadIdx.y)) + ((int)threadIdx.x)) / 6) % 30) < 29)) && ((1 - ((((((int)threadIdx.z) * 3) + ((int)threadIdx.y)) + ((int)threadIdx.x)) % 6)) <= (((int)blockIdx.x) * 2))) && ((((int)blockIdx.x) * 2) < (29 - ((((((int)threadIdx.z) * 3) + ((int)threadIdx.y)) + ((int)threadIdx.x)) % 6)))) ? input0[(((((((int)blockIdx.x) * 2) + (((((((int)threadIdx.z) * 9) + ((int)threadIdx.y)) + ((int)threadIdx.x)) / 180) * 784)) + ((((((int)threadIdx.z) * 3) + ((int)threadIdx.y)) + ((int)threadIdx.x)) % 6)) + ((((((((int)threadIdx.z) * 9) + ((int)threadIdx.y)) + ((int)threadIdx.x)) / 6) % 30) * 28)) - 29)] : 0.000000e+00f);
  }
  if (((int)threadIdx.z) < (20 - (((((int)threadIdx.y) * 2) + ((int)threadIdx.x)) / 25))) {
    input1_shared[((((((int)threadIdx.z) * 25) + ((((((int)threadIdx.y) * 2) + ((int)threadIdx.x)) / 25) * 25)) + (((((int)threadIdx.y) * 2) + ((int)threadIdx.x)) % 5)) + (((((((int)threadIdx.y) * 2) + ((int)threadIdx.x)) / 5) % 5) * 5))] = input1[((((((int)threadIdx.z) * 25) + ((((((int)threadIdx.y) * 2) + ((int)threadIdx.x)) / 25) * 25)) + (((((int)threadIdx.y) * 2) + ((int)threadIdx.x)) % 5)) + (((((((int)threadIdx.y) * 2) + ((int)threadIdx.x)) / 5) % 5) * 5))];
  }
  __syncthreads();
  for (int ry_inner = 0; ry_inner < 5; ++ry_inner) {
    for (int rx_inner = 0; rx_inner < 5; ++rx_inner) {
      for (int yy_c = 0; yy_c < 2; ++yy_c) {
        compute_local[yy_c] = (compute_local[yy_c] + (pad_temp_shared[(((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + (ry_inner * 6)) + rx_inner) + (yy_c * 6))] * input1_shared[(((((int)threadIdx.z) * 25) + (ry_inner * 5)) + rx_inner)]));
      }
    }
  }
  for (int yy_inner_inner_inner = 0; yy_inner_inner_inner < 2; ++yy_inner_inner_inner) {
    compute[(((((((int)blockIdx.x) * 2) + (((int)threadIdx.z) * 676)) + (((int)threadIdx.y) * 52)) + ((int)threadIdx.x)) + (yy_inner_inner_inner * 26))] = compute_local[yy_inner_inner_inner];
  }
}

__global__ void fuse_tanh_kernel0( float* __restrict__ tensor,  float* __restrict__ input0) {
  if ((((int)blockIdx.x) * 512) < (13520 - ((int)threadIdx.x))) {
    tensor[((((int)blockIdx.x) * 512) + ((int)threadIdx.x))] = tanhf(input0[((((int)blockIdx.x) * 512) + ((int)threadIdx.x))]);
  }
}

__global__ void fuse_max_pool2d_kernel0( float* __restrict__ input0,  float* __restrict__ tensor) {
   float tensor_local[1];
  tensor_local[0] = -3.402823e+38f;
  for (int rv = 0; rv < 2; ++rv) {
    for (int rv1 = 0; rv1 < 2; ++rv1) {
      if ((((int)blockIdx.x) * 512) < (3380 - ((int)threadIdx.x))) {
        tensor_local[0] = max(tensor_local[0], input0[(((((((((((int)blockIdx.x) * 512) + ((int)threadIdx.x)) / 3380) * 13520) + ((((((int)blockIdx.x) * 5) + ((int)threadIdx.x)) % 13) * 2)) + (((((((int)blockIdx.x) * 512) + ((int)threadIdx.x)) / 13) % 13) * 52)) + (((((((int)blockIdx.x) * 512) + ((int)threadIdx.x)) / 169) % 20) * 676)) + (rv * 26)) + rv1)]);
      }
    }
  }
  if ((((int)blockIdx.x) * 512) < (3380 - ((int)threadIdx.x))) {
    tensor[(((((((((int)blockIdx.x) * 512) + ((int)threadIdx.x)) / 3380) * 3380) + (((((int)blockIdx.x) * 5) + ((int)threadIdx.x)) % 13)) + (((((((int)blockIdx.x) * 512) + ((int)threadIdx.x)) / 13) % 13) * 13)) + (((((((int)blockIdx.x) * 512) + ((int)threadIdx.x)) / 169) % 20) * 169))] = tensor_local[0];
  }
}


__global__ void fuse_conv2d_1_kernel0( float* __restrict__ input0,  float* __restrict__ input1,  float* __restrict__ compute) {
   float compute_local[1];
  __shared__ float pad_temp_shared[150];
  __shared__ float input1_shared[500];
  compute_local[0] = 0.000000e+00f;
  for (int rc_outer = 0; rc_outer < 10; ++rc_outer) {
    __syncthreads();
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner < 2; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) {
      if ((((int)threadIdx.z) * 3) < (30 - (((((int)threadIdx.y) * 2) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) / 5))) {
        pad_temp_shared[(((((((((int)threadIdx.z) * 3) + (((((int)threadIdx.y) * 2) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) / 5)) / 30) * 150) + ((((((int)threadIdx.z) * 3) + (((((int)threadIdx.y) * 2) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) / 5)) % 15) * 5)) + (((((int)threadIdx.y) * 2) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) % 5)) + (((((((int)threadIdx.z) * 3) + (((((int)threadIdx.y) * 2) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) / 5)) / 15) % 2) * 75))] = (((((1 <= (((((int)threadIdx.z) * 3) + (((((int)threadIdx.y) * 2) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) / 5)) % 15)) && ((((((int)threadIdx.z) * 3) + (((((int)threadIdx.y) * 2) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) / 5)) % 15) < 14)) && ((1 - (((((int)threadIdx.y) * 2) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) % 5)) <= ((int)blockIdx.x))) && (((int)blockIdx.x) < (14 - (((((int)threadIdx.y) * 2) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) % 5)))) ? input0[((((((((int)blockIdx.x) + (rc_outer * 338)) + ((((((int)threadIdx.z) * 3) + (((((int)threadIdx.y) * 2) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) / 5)) / 30) * 3380)) + ((((((int)threadIdx.z) * 3) + (((((int)threadIdx.y) * 2) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) / 5)) % 15) * 13)) + (((((int)threadIdx.y) * 2) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) % 5)) + (((((((int)threadIdx.z) * 3) + (((((int)threadIdx.y) * 2) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) / 5)) / 15) % 2) * 169)) - 14)] : 0.000000e+00f);
      }
    }
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1 = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1 < 5; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1) {
      if (((int)threadIdx.z) < (10 - (((int)threadIdx.y) / 10))) {
        if ((((int)blockIdx.z) * 10) < ((50 - ((int)threadIdx.z)) - (((int)threadIdx.y) / 10))) {
          input1_shared[(((((((int)threadIdx.z) * 50) + ((((int)threadIdx.y) / 10) * 50)) + ((((int)threadIdx.y) % 5) * 5)) + (((((int)threadIdx.y) / 5) % 2) * 25)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1)] = input1[(((((((((int)blockIdx.z) * 5000) + (rc_outer * 50)) + (((int)threadIdx.z) * 500)) + ((((int)threadIdx.y) / 10) * 500)) + ((((int)threadIdx.y) % 5) * 5)) + (((((int)threadIdx.y) / 5) % 2) * 25)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1)];
        }
      }
    }
    __syncthreads();
for (int rc_inner = 0; rc_inner < 2; ++rc_inner) {
      for (int ry_inner = 0; ry_inner < 5; ++ry_inner) {
        for (int rx_inner = 0; rx_inner < 5; ++rx_inner) {
          compute_local[0] = (compute_local[0] + (pad_temp_shared[((((((int)threadIdx.y) * 5) + (rc_inner * 75)) + (ry_inner * 5)) + rx_inner)] * input1_shared[((((((int)threadIdx.z) * 50) + (rc_inner * 25)) + (ry_inner * 5)) + rx_inner)]));
        }
      }
    }
  }
  compute[((((((int)blockIdx.z) * 1210) + ((int)blockIdx.x)) + (((int)threadIdx.z) * 121)) + (((int)threadIdx.y) * 11))] = compute_local[0];
}

__global__ void fuse_tanh_1_kernel0( float* __restrict__ tensor,  float* __restrict__ input0) {
  if ((((int)blockIdx.x) * 512) < (6050 - ((int)threadIdx.x))) {
    tensor[((((int)blockIdx.x) * 512) + ((int)threadIdx.x))] = tanhf(input0[((((int)blockIdx.x) * 512) + ((int)threadIdx.x))]);
  }
}

__global__ void fuse_max_pool2d_1_kernel0( float* __restrict__ input0,  float* __restrict__ tensor) {
   float tensor_local[1];
  tensor_local[0] = -3.402823e+38f;
  for (int rv = 0; rv < 2; ++rv) {
    for (int rv1 = 0; rv1 < 2; ++rv1) {
      if ((((int)blockIdx.x) * 512) < (1250 - ((int)threadIdx.x))) {
        tensor_local[0] = max(tensor_local[0], input0[(((((((((((int)blockIdx.x) * 512) + ((int)threadIdx.x)) / 1250) * 6050) + ((((((int)blockIdx.x) * 2) + ((int)threadIdx.x)) % 5) * 2)) + (((((((int)blockIdx.x) * 512) + ((int)threadIdx.x)) / 5) % 5) * 22)) + (((((((int)blockIdx.x) * 512) + ((int)threadIdx.x)) / 25) % 50) * 121)) + (rv * 11)) + rv1)]);
      }
    }
  }
  if ((((int)blockIdx.x) * 512) < (1250 - ((int)threadIdx.x))) {
    tensor[(((((((((int)blockIdx.x) * 512) + ((int)threadIdx.x)) / 1250) * 1250) + (((((int)blockIdx.x) * 2) + ((int)threadIdx.x)) % 5)) + (((((((int)blockIdx.x) * 512) + ((int)threadIdx.x)) / 5) % 5) * 5)) + (((((((int)blockIdx.x) * 512) + ((int)threadIdx.x)) / 25) % 50) * 25))] = tensor_local[0];
  }
}

__global__ void fuse_flatten_kernel0( float* __restrict__ tensor,  float* __restrict__ input0) {
  if ((((int)blockIdx.x) * 512) < (1250 - ((int)threadIdx.x))) {
    tensor[((((int)blockIdx.x) * 512) + ((int)threadIdx.x))] = input0[(((((((((int)blockIdx.x) * 512) + ((int)threadIdx.x)) / 1250) * 1250) + (((((((int)blockIdx.x) * 512) + ((int)threadIdx.x)) % 1250) / 25) * 25)) + ((((((((int)blockIdx.x) * 512) + ((int)threadIdx.x)) % 1250) / 5) % 5) * 5)) + ((((((int)blockIdx.x) * 512) + ((int)threadIdx.x)) % 1250) % 5))];
  }
}

__global__ void fuse_dense_kernel0( float* __restrict__ input0,  float* __restrict__ input1,  float* __restrict__ compute,  float* __restrict__ input2) {
   float compute_rf[1];
  __shared__ float red_buf0[64];
  __shared__ float compute1[1];
  compute_rf[0] = 0.000000e+00f;
  for (int k_outer = 0; k_outer < 20; ++k_outer) {
    if (((int)threadIdx.x) < (1250 - (k_outer * 64))) {
      compute_rf[0] = (compute_rf[0] + (input0[(((int)threadIdx.x) + (k_outer * 64))] * input1[(((((int)blockIdx.x) * 1250) + ((int)threadIdx.x)) + (k_outer * 64))]));
    }
  }
  ((volatile __shared__ float*)red_buf0)[((int)threadIdx.x)] = compute_rf[0];
  __syncthreads();
  if (((int)threadIdx.x) < 32) {
    ((volatile __shared__ float*)red_buf0)[((int)threadIdx.x)] = (((volatile __shared__ float*)red_buf0)[((int)threadIdx.x)] + ((volatile __shared__ float*)red_buf0)[(32 + ((int)threadIdx.x))]);
  }
  __syncthreads();
  if (((int)threadIdx.x) < 16) {
    ((volatile __shared__ float*)red_buf0)[((int)threadIdx.x)] = (((volatile __shared__ float*)red_buf0)[((int)threadIdx.x)] + ((volatile __shared__ float*)red_buf0)[(16 + ((int)threadIdx.x))]);
    ((volatile __shared__ float*)red_buf0)[((int)threadIdx.x)] = (((volatile __shared__ float*)red_buf0)[((int)threadIdx.x)] + ((volatile __shared__ float*)red_buf0)[(8 + ((int)threadIdx.x))]);
    ((volatile __shared__ float*)red_buf0)[((int)threadIdx.x)] = (((volatile __shared__ float*)red_buf0)[((int)threadIdx.x)] + ((volatile __shared__ float*)red_buf0)[(4 + ((int)threadIdx.x))]);
    ((volatile __shared__ float*)red_buf0)[((int)threadIdx.x)] = (((volatile __shared__ float*)red_buf0)[((int)threadIdx.x)] + ((volatile __shared__ float*)red_buf0)[(2 + ((int)threadIdx.x))]);
    ((volatile __shared__ float*)red_buf0)[((int)threadIdx.x)] = (((volatile __shared__ float*)red_buf0)[((int)threadIdx.x)] + ((volatile __shared__ float*)red_buf0)[(1 + ((int)threadIdx.x))]);
  }
  __syncthreads();
  if (((int)threadIdx.x) == 0) {
    compute1[0] = ((volatile __shared__ float*)red_buf0)[0];
  }
 if (((int)threadIdx.x) == 0) {
    compute[((int)blockIdx.x)] = (compute1[0] + input2[((int)blockIdx.x)]);
  }
}

__global__ void fuse_softmax_kernel0( float* __restrict__ tensor,  float* __restrict__ input0) {
  tensor[0] = -3.402823e+38f;
  for (int k1 = 0; k1 < 10; ++k1) {
    tensor[0] = max(tensor[0], input0[k1]);
  }
}

__global__ void fuse_softmax_kernel1( float* __restrict__ input0,  float* __restrict__ tensor,  float* __restrict__ tensor1) {
   float tensor_rf[1];
  __shared__ float red_buf0[64];
  tensor_rf[0] = 0.000000e+00f;
  if (((int)threadIdx.x) < 10) {
    tensor_rf[0] = (tensor_rf[0] + __expf((input0[((int)threadIdx.x)] - tensor[0])));
  }
  ((volatile __shared__ float*)red_buf0)[((int)threadIdx.x)] = tensor_rf[0];
  __syncthreads();
  if (((int)threadIdx.x) < 32) {
    ((volatile __shared__ float*)red_buf0)[((int)threadIdx.x)] = (((volatile __shared__ float*)red_buf0)[((int)threadIdx.x)] + ((volatile __shared__ float*)red_buf0)[(32 + ((int)threadIdx.x))]);
  }
  __syncthreads();
  if (((int)threadIdx.x) < 16) {
    ((volatile __shared__ float*)red_buf0)[((int)threadIdx.x)] = (((volatile __shared__ float*)red_buf0)[((int)threadIdx.x)] + ((volatile __shared__ float*)red_buf0)[(16 + ((int)threadIdx.x))]);
    ((volatile __shared__ float*)red_buf0)[((int)threadIdx.x)] = (((volatile __shared__ float*)red_buf0)[((int)threadIdx.x)] + ((volatile __shared__ float*)red_buf0)[(8 + ((int)threadIdx.x))]);
    ((volatile __shared__ float*)red_buf0)[((int)threadIdx.x)] = (((volatile __shared__ float*)red_buf0)[((int)threadIdx.x)] + ((volatile __shared__ float*)red_buf0)[(4 + ((int)threadIdx.x))]);
    ((volatile __shared__ float*)red_buf0)[((int)threadIdx.x)] = (((volatile __shared__ float*)red_buf0)[((int)threadIdx.x)] + ((volatile __shared__ float*)red_buf0)[(2 + ((int)threadIdx.x))]);
    ((volatile __shared__ float*)red_buf0)[((int)threadIdx.x)] = (((volatile __shared__ float*)red_buf0)[((int)threadIdx.x)] + ((volatile __shared__ float*)red_buf0)[(1 + ((int)threadIdx.x))]);
  }
  __syncthreads();
  if (((int)threadIdx.x) == 0) {
    tensor1[0] = ((volatile __shared__ float*)red_buf0)[0];
  }
}
__global__ void fuse_softmax_kernel2( float* __restrict__ tensor,  float* __restrict__ input0,  float* __restrict__ tensor1,  float* __restrict__ tensor2) {
  if (((int)threadIdx.x) < 10) {
    tensor[((int)threadIdx.x)] = (__expf((input0[((int)threadIdx.x)] - tensor1[0])) / tensor2[0]);
//	printf("out: %f\n", tensor[((int)threadIdx.x)]);
  }
}

__global__ void convf(volatile char *in_data, float *data, volatile float* oo) {
  int x = (volatile unsigned char)in_data[threadIdx.x] - 127;
  data[threadIdx.x] = __fdividef((float)x, 128);
	//if(threadIdx.x < 2) {
  //  // SOCKPERF HEADER MANAGEMENT
	//	*((volatile unsigned int *)oo + threadIdx.x) = *((volatile unsigned int *)in_data+threadIdx.x);
	//	if(threadIdx.x == 0)
	//			*(volatile int *)&oo[threadIdx.x] = 0x00000000;//*(int *)&oo[threadIdx.x] & 0xFFFFFF00;
////		if(threadIdx.x == 3)
////				*(volatile int *)&oo[threadIdx.x] = 0x00001400;//*(int *)&oo[threadIdx.x] & 0xFFFFFF00;
	//	
	//} else {

	//    int x;
  //  	float o;
	//	int index = 8 + threadIdx.x - 2;
	//    x = (volatile unsigned char)in_data[index] - 127;
  //  	data[threadIdx.x - 2] = __fdividef((float)x, 128);
	////    printf("%f\n",data[threadIdx.x - 4]);
	//}
}

//bool graphCreated = false;
//cudaGraph_t graph;
//cudaGraphExec_t instance;
//void lenet_graph_launch(float *data,
//    float *conv1_weight,
//    float *conv2_weight,
//    float *fc2_weight,
//    float *fc2_bias,
//    float *o0,
//    float *o1,
//    float *o2,
//    float *o3,
//    float *o4,
//    volatile g_params *call_params,
//    volatile uint32_t *door_bell,
//    cudaStream_t stream
//    ) {
//  if(!graphCreated) {
//    printf("Constructing CUDA graph\n");
//    checkCudaErrors(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
//
//    dim3 dimGrid_convf(1,1,1),  dimBlock_convf(784,1,1);
//    dim3 dimGrid_conv2d(13,1,1),  dimBlock_conv2d(2,13,20);
//    dim3 dimGrid_tanh(27,1,1),  dimBlock_tanh(512,1,1);
//    dim3 dimGrid_pool2d(7,1,1),  dimBlock_pool2d(512,1,1);
//    dim3 dimGrid_conv2d1(11,1,5),  dimBlock_conv2d1(1,11,10);
//    dim3 dimGrid_tanh1(12,1,1),  dimBlock_tanh1(512,1,1);
//    dim3 dimGrid_pool2d1(3,1,1),  dimBlock_pool2d1(512,1,1);
//    dim3 dimGrid_flatten(3,1,1),  dimBlock_flatten(512,1,1);
//    dim3 dimGrid_dense(10,1,1),  dimBlock_dense(64,1,1);
//    dim3 dimGrid_softmax0(1,1,1),  dimBlock_softmax0(1,1,1);
//    dim3 dimGrid_softmax1(1,1,1),  dimBlock_softmax1(64,1,1);
//    dim3 dimGrid_softmax2(1,1,1),  dimBlock_softmax2(64,1,1);
//
//    convf<<<dimGrid_convf,dimBlock_convf, 0, stream>>>((volatile char *)call_params->req, data, (volatile float*)call_params->req);
//    fuse_conv2d_kernel0<<<dimGrid_conv2d,dimBlock_conv2d, 0, stream>>>(data, conv1_weight, o1);
//    fuse_tanh_kernel0<<<dimGrid_tanh,dimBlock_tanh, 0, stream>>>(o2,o1);
//    fuse_max_pool2d_kernel0<<<dimGrid_pool2d,dimBlock_pool2d, 0, stream>>>(o2, o1);
//    fuse_conv2d_1_kernel0<<<dimGrid_conv2d1,dimBlock_conv2d1, 0, stream>>>(o1, conv2_weight, o2);
//    fuse_tanh_1_kernel0<<<dimGrid_tanh1,dimBlock_tanh1, 0, stream>>>(o1, o2);
//    fuse_max_pool2d_1_kernel0<<<dimGrid_pool2d1,dimBlock_pool2d1, 0, stream>>>(o1, o2);
//    fuse_flatten_kernel0<<<dimGrid_flatten,dimBlock_flatten, 0, stream>>>(o1, o2);
//    fuse_dense_kernel0<<<dimGrid_dense,dimBlock_dense, 0, stream>>>(o1, fc2_weight, o3, fc2_bias);
//    fuse_softmax_kernel0<<<dimGrid_softmax0,dimBlock_softmax0, 0, stream>>>(o4, o3);
//    fuse_softmax_kernel1<<<dimGrid_softmax1,dimBlock_softmax1, 0, stream>>>(o3, o4, o0);
//    fuse_softmax_kernel2<<<dimGrid_softmax2,dimBlock_softmax2, 0, stream>>>( ((float*)call_params->resp), o3, o4, o0);
//
//    checkCudaErrors(cudaStreamEndCapture(stream, &graph));
//    checkCudaErrors(cudaGraphInstantiate(&instance, graph, NULL, NULL, 0));
//    graphCreated=true;
//  }
//  checkCudaErrors(cudaGraphLaunch(instance, stream));
//  checkCudaErrors(cudaStreamSynchronize(stream));
//}

void lenet_kernel_launch(float *data,
    float *conv1_weight,
    float *conv2_weight,
    float *fc2_weight,
    float *fc2_bias,
    float *o0,
    float *o1,
    float *o2,
    float *o3,
    float *o4,
    volatile g_params *call_params,
    volatile uint32_t *door_bell,
    cudaStream_t stream
    ) {
  TRACE_PRINTF("LeNet req: %p, resp: %p\n", (void*)call_params->req, (void*)call_params->resp);
  dim3 dimGrid_convf(1,1,1),  dimBlock_convf(784,1,1);
  dim3 dimGrid_conv2d(13,1,1),  dimBlock_conv2d(2,13,20);
  dim3 dimGrid_tanh(27,1,1),  dimBlock_tanh(512,1,1);
  dim3 dimGrid_pool2d(7,1,1),  dimBlock_pool2d(512,1,1);
  dim3 dimGrid_conv2d1(11,1,5),  dimBlock_conv2d1(1,11,10);
  dim3 dimGrid_tanh1(12,1,1),  dimBlock_tanh1(512,1,1);
  dim3 dimGrid_pool2d1(3,1,1),  dimBlock_pool2d1(512,1,1);
  dim3 dimGrid_flatten(3,1,1),  dimBlock_flatten(512,1,1);
  dim3 dimGrid_dense(10,1,1),  dimBlock_dense(64,1,1);
  dim3 dimGrid_softmax0(1,1,1),  dimBlock_softmax0(1,1,1);
  dim3 dimGrid_softmax1(1,1,1),  dimBlock_softmax1(64,1,1);
  dim3 dimGrid_softmax2(1,1,1),  dimBlock_softmax2(64,1,1);

  //unsigned long long pkt_start = __globaltimer();
  convf<<<dimGrid_convf,dimBlock_convf, 0, stream>>>((volatile char *)call_params->req, data, (volatile float*)call_params->req);
  fuse_conv2d_kernel0<<<dimGrid_conv2d,dimBlock_conv2d, 0, stream>>>(data, conv1_weight, o1);
  fuse_tanh_kernel0<<<dimGrid_tanh,dimBlock_tanh, 0, stream>>>(o2,o1);
  fuse_max_pool2d_kernel0<<<dimGrid_pool2d,dimBlock_pool2d, 0, stream>>>(o2, o1);
  fuse_conv2d_1_kernel0<<<dimGrid_conv2d1,dimBlock_conv2d1, 0, stream>>>(o1, conv2_weight, o2);
  fuse_tanh_1_kernel0<<<dimGrid_tanh1,dimBlock_tanh1, 0, stream>>>(o1, o2);
  fuse_max_pool2d_1_kernel0<<<dimGrid_pool2d1,dimBlock_pool2d1, 0, stream>>>(o1, o2);
  fuse_flatten_kernel0<<<dimGrid_flatten,dimBlock_flatten, 0, stream>>>(o1, o2);
  fuse_dense_kernel0<<<dimGrid_dense,dimBlock_dense, 0, stream>>>(o1, fc2_weight, o3, fc2_bias);
  fuse_softmax_kernel0<<<dimGrid_softmax0,dimBlock_softmax0, 0, stream>>>(o4, o3);
  fuse_softmax_kernel1<<<dimGrid_softmax1,dimBlock_softmax1, 0, stream>>>(o3, o4, o0);
  fuse_softmax_kernel2<<<dimGrid_softmax2,dimBlock_softmax2, 0, stream>>>( ((float*)call_params->resp), o3, o4, o0);
  //printf("[globaltimer] Time in GPU %u ns\n", (__globaltimer() - pkt_start));
  //cudaStreamSynchronize(stream);
}

__global__ void lenet_dyn_kernel_launch(float *data,
    float *conv1_weight,
    float *conv2_weight,
    float *fc2_weight,
    float *fc2_bias,
    float *o0,
    float *o1,
    float *o2,
    float *o3,
    float *o4,
    char *inp_buf,
    char *out_buf
    ) {
  dim3 dimGrid_convf(1,1,1),  dimBlock_convf(784,1,1);
  dim3 dimGrid_conv2d(13,1,1),  dimBlock_conv2d(2,13,20);
  dim3 dimGrid_tanh(27,1,1),  dimBlock_tanh(512,1,1);
  dim3 dimGrid_pool2d(7,1,1),  dimBlock_pool2d(512,1,1);
  dim3 dimGrid_conv2d1(11,1,5),  dimBlock_conv2d1(1,11,10);
  dim3 dimGrid_tanh1(12,1,1),  dimBlock_tanh1(512,1,1);
  dim3 dimGrid_pool2d1(3,1,1),  dimBlock_pool2d1(512,1,1);
  dim3 dimGrid_flatten(3,1,1),  dimBlock_flatten(512,1,1);
  dim3 dimGrid_dense(10,1,1),  dimBlock_dense(64,1,1);
  dim3 dimGrid_softmax0(1,1,1),  dimBlock_softmax0(1,1,1);
  dim3 dimGrid_softmax1(1,1,1),  dimBlock_softmax1(64,1,1);
  dim3 dimGrid_softmax2(1,1,1),  dimBlock_softmax2(64,1,1);

  //unsigned long long pkt_start = __globaltimer();
  convf<<<dimGrid_convf,dimBlock_convf>>>((volatile char *)inp_buf, data, (volatile float*)inp_buf);
  cudaDeviceSynchronize();
  fuse_conv2d_kernel0<<<dimGrid_conv2d,dimBlock_conv2d>>>(data, conv1_weight, o1);
  cudaDeviceSynchronize();
  fuse_tanh_kernel0<<<dimGrid_tanh,dimBlock_tanh>>>(o2,o1);
  cudaDeviceSynchronize();
  fuse_max_pool2d_kernel0<<<dimGrid_pool2d,dimBlock_pool2d>>>(o2, o1);
  cudaDeviceSynchronize();
  fuse_conv2d_1_kernel0<<<dimGrid_conv2d1,dimBlock_conv2d1>>>(o1, conv2_weight, o2);
  cudaDeviceSynchronize();
  fuse_tanh_1_kernel0<<<dimGrid_tanh1,dimBlock_tanh1>>>(o1, o2);
  cudaDeviceSynchronize();
  fuse_max_pool2d_1_kernel0<<<dimGrid_pool2d1,dimBlock_pool2d1>>>(o1, o2);
  cudaDeviceSynchronize();
  fuse_flatten_kernel0<<<dimGrid_flatten,dimBlock_flatten>>>(o1, o2);
  cudaDeviceSynchronize();
  fuse_dense_kernel0<<<dimGrid_dense,dimBlock_dense>>>(o1, fc2_weight, o3, fc2_bias);
  cudaDeviceSynchronize();
  fuse_softmax_kernel0<<<dimGrid_softmax0,dimBlock_softmax0>>>(o4, o3);
  cudaDeviceSynchronize();
  fuse_softmax_kernel1<<<dimGrid_softmax1,dimBlock_softmax1>>>(o3, o4, o0);
  cudaDeviceSynchronize();
  fuse_softmax_kernel2<<<dimGrid_softmax2,dimBlock_softmax2>>>( ((float*)out_buf), o3, o4, o0);
  //fuse_softmax_kernel2<<<dimGrid_softmax2,dimBlock_softmax2>>>( ((float*)inp_buf), o3, o4, o0);
  //cudaDeviceSynchronize();
  //printf("Time in GPU %u ns\n", (__globaltimer() - pkt_start));
}

__launch_bounds__(1) __global__ void lenet(float *data,
        float *conv1_weight,
        float *conv2_weight,
        float *fc2_weight,
        float *fc2_bias,
        float *o0,
        float *o1,
        float *o2,
        float *o3,
        float *o4,
        volatile g_params *call_params,
        volatile uint32_t *door_bell
        ) {
  dim3 dimGrid_convf(1,1,1),  dimBlock_convf(784,1,1);
  dim3 dimGrid_conv2d(13,1,1),  dimBlock_conv2d(2,13,20);
  dim3 dimGrid_tanh(27,1,1),  dimBlock_tanh(512,1,1);
  dim3 dimGrid_pool2d(7,1,1),  dimBlock_pool2d(512,1,1);
  dim3 dimGrid_conv2d1(11,1,5),  dimBlock_conv2d1(1,11,10);
  dim3 dimGrid_tanh1(12,1,1),  dimBlock_tanh1(512,1,1);
  dim3 dimGrid_pool2d1(3,1,1),  dimBlock_pool2d1(512,1,1);
  dim3 dimGrid_flatten(3,1,1),  dimBlock_flatten(512,1,1);
  dim3 dimGrid_dense(10,1,1),  dimBlock_dense(64,1,1);
  dim3 dimGrid_softmax0(1,1,1),  dimBlock_softmax0(1,1,1);
  dim3 dimGrid_softmax1(1,1,1),  dimBlock_softmax1(64,1,1);
  dim3 dimGrid_softmax2(1,1,1),  dimBlock_softmax2(64,1,1);

  // Get thread ID.
  uint32_t wait_status;

  while(1) {
    while (1) {
      wait_status = ACCESS_ONCE(*(door_bell));
      if(wait_status == 1 || wait_status == 3) {
        break;
      }
    }

    if (wait_status != 1 && wait_status != 2)
      break;

    // Do Work
    //unsigned long long pkt_start = __globaltimer();
    convf<<<dimGrid_convf,dimBlock_convf>>>((volatile char *)call_params->req, data, (volatile float*)call_params->req);
    cudaDeviceSynchronize();
    fuse_conv2d_kernel0<<<dimGrid_conv2d,dimBlock_conv2d>>>(data, conv1_weight, o1);
    cudaDeviceSynchronize();
    fuse_tanh_kernel0<<<dimGrid_tanh,dimBlock_tanh>>>(o2,o1);
    cudaDeviceSynchronize();
    fuse_max_pool2d_kernel0<<<dimGrid_pool2d,dimBlock_pool2d>>>(o2, o1);
    cudaDeviceSynchronize();
    fuse_conv2d_1_kernel0<<<dimGrid_conv2d1,dimBlock_conv2d1>>>(o1, conv2_weight, o2);
    cudaDeviceSynchronize();
    fuse_tanh_1_kernel0<<<dimGrid_tanh1,dimBlock_tanh1>>>(o1, o2);
    cudaDeviceSynchronize();
    fuse_max_pool2d_1_kernel0<<<dimGrid_pool2d1,dimBlock_pool2d1>>>(o1, o2);
    cudaDeviceSynchronize();
    fuse_flatten_kernel0<<<dimGrid_flatten,dimBlock_flatten>>>(o1, o2);
    cudaDeviceSynchronize();
    fuse_dense_kernel0<<<dimGrid_dense,dimBlock_dense>>>(o1, fc2_weight, o3, fc2_bias);
    cudaDeviceSynchronize();
    fuse_softmax_kernel0<<<dimGrid_softmax0,dimBlock_softmax0>>>(o4, o3);
    cudaDeviceSynchronize();
    fuse_softmax_kernel1<<<dimGrid_softmax1,dimBlock_softmax1>>>(o3, o4, o0);
    cudaDeviceSynchronize();
    fuse_softmax_kernel2<<<dimGrid_softmax2,dimBlock_softmax2>>>( ((float*)call_params->resp), o3, o4, o0);
    cudaDeviceSynchronize();
    //printf("[globaltimer based] Time in GPU %u ns\n", (__globaltimer() - pkt_start));

    // Signal work to be complete
    ACCESS_ONCE(*(door_bell)) = 2;
    __threadfence_system();
  }
}

void load_from_file(std::string fname, float* buffer, unsigned int size, unsigned int offset) {
    int fd, ret;
    printf("Loading file: %s\n", fname.c_str());
    fd = open(fname.c_str(), O_RDONLY);
    float *h_buff = new float[size];
    ret = pread(fd, h_buff, size, offset);
    if (ret != size) {
        perror("read error");
        delete[] h_buff;
        exit(-1);
    }
    cudaMemcpy(buffer, h_buff, size, cudaMemcpyHostToDevice);
    delete[] h_buff;
    close(fd);
}

// Globals
float *o0, *o1, *o2, *o3, *o4, /**oo ,*/ *data, 
      *conv1_weight, *conv2_weight, 
      *fc2_weight, *fc2_bias;

void cuda_init()
{
  cudaMalloc(&o0, sizeof(float)* 1024);
  cudaMalloc(&o1, sizeof(float)* 13520);
  cudaMalloc(&o2, sizeof(float)* 13520);
  cudaMalloc(&o3, sizeof(float)* 10);
  cudaMalloc(&o4, sizeof(float)* 1024);
  cudaMalloc(&data, sizeof(float)* 28 * 28);
  cudaMalloc(&conv1_weight, sizeof(float)* 500);
  cudaMalloc(&conv2_weight, sizeof(float)* 25000);
  cudaMalloc(&fc2_weight, sizeof(float)* 12500);
  cudaMalloc(&fc2_bias, sizeof(float)* 10);

  std::string data_set_path = getDatasetBasePath() + std::string("data/lenet/");
  load_from_file(data_set_path + std::string("data2.dat"), conv1_weight,2000,0);
  load_from_file(data_set_path + std::string("data0.dat"), conv2_weight,100000,0);
  load_from_file(data_set_path + std::string("data1.dat"), fc2_weight,50000,0);
  load_from_file(data_set_path + std::string("data3.dat"), fc2_bias,40,0);
}

void kernel_entry(AppCtx *app_ctx) 
{
  lenet_kernel_launch(data, conv1_weight, conv2_weight, fc2_weight, fc2_bias, 
      o0, o1, o2, o3, o4, 
      app_ctx->h_stub, app_ctx->door_bell, app_ctx->work_stream); 
}

void cuda_graph_entry(AppCtx *app_ctx)
{
  if(!app_ctx->graphCreated) {
    printf("Constructing CUDA graph for LeNet\n");
    checkCudaErrors(cudaStreamBeginCapture(app_ctx->work_stream, cudaStreamCaptureModeGlobal));

    kernel_entry(app_ctx);
    checkCudaErrors(cudaStreamEndCapture(app_ctx->work_stream, &app_ctx->graph));
    checkCudaErrors(cudaGraphInstantiate(&app_ctx->instance, app_ctx->graph, NULL, NULL, 0));
    app_ctx->graphCreated = true;
  }
  checkCudaErrors(cudaGraphLaunch(app_ctx->instance, app_ctx->work_stream));

  //lenet_graph_launch(data, conv1_weight, conv2_weight, fc2_weight, fc2_bias, 
  //    o0, o1, o2, o3, o4, 
  //    app_ctx->h_stub, app_ctx->door_bell, app_ctx->work_stream); 
}

void cdp_entry(AppCtx *app_ctx)
{
  lenet_dyn_kernel_launch<<<1, 1, 0, app_ctx->work_stream>>>(data, conv1_weight, conv2_weight, fc2_weight, fc2_bias, 
      o0, o1, o2, o3, o4, 
      (char*)app_ctx->h_stub->req, (char*)app_ctx->h_stub->resp);
}

void pt_entry(AppCtx *app_ctx)
{
  lenet<<<1, 1, 0, app_ctx->work_stream>>>(data, conv1_weight, conv2_weight, fc2_weight, fc2_bias, 
      o0, o1, o2, o3, o4, 
      app_ctx->d_stub, app_ctx->d_door_bell);
}
