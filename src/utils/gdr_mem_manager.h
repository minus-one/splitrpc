// Copyright (c) Adithya Kumar, The Pennsylvania State University. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "gdrapi.h"
#include <cuda_runtime.h>
#include <cuda.h>
#include <stdio.h>
#include "debug_utils.h"

static constexpr size_t NV_MIN_PIN_SIZE = 4;

#include <cuda.h>
#include <cuda_runtime.h>

#define CUDA_DRIVER_API
#include <helper_functions.h>
#include <helper_cuda.h>

// FIXME: Clean this and move to gdr_mem_manager
static inline uint8_t* alloc_gpu_mem(size_t mem_size, int device_id = 0)
{
  void *d_addr;
  checkCudaErrors(cudaSetDevice(device_id));
  if (cudaMalloc((void **)&d_addr, mem_size * sizeof(uint8_t)) != cudaSuccess) {
    printf("CudaMalloc failed, unable to create tmp_buffers\n");
    d_addr = NULL;
  }
  checkCudaErrors(cudaMemset(d_addr, 1, mem_size));
  return (uint8_t*)d_addr;
}

// FIXME: Complete this
static inline void free_gpu_mem(uint8_t* mem, int device_id = 0)
{
  checkCudaErrors(cudaSetDevice(device_id));
  cuMemFree((CUdeviceptr)mem);
}

// Container for collecting memory segment related info.
struct gdr_memseg_info {
  gdr_mh_t pgdr_handle;
  uintptr_t pdev_addr; // ptr to device-ptr
  uintptr_t phost_ptr; // ptr to host-ptr
  uintptr_t free_address; // ptr to device-ptr-free-addr (can be different because of offsets)
  size_t palloc_size; // size of mem_seg that is allocated 
  size_t input_size;  // size of mem_seg requested
};

class gdr_mem_manager {
  gdr_t gdr_descr;
  gdr_memseg_info g_m_flush;
  int device_id;

  public:
  gdr_mem_manager(int _device_id) {
    device_id = _device_id;
    TRACE_PRINTF("GDR_MM: Creating on device %d\n", device_id);
    cudaSetDevice(device_id);
    cudaFree(0);
    gdr_descr = gdr_open();
    if (gdr_descr == NULL) {
      fprintf(stderr, "gdr_open() on device: %d failed\n", device_id);
      exit(EXIT_FAILURE);
    }
    this->g_m_flush.input_size = sizeof(uint32_t);
    if (0 != alloc(&g_m_flush)) {
      fprintf(stderr, "Alloc of flush failed\n");
      exit(EXIT_FAILURE);
    }
    TRACE_PRINTF("GDR_MM: initialized, flush setup\n");
  }

  ~gdr_mem_manager() {
    cudaSetDevice(device_id);
    cudaFree(0);
    cleanup(&(g_m_flush));
  }

  gdr_t get_descr() {
    return gdr_descr;
  }

  int alloc(gdr_memseg_info * g_m);
  void cleanup(gdr_memseg_info * g_m); 
  int pin_and_map_memory(gdr_memseg_info *g_m, CUdeviceptr dev_addr, size_t len);
};

extern "C" gdr_mem_manager* get_gdr_mem_manager(int device_id);
