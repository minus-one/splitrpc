// Copyright (c) Adithya Kumar, The Pennsylvania State University. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>

#define CUDA_DRIVER_API
#include <helper_functions.h>
#include <helper_cuda.h>

#include <stdint.h>
#include <emmintrin.h> 

#ifdef __cplusplus                                                             
extern "C" {                                                                   
#endif                                                                         

  void app_run_notifier(cudaStream_t& stream, volatile uint32_t *d_door_bell, int value = 1);
  void app_complete_notifier(cudaStream_t& stream, volatile uint32_t *d_door_bell, uint32_t value = 2);
  void SetDummyData(void *start_addr, int len, uint8_t dummy_value);
  void g_floatDump(void *d_addr, size_t len, size_t trunc_len = 5);
  void g_intDump(void *d_addr, size_t len, size_t trunc_len = 5);
  void g_hexDump(void *d_addr, size_t len, size_t trunc_len = 64);

#ifdef __cplusplus                                                             
}                                                                              
#endif                                                                         
