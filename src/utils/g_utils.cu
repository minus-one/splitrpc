// Copyright (c) Adithya Kumar, The Pennsylvania State University. All rights reserved.
// Licensed under the MIT License.

#include "g_utils.cuh"
#include "debug_utils.h"
#include "config_utils.h"

// Waits on a door-bell for a value
__launch_bounds__(1) __global__ void work_notifier(volatile uint32_t *door_bell, uint32_t wait_value) 
{
  // Get thread ID.
  uint32_t wait_status;
  while (1) {
    wait_status = ACCESS_ONCE(*(door_bell));
    if(wait_status == wait_value) {
      break;
    }
  }
}

// Sets the door-bell with a value
__launch_bounds__(1) __global__ void completion_notifier(volatile uint32_t *door_bell, uint32_t signal_value)
{
  // Signal work to be complete
  ACCESS_ONCE(*(door_bell)) = signal_value;
  __threadfence_system();
}

// Sets a dummy value to an array (you can also use cudaMemset instead)
__launch_bounds__(1024) __global__ void SetDummyDataKernel(void *start_addr, int len, uint8_t dummy_value) {
  int lane_id = threadIdx.x;
  uint8_t* buf_start = (uint8_t*)start_addr;
  for(int elem_id = lane_id ; elem_id < len; elem_id += 1024) {
    buf_start[elem_id] = dummy_value;
  }
}

#ifdef __cplusplus                                                             
extern "C" {                                                                   
#endif                                                                         

void app_run_notifier(cudaStream_t& stream, volatile uint32_t *d_door_bell, int value)
{
  //work_notifier<<<1, 1, 0, stream>>>(d_door_bell, value);
  checkCudaErrors(cuStreamWaitValue32(stream, (CUdeviceptr)d_door_bell, value, 0));
}

//void app_complete_notifier(AppCtx *app_ctx, int i_idx, int signal_value)
void app_complete_notifier(cudaStream_t& stream, volatile uint32_t *d_door_bell, uint32_t value)
{
  //completion_notifier<<<1, 1, 0, stream>>>(d_door_bell, value);
  checkCudaErrors(cuStreamWriteValue32(stream, (CUdeviceptr)d_door_bell, value, 0));
}

void SetDummyData(void *start_addr, int len, uint8_t dummy_value) {
  printf("Setting dummy data for %p, len: %d, value: %d\n", start_addr, len, dummy_value);
  SetDummyDataKernel<<<1, 1024, 0, 0>>>(start_addr, len, dummy_value);
  checkCudaErrors(cudaStreamSynchronize(0));
}

// These are used to read arbitrary addresses from GPU memory and print their values
void g_floatDump(void *d_addr, size_t len, size_t trunc_len)
{
  float h_data[trunc_len] = {0.0f};
  len = std::min(trunc_len, len);
  printf("print_rr(int) d_addr: %p, len: %ld\n", d_addr, len);
  checkCudaErrors(cudaMemcpy((void*)h_data, (void*)d_addr, len * sizeof(float), cudaMemcpyDeviceToHost));
  for(int i = 0; i < len ; i++)
    printf("h_data[%d] = %f [%p]\n", i, h_data[i], (void*)((uint8_t*)d_addr + i*sizeof(float)));
}

void g_intDump(void *d_addr, size_t len, size_t trunc_len)
{
  uint8_t h_data[trunc_len] = {0U};
  len = std::min(size_t(trunc_len), len);
  printf("print_rr(int) d_addr: %p, len: %ld\n", d_addr, len);
  checkCudaErrors(cudaMemcpy((void*)h_data, (void*)d_addr, len * sizeof(uint8_t), cudaMemcpyDeviceToHost));
  for(int i = 0; i < len ; i++)
    printf("h_data[%d] = %d\n", i, h_data[i]);
}

void g_hexDump(void *d_addr, size_t len, size_t trunc_len)
{
  uint8_t h_data[trunc_len] = {0U};
  len = std::min(size_t(trunc_len), len);
  printf("print_rr(hex) d_addr: %p, len: %ld\n", d_addr, len);
  checkCudaErrors(cudaMemcpy((void*)h_data, (void*)d_addr, len, cudaMemcpyDeviceToHost));
  hexDump("HexDump", h_data, 64);
}

#ifdef __cplusplus                                                             
}                                                                              
#endif                                                                         
