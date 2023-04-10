// Copyright (c) Adithya Kumar, The Pennsylvania State University. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include <cstdint>

// This is used when you need there is a group of inputs/outputs
// for each request (bounded by MAX_NUM_TENSORS)
// Ensure each each input tensor is contiguous by itself,
// so that requests can be batched together
#define MAX_NUM_TENSORS 4
class g_params_v2 {
  public:
    uintptr_t inputs[MAX_NUM_TENSORS];
    uintptr_t outputs[MAX_NUM_TENSORS];
    size_t input_tensor_size[MAX_NUM_TENSORS];
    size_t output_tensor_size[MAX_NUM_TENSORS];
    //int input_device_id[MAX_NUM_TENSORS];
    //int output_device_id[MAX_NUM_TENSORS];
}__attribute__((packed));

// Classical way to run apps. Every app has one request
// and gives one response. Inputs and Outputs by themselves
// maybe contiguous.
class g_params {
  public:
  uint8_t *req;   // Ptr to the req payload
  uint8_t *resp;  // Ptr to the resp payload 
}__attribute__((packed));

