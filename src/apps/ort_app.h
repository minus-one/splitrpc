// Copyright (c) Adithya Kumar, The Pennsylvania State University. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/session/onnxruntime_c_api.h"
#include <assert.h>
#include <stdlib.h>
#include <vector>
#include <iostream>

#ifdef USE_TRT
#include "core/providers/tensorrt/tensorrt_provider_factory.h"
#endif

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#ifdef USE_DNNL
#include "core/providers/dnnl/dnnl_provider_factory.h"
#endif

#include "debug_utils.h"
#define CUDA_DRIVER_API
#include <helper_functions.h>
#include <helper_cuda.h>

#include "ort_defs.h"

const OrtApi* g_ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);

//*****************************************************************************
// helper function to check for status
void CheckStatus(OrtStatus* status)
{
  if (status != NULL) {
    const char* msg = g_ort->GetErrorMessage(status);
    fprintf(stderr, "%s\n", msg);
    g_ort->ReleaseStatus(status);
    exit(1);
  }
}

// Overloaded
uint64_t getTensorSize(std::vector<int64_t>& node_dims) {
  uint64_t len = 1;
  //std::cout<<"Shape--> ";
  for(int64_t& dim : node_dims) {
    int64_t t_dim = (dim > 0 ? dim: 1);
    dim = t_dim;
    len *= t_dim; 
    //std::cout<<t_dim<<" ";
  }
  //std::cout<<"Total Size: "<<len<<std::endl;
  return len;
}

// Overloaded
uint64_t getTensorSize(OrtValue* tensor) {
  OrtTensorTypeAndShapeInfo* tensor_info;
  CheckStatus(g_ort->CreateTensorTypeAndShapeInfo(&tensor_info));
  CheckStatus(g_ort->GetTensorTypeAndShape(tensor, &tensor_info));

  size_t num_dims;
  CheckStatus(g_ort->GetDimensionsCount(tensor_info, &num_dims));

  std::vector<int64_t> node_dims;
  node_dims.resize(num_dims);
  CheckStatus(g_ort->GetDimensions(tensor_info, (int64_t*)node_dims.data(), num_dims));
  g_ort->ReleaseTensorTypeAndShapeInfo(tensor_info);

  return getTensorSize(node_dims);
}

class OrtFacade
{
  private:
    // ORT specific
    OrtEnv* env_;
    OrtSessionOptions* session_options_;
    OrtMemoryInfo* cpu_memory_info_;
    OrtMemoryInfo *gpu_memory_info_;

    // Session specific info
    OrtSession* session_;
    std::array<ORT_IO_INFO, MAX_BS + 1> input_io_info_map;
    std::array<ORT_IO_INFO, MAX_BS + 1> output_io_info_map;

    std::vector<OrtValue*> input_tensor_values, output_tensor_values;
    
    // IO Binding related
    OrtIoBinding *io_bind;
    std::vector<void*> input_tensor_ptrs, output_tensor_ptrs;

    cudaStream_t work_stream;

  public:
    OrtFacade() {
      //CheckStatus(g_ort->CreateEnv(ORT_LOGGING_LEVEL_VERBOSE, "verbose", &env_));
      CheckStatus(g_ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "warn", &env_));
      CheckStatus(g_ort->CreateSessionOptions(&session_options_));
      CheckStatus(g_ort->SetSessionGraphOptimizationLevel(session_options_, ORT_ENABLE_ALL));
      //CheckStatus(g_ort->SetSessionLogVerbosityLevel(session_options_, 0));
      CheckStatus(g_ort->SetSessionExecutionMode(session_options_, ORT_SEQUENTIAL));
      //CheckStatus(g_ort->EnableProfiling(session_options_, "ort"));

      CheckStatus(g_ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &cpu_memory_info_));
      CheckStatus(g_ort->CreateMemoryInfo("Cuda", OrtArenaAllocator, 0, OrtMemTypeDefault, &gpu_memory_info_));

      cudaStreamCreateWithFlags(&work_stream, cudaStreamNonBlocking);
#ifndef USE_TRT
      OrtCUDAProviderOptions cuda_options;
      cuda_options.has_user_compute_stream = 1;
      cuda_options.user_compute_stream = (void*)work_stream; 
      CheckStatus(g_ort->SessionOptionsAppendExecutionProvider_CUDA(session_options_, &cuda_options));
#else
      OrtTensorRTProviderOptions trt_options;
      trt_options.has_user_compute_stream = 1;
      trt_options.user_compute_stream = (void*)work_stream;
      trt_options.trt_engine_decryption_enable = 0;
      CheckStatus(g_ort->SessionOptionsAppendExecutionProvider_TensorRT(session_options_, &trt_options));
#endif

#ifdef USE_DNNL
      // params: session_options, mem_arena_enable?
      CheckStatus(OrtSessionOptionsAppendExecutionProvider_Dnnl(session_options_, 1));
#endif
      //input_tensor = NULL;
      //output_tensor = NULL;
    }

    OrtFacade(int _device_id, cudaStream_t _work_stream) {
      TRACE_PRINTF("OrtFacade: device_id: %d, stream: %p\n", 
          _device_id, (void*)_work_stream);
      CheckStatus(g_ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "warn", &env_));
      CheckStatus(g_ort->CreateSessionOptions(&session_options_));
      CheckStatus(g_ort->SetSessionGraphOptimizationLevel(session_options_, ORT_ENABLE_ALL));
      CheckStatus(g_ort->SetSessionExecutionMode(session_options_, ORT_SEQUENTIAL));
      work_stream = _work_stream;

#ifdef USE_TRT
      OrtTensorRTProviderOptions trt_options;
      trt_options.device_id = _device_id;
      trt_options.has_user_compute_stream = 1;
      trt_options.user_compute_stream = (void*)work_stream;
      CheckStatus(g_ort->SessionOptionsAppendExecutionProvider_TensorRT(session_options_, &trt_options));
      //OrtSessionOptionsAppendExecutionProvider_Tensorrt(session_options_, _device_id);
#else
      OrtCUDAProviderOptions cuda_options;
      cuda_options.device_id = _device_id;
      cuda_options.has_user_compute_stream = 1;
      cuda_options.user_compute_stream = (void*)work_stream; 
      CheckStatus(g_ort->SessionOptionsAppendExecutionProvider_CUDA(session_options_, &cuda_options));
#endif

      CheckStatus(g_ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &cpu_memory_info_));
      CheckStatus(g_ort->CreateMemoryInfo("Cuda", OrtArenaAllocator, 0, OrtMemTypeDefault, &gpu_memory_info_));
    }

    ~OrtFacade() {
      TRACE_PRINTF("Destructing facade\n");
      g_ort->ReleaseMemoryInfo(cpu_memory_info_);
      g_ort->ReleaseMemoryInfo(gpu_memory_info_);
      //g_ort->ReleaseSession(session_);
      //g_ort->ReleaseSessionOptions(session_options_);
      //g_ort->ReleaseEnv(env_);
    }

    void loadModel(std::string model_filename)
    {
      printf("Loading ORT MODEL: %s, MAX_BS: %d\n", model_filename.c_str(), MAX_BS);
      CheckStatus(g_ort->CreateSession(env_, model_filename.c_str(), session_options_, &session_));

      // Gather model info
      OrtAllocator* allocator;
      CheckStatus(g_ort->GetAllocatorWithDefaultOptions(&allocator));

      size_t num_input_nodes, num_output_nodes;
      CheckStatus(g_ort->SessionGetInputCount(session_, &num_input_nodes));
      CheckStatus(g_ort->SessionGetOutputCount(session_, &num_output_nodes));
      ORT_IO_INFO &input_io_info = input_io_info_map[0];
      ORT_IO_INFO &output_io_info = output_io_info_map[0];

      input_io_info.set_num_nodes_and_resize_info(num_input_nodes);
      output_io_info.set_num_nodes_and_resize_info(num_output_nodes);

      char *symbolic_shape_names[10];

      for (size_t i = 0; i < num_input_nodes; i++) {
        char* input_name;
        CheckStatus(g_ort->SessionGetInputName(session_, i, allocator, &input_name));
        input_io_info.io_names[i] = input_name;

        // save input node types
        OrtTypeInfo* typeinfo;
        CheckStatus(g_ort->SessionGetInputTypeInfo(session_, i, &typeinfo));
        const OrtTensorTypeAndShapeInfo* tensor_info;
        CheckStatus(g_ort->CastTypeInfoToTensorInfo(typeinfo, &tensor_info));
        size_t num_dims;
        CheckStatus(g_ort->GetDimensionsCount(tensor_info, &num_dims));
        input_io_info.io_dims[i].resize(num_dims);

        CheckStatus(g_ort->GetDimensions(tensor_info, (int64_t*)input_io_info.get_node_shape(i), num_dims));

        enum ONNXTensorElementDataType type_of_tensor;
        CheckStatus(g_ort->GetTensorElementType(tensor_info, &type_of_tensor));
        input_io_info.io_types[i] = type_of_tensor;

        printf("Input[%ld]: %s, num_dims: %ld, Type: %d\n", i, input_name, num_dims, type_of_tensor);

        // Get the dimensions which are symbolic
        CheckStatus(g_ort->GetSymbolicDimensions(tensor_info, (const char**)symbolic_shape_names, num_dims));
        for(size_t j = 0 ; j < num_dims; j++) {
          printf("DIM[%ld]: value: %ld, Name: %s\n", j, input_io_info.get_node_shape(i)[j], symbolic_shape_names[j]);
        }

        if(type_of_tensor == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
          input_io_info.io_sizes[i] = getTensorSize(input_io_info.io_dims[i]) * sizeof(float);
        else if(type_of_tensor == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32)
          input_io_info.io_sizes[i] = getTensorSize(input_io_info.io_dims[i]) * sizeof(uint32_t);
        else if(type_of_tensor == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64)
          input_io_info.io_sizes[i] = getTensorSize(input_io_info.io_dims[i]) * sizeof(uint64_t);
        else
          input_io_info.io_sizes[i] = getTensorSize(input_io_info.io_dims[i]) * sizeof(float);

        input_io_info.calc_tot_io_size();

        g_ort->ReleaseTypeInfo(typeinfo);
        OrtValue* i_t = {nullptr};
        input_tensor_values.push_back(i_t);
      }

      
      for (size_t i = 0; i < num_output_nodes; i++) {
        char* output_name;
        CheckStatus(g_ort->SessionGetOutputName(session_, i, allocator, &output_name));
        output_io_info.io_names[i] = output_name;

        // save output node types
        OrtTypeInfo* typeinfo;
        CheckStatus(g_ort->SessionGetOutputTypeInfo(session_, i, &typeinfo));
        const OrtTensorTypeAndShapeInfo* tensor_info;
        CheckStatus(g_ort->CastTypeInfoToTensorInfo(typeinfo, &tensor_info));
        size_t num_dims;
        CheckStatus(g_ort->GetDimensionsCount(tensor_info, &num_dims));
        output_io_info.io_dims[i].resize(num_dims);

        CheckStatus(g_ort->GetDimensions(tensor_info, (int64_t*)output_io_info.get_node_shape(i), num_dims));

        enum ONNXTensorElementDataType type_of_tensor;
        CheckStatus(g_ort->GetTensorElementType(tensor_info, &type_of_tensor));
        output_io_info.io_types[i] = type_of_tensor;

        printf("Output[%ld]: %s, num_dims: %ld, Type: %d\n", i, output_name, num_dims, type_of_tensor);

        // Get the dimensions which are symbolic
        CheckStatus(g_ort->GetSymbolicDimensions(tensor_info, (const char**)symbolic_shape_names, num_dims));
        for(size_t j = 0 ; j < num_dims; j++) {
          printf("DIM[%ld]: value: %ld, Name: %s\n", j, output_io_info.get_node_shape(i)[j], symbolic_shape_names[j]);
        }

        if(type_of_tensor == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
          output_io_info.io_sizes[i] = getTensorSize(output_io_info.io_dims[i]) * sizeof(float);
        else if(type_of_tensor == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32)
          output_io_info.io_sizes[i] = getTensorSize(output_io_info.io_dims[i]) * sizeof(uint32_t);
        else if(type_of_tensor == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64)
          output_io_info.io_sizes[i] = getTensorSize(output_io_info.io_dims[i]) * sizeof(uint64_t);
        else
          output_io_info.io_sizes[i] = getTensorSize(output_io_info.io_dims[i]) * sizeof(float);

        output_io_info.calc_tot_io_size();

        g_ort->ReleaseTypeInfo(typeinfo);
        OrtValue* o_t = {nullptr};
        output_tensor_values.push_back(o_t); 
      }

      if(setup_io_dim_map(model_filename, input_io_info_map, output_io_info_map)) {
        printf("Setting up dim map for different batch-sizes successful\n");
      } else {
        printf("DIM map for individual batch-sizes failed. Using default for all batch-sizes\n");
      }
    }

    void do_device_mem_allocations(int bs = 1) 
    {
      ORT_IO_INFO &input_io_info = input_io_info_map[bs];
      ORT_IO_INFO &output_io_info = output_io_info_map[bs];

      printf("Setting up input buffers, batch_size: %d\n", bs);
      for(size_t i = 0 ; i < input_io_info.num_nodes; i++) {
        void *d_data;
        printf("Allocating memory using cuda\n");
        checkCudaErrors(cudaMalloc(&d_data, input_io_info.get_node_size(i))); 
        std::cout << "Tensor ptr: " << d_data << " Size: " << input_io_info.get_node_size(i) << "\n";
        input_tensor_ptrs.push_back(d_data);
      }

      printf("Setting up output buffers\n");
      for(size_t i = 0 ; i < output_io_info.num_nodes; i++) {
        void *d_data;
        printf("Allocating memory using cuda\n");
        checkCudaErrors(cudaMalloc(&d_data, output_io_info.get_node_size(i)));
        std::cout << "Tensor ptr: " << d_data << " Size: " << output_io_info.get_node_size(i) << "\n";
        output_tensor_ptrs.push_back(d_data);
      }
    }

    // Memory if passed is assumed to be of correct size
    void setup_io_binding(std::vector<void*> d_inp_data, std::vector<void*> d_out_data, int bs = 1) 
    {
      ORT_IO_INFO &input_io_info = input_io_info_map[bs];
      ORT_IO_INFO &output_io_info = output_io_info_map[bs];

      CheckStatus(g_ort->CreateIoBinding(session_, &io_bind));
      printf("Setting up and binding input buffers, batch_size: %d\n", bs);
      for(size_t i = 0 ; i < input_io_info.num_nodes; i++) {
        OrtValue* i_tensor = {nullptr};
        void *d_data;
        if(i < d_inp_data.size()) {
          d_data = d_inp_data[i];
        } else {
          printf("Allocating memory using cuda\n");
          checkCudaErrors(cudaMalloc(&d_data, input_io_info.get_node_size(i))); 
        }
        std::cout << "Tensor ptr: " << d_data << " Size: " << input_io_info.get_node_size(i) << "\n";
        input_tensor_ptrs.push_back(d_data);
        CheckStatus(
            g_ort->CreateTensorWithDataAsOrtValue(gpu_memory_info_, 
              d_data, input_io_info.get_node_size(i), 
              input_io_info.get_node_shape(i), input_io_info.get_node_shape_len(i), 
              input_io_info.get_node_type(i), &i_tensor));
        CheckStatus(g_ort->BindInput(io_bind, input_io_info.io_names[i], i_tensor));
        //input_tensor_values.push_back(i_tensor);
      }

      printf("Setting up and binding output buffers\n");
      for(size_t i = 0 ; i < output_io_info.num_nodes; i++) {
        OrtValue* o_tensor = {nullptr};
        void *d_data;
        if(i < d_out_data.size()) {
          d_data = d_out_data[i];
        } else {
          printf("Allocating memory using cuda\n");
          checkCudaErrors(cudaMalloc(&d_data, output_io_info.get_node_size(i)));
        }
        std::cout << "Tensor ptr: " << d_data << " Size: " << output_io_info.get_node_size(i) << "\n";
        output_tensor_ptrs.push_back(d_data);

        CheckStatus(
            g_ort->CreateTensorWithDataAsOrtValue(gpu_memory_info_, 
              d_data, output_io_info.get_node_size(i), 
              output_io_info.get_node_shape(i), output_io_info.get_node_shape_len(i),
              output_io_info.get_node_type(i), &o_tensor));
        output_tensor_values[i] = o_tensor;

        CheckStatus(g_ort->BindOutput(io_bind, output_io_info.io_names[i], o_tensor));
        //output_tensor_values.push_back(o_tensor);
      }
    }

    void** getInputPtrs()
    {
      return input_tensor_ptrs.data();
    }

    void** getOutputPtrs()
    {
      return output_tensor_ptrs.data();
    }

    void copyInputs(void *input_data, int bs_idx=0)
    {
      void** i_ptrs = getInputPtrs();
      // Always look up from bs=0, so that we get per-request sizes and not for entire batch
      ORT_IO_INFO &input_io_info = input_io_info_map[0];
      size_t offset = 0;
      for(size_t i = 0 ; i < input_io_info.num_nodes; i++) {
        void *i_ptr = (void*)((uint8_t*)i_ptrs[i] + (bs_idx * input_io_info.get_node_size(i)));
        void *h_data = (void*)((uint8_t*)input_data + offset);
        TRACE_PRINTF("Copy In: h: %p, d: %p, size: %ld\n", h_data, i_ptr, input_io_info.get_node_size(i)); 
        checkCudaErrors(cudaMemcpyAsync(i_ptr, h_data, input_io_info.get_node_size(i), cudaMemcpyHostToDevice, work_stream));
        offset += input_io_info.get_node_size(i);
      }
      //checkCudaErrors(cudaStreamSynchronize(work_stream));
    }

    void copyOutputs(void *output_data, int bs_idx=0)
    {
      void** o_ptrs = getOutputPtrs();
      ORT_IO_INFO &output_io_info = output_io_info_map[0];
      size_t offset = 0;
      for(size_t i = 0 ; i < output_io_info.num_nodes; i++) {
        void *o_ptr = (void*)((uint8_t*)o_ptrs[i] + (bs_idx * output_io_info.get_node_size(i)));
        void *h_data = (void*)((uint8_t*)output_data + offset);
        TRACE_PRINTF("Copy Out: h: %p, d: %p, size: %ld\n", h_data, o_ptr, output_io_info.get_node_size(i)); 
        checkCudaErrors(cudaMemcpyAsync(h_data, o_ptr, output_io_info.get_node_size(i), cudaMemcpyDeviceToHost, work_stream));
        offset += output_io_info.get_node_size(i);
      }
      checkCudaErrors(cudaStreamSynchronize(work_stream));
    }

    void run_on_loaded_data(int bs = 1) 
    {
      TRACE_PRINTF("OrtApp bs: %d\n", bs);

      ORT_IO_INFO &input_io_info = input_io_info_map[bs];
      ORT_IO_INFO &output_io_info = output_io_info_map[bs];
      void** i_ptrs = getInputPtrs();
      void** o_ptrs = getOutputPtrs();

      for(size_t i = 0 ; i < input_io_info.num_nodes; i++) {
        OrtValue* i_tensor = {nullptr};
        CheckStatus(
            g_ort->CreateTensorWithDataAsOrtValue(gpu_memory_info_, 
              i_ptrs[i], input_io_info.get_node_size(i), 
              input_io_info.get_node_shape(i), input_io_info.get_node_shape_len(i), 
              input_io_info.get_node_type(i), &i_tensor));
        input_tensor_values[i] = i_tensor;
      }

      for(size_t i = 0 ; i < output_io_info.num_nodes; i++) {
        OrtValue* o_tensor = {nullptr};
        CheckStatus(
            g_ort->CreateTensorWithDataAsOrtValue(gpu_memory_info_, 
              o_ptrs[i], output_io_info.get_node_size(i), 
              output_io_info.get_node_shape(i), output_io_info.get_node_shape_len(i),
              output_io_info.get_node_type(i), &o_tensor));
        output_tensor_values[i] = o_tensor;
      }
      CheckStatus(
          g_ort->Run(session_, NULL, 
            (const char *const *)input_io_info.get_node_names(), 
            (const OrtValue *const *)input_tensor_values.data(), input_io_info.num_nodes, 
            (const char *const *)output_io_info.get_node_names(), 
            output_io_info.num_nodes, (OrtValue**)output_tensor_values.data()));
      sync();
      for(size_t i = 0 ; i < input_io_info_map[0].num_nodes; i++) {
        g_ort->ReleaseValue(input_tensor_values[i]);
      }
      for(size_t i = 0 ; i < output_io_info_map[0].num_nodes; i++) {
        g_ort->ReleaseValue(output_tensor_values[i]);
      }
    }

    void sync()
    {
      cudaStreamSynchronize(work_stream);
    }

    void predict_with_io_binding()
    {
      CheckStatus(g_ort->RunWithBinding(session_, NULL, io_bind));
      //CheckStatus(g_ort->SynchronizeBoundOutputs(io_bind));
    }

    void printModelInfo()
    {
      printf("============== DEFAULTS =============\n");
      printf("Input IO_INFO:\n");
      input_io_info_map[0].print_io_info();

      printf("Output IO_INFO:\n");
      output_io_info_map[0].print_io_info();
      printf("=======================================\n");
      for(int i = 1 ; i < MAX_BS + 1; i++) {
        printf("============== BS = %d=============\n", i);
        printf("Input IO_INFO:\n");
        input_io_info_map[i].print_io_info();
        printf("Output IO_INFO:\n");
        output_io_info_map[i].print_io_info();
        printf("=======================================\n");
      }
    }

    inline void setup_run_with_data(uint8_t* d_inp_data, uint8_t* d_out_data, int bs = 1) 
    {
      TRACE_PRINTF("OrtApp input: %p, output: %p, bs: %d\n", 
          (void*)d_inp_data, (void*)d_out_data, bs);

      size_t offset = 0;
      ORT_IO_INFO &input_io_info = input_io_info_map[bs];
      ORT_IO_INFO &output_io_info = output_io_info_map[bs];

      for(size_t i = 0 ; i < input_io_info.num_nodes; i++) {
        OrtValue* i_tensor = {nullptr};
        void *d_data = (void*)(d_inp_data + offset);
        CheckStatus(
            g_ort->CreateTensorWithDataAsOrtValue(gpu_memory_info_, 
              d_data, input_io_info.get_node_size(i), 
              input_io_info.get_node_shape(i), input_io_info.get_node_shape_len(i), 
              input_io_info.get_node_type(i), &i_tensor));
        input_tensor_values[i] = i_tensor;
        offset += input_io_info.get_node_size(i);
      }

      offset = 0;
      for(size_t i = 0 ; i < output_io_info.num_nodes; i++) {
        OrtValue* o_tensor = {nullptr};
        void *d_data = (void*)(d_out_data + offset);
        CheckStatus(
            g_ort->CreateTensorWithDataAsOrtValue(gpu_memory_info_, 
              d_data, output_io_info.get_node_size(i), 
              output_io_info.get_node_shape(i), output_io_info.get_node_shape_len(i),
              output_io_info.get_node_type(i), &o_tensor));
        output_tensor_values[i] = o_tensor;
        offset += output_io_info.get_node_size(i);
      }

      CheckStatus(
          g_ort->Run(session_, NULL, 
            (const char *const *)input_io_info.get_node_names(), 
            (const OrtValue *const *)input_tensor_values.data(), input_io_info.num_nodes, 
            (const char *const *)output_io_info.get_node_names(), 
            output_io_info.num_nodes, (OrtValue**)output_tensor_values.data()));
    }

    inline void run_complete() 
    {
      for(size_t i = 0 ; i < input_io_info_map[0].num_nodes; i++) {
        g_ort->ReleaseValue(input_tensor_values[i]);
      }
      for(size_t i = 0 ; i < output_io_info_map[0].num_nodes; i++) {
        g_ort->ReleaseValue(output_tensor_values[i]);
      }
    }

    //void predict_with_loaded_data()
    //{
    //  CheckStatus(
    //      g_ort->Run(session_, NULL, 
    //      (const char *const *)input_node_names.data(), 
    //      (const OrtValue *const *)input_tensor_values.data(), num_input_nodes, 
    //      (const char *const *)output_node_names.data(), num_output_nodes, (OrtValue**)output_tensor_values.data()));
    //}

    // Single-prediction
    //size_t predict(void* data, size_t inputLen, void* outputData, uint64_t batchSize=1)
    //{
    //  //uint64_t estimatedBatchSize = 1;
    //  //if(batchSize == 0) {
    //  //  estimatedBatchSize = inputLen / (input_tensor_sizes[0]);
    //  //} else {
    //  //  estimatedBatchSize = batchSize;
    //  //}
    //  //assert(estimatedBatchSize >= 1);

    //  //input_node_dims[0][0]   = estimatedBatchSize;
    //  //output_node_dims[0][0]  = estimatedBatchSize;
    //  input_node_dims[0][0] = 1;
    //  input_node_dims[0][1] = 3;
    //  input_node_dims[0][2] = 224;
    //  input_node_dims[0][3] = 224;
    //  CheckStatus(
    //      g_ort->CreateTensorWithDataAsOrtValue(gpu_memory_info_, data, inputLen, input_node_dims[0].data(), input_node_dims[0].size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &input_tensor));
    //  int is_tensor;
    //  CheckStatus(g_ort->IsTensor(input_tensor, &is_tensor));
    //  assert(is_tensor);
    //  //      uint64_t cp3 = getCurNs();

    //  // score model & input tensor, get back output tensor
    //  // TODO: This assumes a single input via a single tensor
    //  CheckStatus(
    //      g_ort->Run(session_, NULL, 
    //      (const char *const *)input_node_names.data(), 
    //      (const OrtValue *const *)&input_tensor, 1, 
    //      (const char *const *)output_node_names.data(), 1, &output_tensor));
    //  CheckStatus(g_ort->IsTensor(output_tensor, &is_tensor));
    //  //      uint64_t cp4 = getCurNs();
    //  assert(is_tensor);

    //  void *outputDataTensor;
    //  // Get pointer to output tensor float value
    //  CheckStatus(g_ort->GetTensorMutableData(output_tensor, &outputDataTensor));

    //  // copy to outputData (as outputDataTensor is owned by ort)
    //  size_t len_tensor_in_bytes = getTensorSize(output_node_dims[0]) * sizeof(float);
    //  //memcpy(outputData, (void*) outputDataTensor, len_tensor_in_bytes);

    //  //      uint64_t cp5 = getCurNs();
    //  //      std::cout<<(cp2-cp1)/1E6<<"  "<<(cp3-cp2)/1E6<<"   "<<(cp4-cp3)/1E6<<"   "<<(cp5-cp4)/1E6<<"\n";

    //  g_ort->ReleaseValue(input_tensor);
    //  g_ort->ReleaseValue(output_tensor);
    //  input_tensor = NULL;
    //  output_tensor = NULL;
    //  return len_tensor_in_bytes;
    //}
};
