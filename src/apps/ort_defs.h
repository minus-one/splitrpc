// Copyright (c) Adithya Kumar, The Pennsylvania State University. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/session/onnxruntime_c_api.h"
#include <array>
#include <vector>

// The batch-sizes will be 1, 2...MAX_BS for now
const int MAX_BS = 8; // Number of possible batch-sizes 

typedef std::vector<const char*> ORT_NODE_NAMES;
typedef std::vector<int64_t> ORT_DIMS;                              // Shape of a single ORT_NODE
typedef std::vector<ORT_DIMS> ORT_NODE_DIMS;                        // Set of input/output shapes 
typedef std::vector<size_t> ORT_NODE_SIZES;                         // Set of input/output sizes
typedef std::vector<enum ONNXTensorElementDataType> ORT_NODE_TYPES; // Set of input/output types

// This is defined per input / output for a model
class ORT_IO_INFO
{
  public:
    size_t num_nodes;
    ORT_NODE_NAMES io_names;
    ORT_NODE_DIMS io_dims;
    ORT_NODE_SIZES io_sizes;
    ORT_NODE_TYPES io_types;
    size_t tot_io_size;

    // Warning, this might lose data
    void set_num_nodes_and_resize_info(size_t _num_nodes) {
      num_nodes = _num_nodes;
      io_names.resize(num_nodes);
      io_dims.resize(num_nodes);
      io_sizes.resize(num_nodes);
      io_types.resize(num_nodes);
    }

    size_t get_num_nodes() {
      return num_nodes;
    }

    const char** get_node_names() {
      return io_names.data();
    }

    size_t get_node_size(size_t idx) {
      return io_sizes[idx];
    }

    enum ONNXTensorElementDataType get_node_type(size_t idx) {
      return io_types[idx];
    }

    size_t get_node_shape_len(size_t idx) {
      return io_dims[idx].size();
    }

    int64_t* get_node_shape(size_t idx) {
      return io_dims[idx].data();
    }

    void calc_tot_io_size() {
      tot_io_size = 0;
      for (size_t i = 0; i < num_nodes; i++) {
        tot_io_size += io_sizes[i];
      }
    }

    void print_io_info() {
      if(num_nodes != io_names.size() ||
          num_nodes != io_sizes.size() ||
          num_nodes != io_types.size() ||
          num_nodes != io_dims.size()) {
        printf("Error in printing IO_INFO, num_nodes and IO_INFO sizes do not match\n");
      }
      // iterate over all input nodes
      for (size_t i = 0; i < num_nodes; i++) {
        // print input node names
        printf("%zu : name= %s, type= %d, Shape -> ( ", i, io_names[i], io_types[i]);
        for (size_t j = 0; j < io_dims[i].size(); j++)
          printf("%jd ", io_dims[i][j]);
        printf(" ), Size= %lu\n", io_sizes[i]);
      }
      printf("Total size: %ld\n", tot_io_size);
    }
};

// Given a model name, generates a map of the dimensions for different batch-size configurations
bool setup_io_dim_map(std::string model_name, 
    std::array<ORT_IO_INFO, MAX_BS + 1>& input_io_info_map, 
    std::array<ORT_IO_INFO, MAX_BS + 1>& output_io_info_map)
{
  if(model_name.find("resnet") != std::string::npos) {
    for(int i = 1; i <= MAX_BS; i++) {
      ORT_IO_INFO &input_io_info = input_io_info_map[i];
      ORT_IO_INFO &output_io_info = output_io_info_map[i];
      // Copy all essentials like name and num_nodes
      input_io_info = input_io_info_map[0];
      output_io_info = output_io_info_map[0];

      // Setting input specific params
      ORT_NODE_DIMS &input_dims = input_io_info.io_dims; 
      input_dims.resize(1);
      input_dims[0].resize(4);
      input_dims[0][0] = i;
      input_dims[0][1] = 3;
      input_dims[0][2] = 224;
      input_dims[0][3] = 224;
      ORT_NODE_SIZES &input_sizes = input_io_info.io_sizes;
      input_sizes.resize(1);
      input_sizes[0] = i * 3 * 224 * 224 * sizeof(float);
      ORT_NODE_TYPES &input_types = input_io_info.io_types;
      input_types.resize(1);
      input_types[0] = ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
      input_io_info.calc_tot_io_size();

      // Setting output specific params
      ORT_NODE_DIMS &output_dims = output_io_info.io_dims;
      output_dims.resize(1);
      output_dims[0].resize(2);
      output_dims[0][0] = i;
      output_dims[0][1] = 1000;
      ORT_NODE_SIZES &output_sizes = output_io_info.io_sizes;
      output_sizes.resize(1);
      output_sizes[0] = i * 1000 * sizeof(float);
      ORT_NODE_TYPES &output_types = output_io_info.io_types;
      output_types.resize(1);
      output_types[0] = ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
      output_io_info.calc_tot_io_size();
    }
    return true;
  } else if(model_name.find("bert") != std::string::npos) {
    for(int i = 1; i <= MAX_BS; i++) {
      ORT_IO_INFO &input_io_info = input_io_info_map[i];
      ORT_IO_INFO &output_io_info = output_io_info_map[i];
      // Copy all essentials like name and num_nodes
      input_io_info = input_io_info_map[0];
      output_io_info = output_io_info_map[0];

      ORT_NODE_DIMS &input_dims = input_io_info.io_dims;
      input_dims.resize(4);
      input_dims[0].resize(1);
      input_dims[0][0] = i;
      input_dims[1].resize(2);
      input_dims[1][0] = i;
      input_dims[1][1] = 256;
      input_dims[2].resize(2);
      input_dims[2][0] = i;
      input_dims[2][1] = 256;
      input_dims[3].resize(2);
      input_dims[3][0] = i;
      input_dims[3][1] = 256;

      ORT_NODE_SIZES &input_sizes = input_io_info.io_sizes;
      input_sizes.resize(4);
      input_sizes[0] = i * sizeof(uint64_t);
      input_sizes[1] = i * 256 * sizeof(uint64_t);
      input_sizes[2] = i * 256 * sizeof(uint64_t);
      input_sizes[3] = i * 256 * sizeof(uint64_t);

      ORT_NODE_DIMS &output_dims = output_io_info.io_dims;
      output_dims.resize(3);
      output_dims[0].resize(2);
      output_dims[0][0] = i;
      output_dims[0][1] = 256;
      output_dims[1].resize(2);
      output_dims[1][0] = i;
      output_dims[1][1] = 256;
      output_dims[2].resize(1);
      output_dims[2][0] = i;

      ORT_NODE_SIZES &output_sizes = output_io_info.io_sizes;
      output_sizes.resize(3);
      output_sizes[0] = i * 256 * sizeof(float);
      output_sizes[1] = i * 256 * sizeof(float);
      output_sizes[2] = i * sizeof(uint64_t);
    }
    return true;
  }

  for(int i = 1 ; i <= MAX_BS; i++) {
    ORT_IO_INFO &input_io_info = input_io_info_map[i];
    ORT_IO_INFO &output_io_info = output_io_info_map[i];
    // Copy all essentials like name and num_nodes
    input_io_info = input_io_info_map[0];
    output_io_info = output_io_info_map[0];
  }
  return false;
}
