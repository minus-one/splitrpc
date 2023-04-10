// Copyright (c) Adithya Kumar, The Pennsylvania State University. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string.h>
#include <iostream>
#include <sstream>
#include "spdlog/spdlog.h"
#include "debug_utils.h"

#define ACCESS_ONCE(x) (*(volatile typeof(x) *)&(x))

// Reads parameter passed to the application as environment variables
template<typename T>
static T readEnvInfo(const char* name, T defVal) {
  const char* opt = getenv(name);
  if (!opt) {
    spdlog::info("ENV(default): {} := {}", name, defVal);
    return defVal;
  }
  std::stringstream ss(opt);
  if (ss.str().length() == 0) return defVal;
  T res;
  ss >> res;
  if (ss.fail()) {
    spdlog::error("ENV: {} value := {} could not be parsed, using default", name, opt);
    return defVal;
  }
  spdlog::info("ENV: {} := {}", name, opt);
  return res;
}

static std::string getBasePath() {
  return readEnvInfo<std::string>("P2P_PATH", "./");
}

static std::string getDatasetBasePath() {
  return readEnvInfo<std::string>("P2P_DATA_SET_PATH", "../test_files/");
}

static inline uint16_t get_cuda_device_id() {
    return readEnvInfo<uint16_t>("P2P_RPC_CUDA_DEVICE_ID", 0);
}

static inline size_t get_req_size() {
  return readEnvInfo<size_t>("P2P_RPC_REQ_SIZE", 1024);
}

static inline size_t get_resp_size() {
  return readEnvInfo<size_t>("P2P_RPC_RESP_SIZE", 1024);
}

static inline std::string get_server_mac() {
  return readEnvInfo<std::string>("P2P_RPC_SERVER_MAC", std::string("b8:ce:f6:cc:6a:52"));
}

static inline std::string get_server_ip() {
  return readEnvInfo<std::string>("P2P_RPC_SERVER_IP", std::string("192.168.25.1"));
}

static inline std::string get_server_port() {
  return readEnvInfo<std::string>("P2P_RPC_SERVER_PORT", std::string("50051"));
}

static inline std::string get_client_ip() {
  return readEnvInfo<std::string>("P2P_RPC_CLIENT_IP", std::string("192.168.25.2"));
}

static inline std::string get_client_port() {
  return readEnvInfo<std::string>("P2P_RPC_CLIENT_PORT", std::string("50052"));
}

static inline std::string get_ort_model_name() {
  return readEnvInfo<std::string>("P2P_RPC_ORT_MODEL_NAME", std::string("resnet50v2batched.onnx"));
}

static inline uint16_t get_gpu_copy_type() {
  return readEnvInfo<int16_t>("P2P_RPC_GPU_COPY_TYPE", 1);
}

// 0 = No zero-copy, 1 = Zerocopy enabled
static inline int16_t is_zerocopy_mode() {
  return readEnvInfo<int16_t>("P2P_RPC_ZEROCOPY_MODE", 0);
} 

// 0 = sync server, 1 = async server
static inline int16_t get_server_mode() {
  return readEnvInfo<int16_t>("P2P_RPC_SERVER_MODE", 0);
}

// RPC_MTU = Actual amount of payload per-pkt
// FIXME: Add checks here i.e. MTU <= 9000
static inline size_t get_rpc_mtu() {
  return readEnvInfo<size_t>("P2P_RPC_MTU", 1024);
}
static const size_t RPC_MTU = get_rpc_mtu();

static inline int16_t get_ort_batch_size() {
  return readEnvInfo<int16_t>("P2P_RPC_ORT_BATCH_SIZE", 1);
}
