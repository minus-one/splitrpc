// Copyright (c) Adithya Kumar, The Pennsylvania State University. All rights reserved.
// Licensed under the MIT License.

#include "app.h"
#include "spdlog/spdlog.h"

size_t app_req_size;
size_t app_resp_size;

int CreateOrtRequest(void *request, size_t& len_in_bytes) {
  len_in_bytes = app_req_size; 
  //uint8_t *start = reinterpret_cast<uint8_t*>(request);
  //for(int i = 0 ; i < len_in_bytes ; i++)
  //  start[i] = i;

  float *start = reinterpret_cast<float*>(request);
  for(int i = 0 ; i < len_in_bytes/4 ; i++)
    start[i] = 5.0f;

  spdlog::info("Input: float:[0..4] -> [{}, {}, {}, {}, {}]", start[0], start[1], start[2], start[3], start[4]);
  return 0;
}

int CompleteOrtReply(void *replyData, size_t len_in_bytes) {
  static bool sanity_print = false;
  if(len_in_bytes != app_resp_size) {
    spdlog::info("OrtClient Error!, exp: {}, got: {} response\n", app_resp_size, len_in_bytes);
    return 1;
  }
  //hexDump("OutputBuf", replyData, len_in_bytes);
  uint8_t *start = reinterpret_cast<uint8_t*>(replyData);
  float* floatArr = (float*) replyData;
  spdlog::info("Output: float: [0..5] -> [{}, {}, {}, {}, {}]", floatArr[0], floatArr[1], floatArr[2], floatArr[3], floatArr[4]);

  //for(int i=0; i< 5;i++) {
  //  spdlog::info("float[{}] -> {}", i, floatArr[i]);
  //}
  //uint8_t* intArr= (uint8_t*) replyData;
  //spdlog::info("Output: int: [0..5] -> [{}, {}, {}, {}, {}]", intArr[0], intArr[1], intArr[2], intArr[3], intArr[4]);

  //for(int i=0; i< 5;i++) {
  //  spdlog::info("int[{}] -> {}", i, intArr[i]);
  //}
  return 0;
}

AppReq_cb app_req_cb = &CreateOrtRequest;
AppResp_cb app_resp_cb = &CompleteOrtReply;
