// Copyright (c) Adithya Kumar, The Pennsylvania State University. All rights reserved.
// Licensed under the MIT License.

#include "app.h"
#include "spdlog/spdlog.h"

static uint8_t MAGIC_STRING[4] = {0xDE, 0xAD, 0xBE, 0xEF};
size_t app_req_size;
size_t app_resp_size;

int CreateEchoRequest(void *request, size_t& len_in_bytes) {
  uint8_t *start = reinterpret_cast<uint8_t*>(request);
  len_in_bytes = app_req_size;

  for(int i = 0 ; i < len_in_bytes ; i++) {
    start[i] = MAGIC_STRING[i % 4];
  }
  return 0;
}

int CompleteEchoReply(void *replyData, size_t len_in_bytes) {
 if(len_in_bytes != app_resp_size)
    return 1;
  uint8_t *start = reinterpret_cast<uint8_t*>(replyData);
  for(int i = 0 ; i < app_resp_size; i++) {
    if(start[i] != MAGIC_STRING[i % 4] + 1) {
      spdlog::error("MAGIC STRING MISMATCH: idx: {:d}, Got: {:x}, Exp: {:x}",\
          i, start[i], MAGIC_STRING[i % 4] + 1);
      return 1;
    }
  }
  return 0;
}

AppReq_cb app_req_cb = &CreateEchoRequest;
AppResp_cb app_resp_cb = &CompleteEchoReply;
