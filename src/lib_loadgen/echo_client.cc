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

  uint8_t ctr = 0;
  for(int i = 0 ; i < len_in_bytes; i++) {
    if(i % 8192 == 0) {
      start[i] = MAGIC_STRING[ctr];
      ctr = (ctr + 1) % 4;
    }
    else
      start[i] = 2;
  }

  //for(int i = 0 ; i < len_in_bytes / 8192 ; i++) {
  //  start[(i*8192)] = MAGIC_STRING[i % 4];
  //  for(int j = 1; j < 8192 ; j++) {
  //    start[(i*8192) + j] = 2;
  //  }
  //}
  return 0;
}

int CompleteEchoReply(void *replyData, size_t len_in_bytes) {
  if(len_in_bytes != app_resp_size)
    return 1;
  uint8_t *start = reinterpret_cast<uint8_t*>(replyData);
  uint8_t ctr = 0;
  for(int i = 0 ; i < len_in_bytes; i++) {
    if(i % 8192 == 0) {
      if(start[i] != MAGIC_STRING[ctr]) {
        spdlog::error("MAGIC STRING MISMATCH: idx: {:d}, Got: {:x}, Exp: {:x}",\
            (i), start[i], MAGIC_STRING[ctr]);
      }
      ctr = (ctr + 1) % 4;
    }
    else {
      if(start[i] != 2) {
        spdlog::error("MAGIC STRING MISMATCH: idx: {:d}, Got: {:x}, Exp: {:x}",\
            (i), start[i], 2);
      }
    }
  }

  //for(int i = 0 ; i < app_resp_size / 8192; i++) {
  //  if(start[(i*8192)] != MAGIC_STRING[i % 4]) {
  //    spdlog::error("MAGIC STRING MISMATCH: idx: {:d}, Got: {:x}, Exp: {:x}",\
  //        (i*8192), start[(i*8192)], MAGIC_STRING[i % 4]);
  //    return 1;
  //  }
  //  for(int j = 1 ; j < 8192; j++) {
  //    if(start[(i*8192) + j] != 2) {
  //      spdlog::error("MAGIC STRING MISMATCH: idx: {:d}, Got: {:x}, Exp: {:x}",\
  //          (i*8192 + j), start[(i*8192) + j], 2);
  //      return 1;
  //    }
  //  }
  //}
  return 0;
}

AppReq_cb app_req_cb = &CreateEchoRequest;
AppResp_cb app_resp_cb = &CompleteEchoReply;
