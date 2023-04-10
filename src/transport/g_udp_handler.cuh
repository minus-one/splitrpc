// Copyright (c) Adithya Kumar, The Pennsylvania State University. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include "p2p_bufs.h"
#include "p2p_buf_pool.h"

struct pp_params {
  struct p2p_hbufs hdr_bufs;
}__attribute__((packed));

struct pp_handler_ctx {
  struct pp_params *_stub;
  uint32_t *door_bell;
  cudaStream_t g_stream;
  cudaEvent_t work_complete;
  uint8_t launch_type; // 1 = Kernel Launch, 2 = Persistent kernel launch
};

extern "C" {
  pp_handler_ctx* setup_g_pp();
  void stop_g_pp(pp_handler_ctx*);
  void do_g_pp(pp_handler_ctx*, struct p2p_hbufs *h_wi);
}
