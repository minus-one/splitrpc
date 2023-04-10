// Copyright (c) Adithya Kumar, The Pennsylvania State University. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstdint>
#include <stdlib.h>

// This is a simple ring_buf wrapper
class P2pRpcTring {
  private:
    //void *base_addr_range;
    uintptr_t base_addr_range;
    size_t ring_capacity;
    size_t prev_offset;
    uintptr_t prev_addr;
    //void *prev_addr;
  public:
    P2pRpcTring(void *base_addr, size_t capacity) 
    {
      base_addr_range = (uintptr_t)base_addr;
      ring_capacity = capacity;
      prev_offset = 0;
      prev_addr = base_addr_range;
    }

    inline void* get_next(size_t req_size)
    {
       void *ret = (void*)prev_addr;
       prev_offset = (prev_offset + req_size) % ring_capacity;
       prev_addr = base_addr_range + prev_offset;
       return ret;
    }
};
