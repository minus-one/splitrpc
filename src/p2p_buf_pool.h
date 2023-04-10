// Copyright (c) Adithya Kumar, The Pennsylvania State University. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <unordered_map>
#include <cstring>
#include "gdr_mem_manager.h"

// Not thread safe
// DOES Not implement a pool
template <class T>
class BufItemPool {
  static std::unordered_map<T*, gdr_memseg_info*> buf_pool_info;
  public:

  // This always returns host pointer
  // Type determines the type of memory allocated, X=-1 => HostMemory, X>=0 => GDR-MEM on device id X
  static T* create_buf_item_pool(int num, int type = -1) {
    T *h_wi;
    gdr_memseg_info *_gdr_buf = NULL;
    if(type >= 0) {
      _gdr_buf = (gdr_memseg_info*)malloc(sizeof(gdr_memseg_info));
      _gdr_buf->input_size = sizeof(T) * num;
      gdr_mem_manager *G = get_gdr_mem_manager(type);
      G->alloc(_gdr_buf);
      h_wi = (T*)(_gdr_buf->phost_ptr);
      TRACE_PRINTF("BufItemPool GDR \
          _gdr_buf: %p, h_wi: %p, d_wi: %p\n", \
          (void*)_gdr_buf, (void*)_gdr_buf->phost_ptr, (void*)_gdr_buf->pdev_addr);
    } else {
      h_wi = (T*)malloc(sizeof(T) * num);
      TRACE_PRINTF("BufItemPool HOST \
          h_wi: %p\n", (void*)h_wi);
    }
    BufItemPool<T>::buf_pool_info[h_wi] = _gdr_buf;
    std::memset(h_wi, 0, sizeof(T) * num);
    return h_wi;
  }

  static void delete_buf_item_pool(T* h_wi, int type = -1) {
    if(BufItemPool<T>::buf_pool_info.find(h_wi) != BufItemPool<T>::buf_pool_info.end()) {
      gdr_memseg_info *_gdr_buf = BufItemPool<T>::buf_pool_info[h_wi];
      if(_gdr_buf != NULL) {
#ifdef TRACE_MODE
        printf("BufItemPool::delete_buf_item_pool GDR cleanup h_wi: %p, gdr_buf: %p\n", h_wi, _gdr_buf);
#endif
        // FIXME: Get the appropriate device-id
        gdr_mem_manager *G = get_gdr_mem_manager(type);
        G->cleanup(_gdr_buf);
        delete(_gdr_buf);
        BufItemPool<T>::buf_pool_info.erase(h_wi);
      } else {
#ifdef TRACE_MODE
        printf("BufItemPool::delete_workitems HOST cleanup %p Skipping calling gdr-mem-manager for cleanup\n", h_wi);
#endif
        BufItemPool<T>::buf_pool_info.erase(h_wi);
        delete(h_wi);
      }
    }
  }

  static T* get_dev_ptr(T* h_wi) {
    if(BufItemPool<T>::buf_pool_info.find(h_wi) != BufItemPool<T>::buf_pool_info.end()) {
      if(BufItemPool<T>::buf_pool_info[h_wi] != NULL) {
        return (T*) BufItemPool<T>::buf_pool_info[h_wi]->pdev_addr;
      }
      else {
        printf("Warning! BufItem does not have a dev mapping\n");
        return h_wi;
      }
    }
    return NULL;
  }
};

template <class T>
std::unordered_map<T*, gdr_memseg_info*> BufItemPool<T>::buf_pool_info;
