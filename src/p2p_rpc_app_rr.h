// Copyright (c) Adithya Kumar, The Pennsylvania State University. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "p2p_rpc_app_rr_mem_pool.h"
#include "p2p_buf_pool.h"
#include "p2p_rpc_app_ctx.h"
#include "p2p_rpc_rr_ng.h"
#include "debug_utils.h"
#include "gdr_mem_manager.h"
#include "g_utils.cuh"
#include "p2p_rpc_tring.h"

class P2pRpcAppRr {
  public:
    // Pointers to the parameters of the App - Both input and output
    // h: host accessible, d: device accessible
    g_params *h_stub;
    g_params *d_stub;

    volatile uint32_t *h_state;
    volatile uint32_t *d_state;

    size_t req_size, resp_size;

    // Valid only in RX_COMPLETE, WORK_COMPLETE states
    void *rpc_rr;

    int rr_idx;

    P2pRpcAppRr() {
      h_stub = NULL;
      d_stub = NULL;
      h_state = NULL;
      d_state = NULL;
      req_size = 0;
      resp_size = 0;
      rpc_rr = NULL;
      rr_idx = 0;
    }

    void print_app_rr_info()
    {
      printf("app_rr: %p, rr_idx: %d, h_stub: %p, req: %p, resp: %p\n",
          (void*)this, rr_idx, h_stub, h_stub->req, h_stub->resp);
    }

    void set_dummy_input_on_gpu()
    {
      int dummy_val = 1;
      printf("DummyInput: app_rr: %p, rr_idx: %d, h_stub: %p, req: %p, val: %d\n", 
          (void*)this, rr_idx, (void*)h_stub, (void*)h_stub->req, dummy_val);
      SetDummyData(h_stub->req, req_size, dummy_val);
    }

    void set_dummy_output_on_gpu()
    {
      int dummy_val = 0;
      printf("Output: app_rr: %p, rr_idx: %d, h_stub: %p, resp: %p, val: %d\n", 
          (void*)this, rr_idx, (void*)h_stub, (void*)h_stub->resp, dummy_val);
      SetDummyData(h_stub->resp, resp_size, dummy_val);
    }

    void dump_input_on_gpu() 
    {
      printf("Input: app_rr: %p, rr_idx: %d, h_stub: %p, req: %p\n", 
          (void*)this, rr_idx, (void*)h_stub, (void*)h_stub->req);
      g_floatDump(h_stub->req, req_size);
      g_intDump(h_stub->req, req_size);
    }

    void dump_output_on_gpu() 
    {
      printf("Output: app_rr: %p, rr_idx: %d, h_stub: %p, resp: %p\n", 
          (void*)this, rr_idx, (void*)h_stub, (void*)h_stub->resp);
      g_floatDump(h_stub->resp, resp_size);
      g_intDump(h_stub->resp, resp_size);
    }
};

class P2pRpcAppRrPool {
  private:
    int pool_size;
    int device_id;

    g_params *h_stubs;
    g_params *d_stubs;
    P2pRpcTring *req_pool, *resp_pool;

    volatile uint32_t *h_states;
    volatile uint32_t *d_states;

    size_t app_rr_pi_idx;

  public:
    size_t req_size, resp_size;

    P2pRpcAppRr *rr_pool;
    P2pRpcAppRrMemPool *rr_mem_pool;
    //P2pRpcAppRrSimpleMemPool *rr_mem_pool;

    inline int get_pool_size()
    {
      return pool_size;
    }

    inline P2pRpcAppRr* get_app_rr_pool()
    {
      return rr_pool;
    }

    inline P2pRpcAppRr* get_app_rr(int idx)
    {
      if(idx >= 0 && idx < pool_size) {
        return &rr_pool[idx];
      }
      return NULL;
    }

    inline void init_pool()
    {
      app_rr_pi_idx = 0;
    }

    // Not thread-safe
    inline P2pRpcAppRr* get_next()
    {
      P2pRpcAppRr* app_rr = &rr_pool[app_rr_pi_idx];
      app_rr->h_stub->req = (uint8_t*)req_pool->get_next(req_size); 
      app_rr->h_stub->resp = (uint8_t*)resp_pool->get_next(resp_size);
      app_rr_pi_idx = (app_rr_pi_idx + 1) % pool_size;
      return app_rr;
    }

    P2pRpcAppRrPool(int _pool_size, int _device_id, size_t _req_size, size_t _resp_size)
    {
      pool_size = _pool_size;
      device_id = _device_id;
      req_size = _req_size;
      resp_size = _resp_size;
       
      rr_pool = new P2pRpcAppRr[pool_size];
      rr_mem_pool = new P2pRpcAppRrMemPool(pool_size, device_id, req_size, resp_size);
      //rr_mem_pool = new P2pRpcAppRrSimpleMemPool(pool_size, device_id, req_size, resp_size);
      req_pool = new P2pRpcTring(rr_mem_pool->get_req_addr_range(), rr_mem_pool->get_req_addr_pool_size());
      resp_pool = new P2pRpcTring(rr_mem_pool->get_resp_addr_range(), rr_mem_pool->get_resp_addr_pool_size());
      h_states = rr_mem_pool->get_doorbells_host(); 
      d_states = rr_mem_pool->get_doorbells_device(); 

      //h_stubs = BufItemPool<g_params>::create_buf_item_pool(pool_size, device_id);
      //d_stubs = BufItemPool<g_params>::get_dev_ptr(h_stubs);
      h_stubs = new g_params[pool_size];
      d_stubs = h_stubs;

///////////////////////////////////// END OF ALL MEMORY ALLOCATIONS ////////////////////////////////////////////
      // Set up each app_rr
      TRACE_PRINTF("==================================================================================\n");
      TRACE_PRINTF("Setting up P2pRpcAppRrPool - %d items...\n", pool_size);
      for(int i = 0 ; i < pool_size ; i++) {
        P2pRpcAppRr *app_rr = &rr_pool[i];
        app_rr->req_size = req_size;
        app_rr->resp_size = resp_size;
        app_rr->h_stub = &h_stubs[i];
        app_rr->d_stub = &d_stubs[i];
        app_rr->h_state = &h_states[i];
        app_rr->d_state = &d_states[i];
        *app_rr->h_state = APP_RR_STATUS::FREE;
        app_rr->rpc_rr = NULL;
        app_rr->rr_idx = i;

        app_rr->h_stub->req = NULL;   // (uint8_t*)req_pool_addr_range + (i * req_size);
        app_rr->h_stub->resp = NULL;  // (uint8_t*)resp_pool_addr_range + (i * resp_size);

        TRACE_PRINTF("idx: %d, AppRr: %p, h_stub: %p, Req: %p, Resp: %p, Doorbell: %p\n",
            i, (void*)app_rr, (void*)app_rr->h_stub, 
            (void*)app_rr->h_stub->req, (void*)app_rr->h_stub->resp, (void*)app_rr->h_state);
      }
      TRACE_PRINTF("===================================================================================\n");
      app_rr_pi_idx = 0;
    }

    ~P2pRpcAppRrPool() {
      delete req_pool;
      delete resp_pool;
      delete rr_mem_pool;
      delete rr_pool;
      delete h_stubs;
      //BufItemPool<g_params>::delete_buf_item_pool(h_stubs, device_id);
    }
};

//CUmemGenericAllocationHandle tmp_handle = 0;
//CUdeviceptr tmp = 0;
//size_t tmp_size = PAGE_ROUND_UP(tot_mem_region_sz_va, granularity);
//doorbells_gdr_mm.input_size = tmp_size;
//
//checkCudaErrors(cuMemAddressReserve(&tmp, 2 * tmp_size, granularity, 0, 0));           
//TRACE_PRINTF("tmp: %p, tmp_size: %ld\n", (void*)tmp, tmp_size);

//checkCudaErrors(cuMemCreate(&tmp_handle, tmp_size, &allocProp, 0));
//checkCudaErrors(cuMemMap(tmp, tmp_size, 0, tmp_handle, 0));
//checkCudaErrors(cuMemMap(tmp + tmp_size, tmp_size, 0, tmp_handle, 0));
//checkCudaErrors(cuMemSetAccess(tmp, 2 * tmp_size, &accessDesc, 1));

//TRACE_PRINTF("tmp: %p, tmp_size: %ld\n", (void*)tmp, tmp_size);
//TRACE_PRINTF("tmp2: %p, tmp_size: %ld\n", (void*)(tmp + tmp_size), tmp_size);


