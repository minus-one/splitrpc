// Copyright (c) Adithya Kumar, The Pennsylvania State University. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include "config_utils.h"
#include "p2p_bufs.h"
#include <emmintrin.h>

#include <cuda.h>
#include <cuda_runtime.h>
#define CUDA_DRIVER_API
#include <helper_functions.h>
#include <helper_cuda.h>
#include "g_utils.cuh"

#include "p2p_rpc_app_stub.h"

static inline int8_t get_work_launch_type() {
  return readEnvInfo<int16_t>("P2P_RPC_WORK_LAUNCH_TYPE", 1);
}

enum APP_RR_STATUS {FREE, RX_COMPLETE, WORK_COMPLETE, TX_COMPLETE };

template <class T>
class MetaAppCtx
{
  public:
    MetaAppCtx() 
    {
      curr_batch_size = 1;
      next_batch_size = 1;
    }

    // This is used to make the app-run wait on a specific RR's state
    inline void AppRrRunNotify(volatile uint32_t *d_state, int value) {
      app_run_notifier(work_stream, d_state, value);
    }

    // This is used to make the app-run wait on it's doorbell
    inline void AppRunNotify(int value) {
      app_run_notifier(work_stream, d_door_bell, value);
    }

    // This used to set the door-bell once some op on stream is complete
    inline void AppCompleteNotify(uint32_t value = 2) {
      app_complete_notifier(work_stream, d_door_bell, value);
    }

    inline void NotifyAppRunStart() {
      *ACCESS_ONCE(door_bell) = curr_batch_size;
      _mm_mfence();
    }

    inline bool IsAppRunComplete() {
      return (*ACCESS_ONCE(door_bell) == 1);
    }

    inline void ResetAppRunStatus() {
      *ACCESS_ONCE(door_bell) = UINT_MAX;
      _mm_mfence();
    }

    inline void WaitForAppRunComplete() {
      if (launch_type != 3) {
        checkCudaErrors(cudaStreamSynchronize(work_stream));
      }
      else if(launch_type == 3) {
        while (*ACCESS_ONCE(door_bell) != 2)
          ;
      }
    }

    // Used to convey to the app
    T *h_stub;
    T *d_stub; 

    // Device execution door-bells
    uint32_t *door_bell;
    uint32_t *d_door_bell; // Device side ptr

    // If the app wants to store some internal ctx, it can use this
    void *app_ctx_internal;

    // This is for the current run
    int curr_batch_size;
    // Set this to be used in the next run
    int next_batch_size;

    cudaStream_t work_stream;
    cudaEvent_t work_complete;
    struct p2p_rpc_rr *curr_rr;

    // 0 = No Work, 1 = Kernel Launch, 2 = CDP 
    // 3 = P.T + CDP, 4 = Cuda graphs
    uint8_t launch_type;
    int device_id;

    // CUDAgraph related
    bool graphCreated = false;
    cudaGraph_t graph;
    cudaGraphExec_t instance;

    //uint64_t startNs, endNs;
};

typedef MetaAppCtx<g_params> AppCtx;
using AppInitCB = int (*)(AppCtx *);
using AppRunCB = int (*)(AppCtx *);
using AppCleanupCB = int(*)(AppCtx *);
using AppRunAsyncCB = int (*)(AppCtx*);
using AppRunWaitCB = int (*)(AppCtx *);
using AppCompleteCB = int (*)(AppCtx *);
