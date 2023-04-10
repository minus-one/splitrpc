// Copyright (c) Adithya Kumar, The Pennsylvania State University. All rights reserved.
// Licensed under the MIT License.

#ifndef BASE_CLIENT_H
#define BASE_CLIENT_H

#include "stats_factory.h"
#include "common_defs.h"
#include "app.h"
#include <atomic>

// 3 states of a request
typedef enum {
  R_CREATE = 1,
  R_SENT = 2,
  R_RECV = 3
} R_STATES;

class BaseClient {
  public:
    std::atomic<uint32_t> n_r_create, n_r_sent, n_r_recv;
    StatsManager* StatsCollector;
    uint64_t startTimeNs, endTimeNs;
    bool closedLoop;
    volatile bool exit_flag;
    bool verify_run;
    uint32_t maxRequests, n_r_warmup;

    // Functions to be called for creating and consuming payloads
    AppReq_cb req_cb;
    AppResp_cb resp_cb;

    // struct for keeping state and data information
    struct CallToken {
      void *call_id; // Refers to the addr of token used by the rpc (p2p_rpc_rr)
      uint64_t size;
      bool warmup;
      uint8_t call_state; // R_STATES
      //uint32_t req_idx;
      uint64_t dispatchNs, completionNs;

      uint8_t *req_buf;
      uint8_t *resp_buf;
    };
    size_t max_req_size;
    size_t max_resp_size;

    CallToken **SmRequests; // Starts from 1, index 0 is NULL
    CallToken *AllTokens;
    uintptr_t token_start_addr, token_end_addr;

    explicit BaseClient(StatsManager* stats) : 
      StatsCollector(stats) {
      n_r_create = 0;
      n_r_sent = 0;
      n_r_recv = 0;
      StatsCollector->addMeasurementType("sojournTime");
      endTimeNs = 0;
      maxRequests = UINT_MAX;
      closedLoop = false; 
      max_req_size = MAX_PAYLOAD_SIZE;
      max_resp_size = MAX_PAYLOAD_SIZE;
      n_r_warmup = 0;
      exit_flag = false;
      verify_run = false;
    }

    ~BaseClient() {}
    
    void SetMaxRequests(uint32_t numRequests) {
      maxRequests = numRequests;
      AllTokens = new CallToken[maxRequests+1];
      SmRequests = new CallToken*[maxRequests+1];
      SmRequests[0] = NULL;
      for(int i = 1; i <= maxRequests; i++)
        SmRequests[i] = &(AllTokens[i]);
      token_start_addr = (uintptr_t)(AllTokens + 1);
      token_end_addr = (uintptr_t)(AllTokens + maxRequests + 1);
      printf("Tokens: Start: %p, End: %p\n", (void*)token_start_addr, (void*)token_end_addr);
    }


    // Quick way to determine if the call is a valid one
    inline bool isValidToken(void *call_tag) {
      if((uintptr_t)call_tag >= token_start_addr && (uintptr_t)call_tag <= token_end_addr)
        return true;
      return false;
    }

    void SetMaxPayloadSize(size_t _max_req_size = MAX_PAYLOAD_SIZE, 
    size_t _max_resp_size = MAX_PAYLOAD_SIZE) {
      max_req_size = _max_req_size;
      max_resp_size = _max_resp_size;
    }

    void SetClosedLoop() {
      closedLoop = true;
    }

    void SetWarmupRequests(uint32_t _n_r_warmup) {
      n_r_warmup = _n_r_warmup;
    }

    void SetReqRespCb(AppReq_cb _req_cb, AppResp_cb _resp_cb) {
      if(_req_cb)
        req_cb = _req_cb;
      if(_resp_cb)
        resp_cb = _resp_cb;
    }

    void SetVerifyRun(uint16_t _verify_run) {
      if(_verify_run > 0)
        verify_run = true;
      else
        verify_run = false;
    }

    void exit_client() {
      exit_flag = true;
    }

    bool is_client_listening() {
      return ! exit_flag;
    }

    virtual std::string getName() {
      return std::string("BaseClient");
    }

    // To be implemented by the client-type
    virtual void* CreateRequest(void* RequestParams) = 0;
    virtual void SchedRequest(void* reqTag) = 0;
    virtual void CompleteReqListener() = 0;

    virtual void PreProcessRequests() {

    }

    virtual void PostProcessResponses() {
      // Post process all responses separately
      // Collect stats
      int recorded_calls = 0;
      int warmup_calls = 0;
      int incorrect_state_calls = 0;
      uint64_t firstReqDispatchNs = UINT64_MAX, lastReqComplNs = 0;
      for(int i = 0 ; i <= maxRequests; i++) {
        CallToken *call = SmRequests[i];
        if(call == NULL)
          continue;
        if(call->call_state != R_STATES::R_RECV) {
          //spdlog::error("P2pRpcAppClient Call: {}, Sm-idx: {}, in incorrect state: {}", (void*)call, i, call->call_state);
          incorrect_state_calls++;
        } else {
          if(call->warmup == true)
            warmup_calls++;
          else if(StatsCollector) {
            firstReqDispatchNs = std::min(firstReqDispatchNs, call->dispatchNs);
            lastReqComplNs = std::max(lastReqComplNs, call->completionNs);
            StatsCollector->recordEvent("sojournTime",
                (call->completionNs - call->dispatchNs));
            recorded_calls++;
          }
        } 
      }
      double throughput_est = static_cast<double>(recorded_calls / ((lastReqComplNs - firstReqDispatchNs)/1E9));
      StatsCollector->trackStatInfo("firstReqDispatchNs", firstReqDispatchNs);
      StatsCollector->trackStatInfo("lastReqComplNs", lastReqComplNs);
      StatsCollector->trackStatInfo("throughput_est", throughput_est);
      StatsCollector->trackStatInfo("n_r_sent", uint64_t(n_r_sent.load()));
      StatsCollector->trackStatInfo("n_r_recv", uint64_t(n_r_recv.load()));
      StatsCollector->trackStatInfo("recorded", uint64_t(recorded_calls));
      StatsCollector->trackStatInfo("warmup_calls", uint64_t(warmup_calls));
      StatsCollector->trackStatInfo("incorrect_state_calls", uint64_t(incorrect_state_calls));
      spdlog::info("{} Calls, Total: {}, Sent: {}, Recvd: {}, Recorded: {}, Warmup: {}, IncorrectState: {}", 
          getName(), maxRequests, n_r_sent, n_r_recv, 
          recorded_calls, warmup_calls, incorrect_state_calls);
    }
};

#endif /* BASE_CLIENT_H */
