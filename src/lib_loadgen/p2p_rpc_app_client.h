// Copyright (c) Adithya Kumar, The Pennsylvania State University. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "base_client.h"
#include <bits/stdc++.h>

#include "transport/dpdk_utils.h"
#include "transport/dpdk_init.h"
#include "transport/dpdk_transport.h"

#include "p2p_rpc.h"
#include "p2p_rpc_conn_info.h"
#include "p2p_rpc_rr.h"
#include "p2p_rpc_rr_pool.h"

// A P2P-RPC Client for all applications
/**
 * The RPC client is a wrapper that gets raw bytes that need to be sent
 * from the application, slaps certain header information (to identify the
 * request) along with some timestamps, crafts it as a UDP packets and TXs it.
 * The receiver RXs incoming packet, verifies it is on a predefined port, and matches the
 * request based on the identifier that was sent and calculates latency
 * statistics.
 */
class P2pRpcAppClient : public BaseClient {
  private:
    struct dpdk_ctx *ctx;
    struct p2p_rpc_conn_info *conn_info;

    std::string src_mac_str, dst_mac_str,
      src_ip_str, dst_ip_str,
      src_port_str, dst_port_str;
    int src_port, dst_port;
    mac_addr src_mac, dst_mac;

  P2pRpcRRPool rr_pool;
#ifdef PROFILE_MODE
  uint64_t s_startNs, r_startNs;
  int numMetrics;
  std::vector<uint64_t> SendDelay, RecvDelay;
#endif

  public:
    explicit 
      P2pRpcAppClient(std::string m_uri, StatsManager* stats) 
      : BaseClient(stats) 
      {
        ctx = new struct dpdk_ctx;
        ctx->nic_port = get_dpdk_port();
        ctx->queue_id = 0;
        ctx->mem_alloc_type = HOST_MEM_ONLY;
        ctx->device_id = -1;
        if (init_dpdk_ctx(ctx) == 0) {
          spdlog::error("Cannot setup DPDK port");
          return;
        }

        std::stringstream ss_tmp(m_uri);
        std::getline(ss_tmp, src_mac_str, ',');
        std::getline(ss_tmp, src_ip_str, ',');
        std::getline(ss_tmp, src_port_str, ',');
        std::getline(ss_tmp, dst_mac_str, ',');
        std::getline(ss_tmp, dst_ip_str, ',');
        std::getline(ss_tmp, dst_port_str, ',');

        src_port = std::stoi(src_port_str);
        dst_port = std::stoi(dst_port_str);
        src_mac = mac_from_string(src_mac_str);
        dst_mac = mac_from_string(dst_mac_str);
        
        spdlog::info("SRC: MAC: {}, IP: {}, Port: {}",
            mac_to_string(src_mac), src_ip_str, src_port);
        spdlog::info("DST: MAC: {}, IP: {}, Port: {}",
            mac_to_string(dst_mac), dst_ip_str, dst_port);

        conn_info = 
            init_conn_info(src_port, dst_port,
            src_ip_str.c_str(), dst_ip_str.c_str(),
            src_mac, dst_mac);
        //hexDump("This is the CONNINFO Template", (void*)&conn_info->hdr_template, RPC_HEADER_LEN);
      }

    ~P2pRpcAppClient()
    {
      //conn_info_teardown
      release_conn_info(conn_info);
      //dpdk_teardown
      stop_dpdk(ctx);
      delete ctx;
    }

    virtual std::string getName() {
      return std::string("P2pRpcClient");
    }

    virtual void* CreateRequest(void *RequestParams);
    virtual void SchedRequest(void *reqTag);

    // Loop while listening for completed responses.
    // Prints out the response from the server.
    virtual void CompleteReqListener();

    void PreProcessRequests() override ;
    void PostProcessResponses() override;
};

// This is just a dummy op. uses a call
void* P2pRpcAppClient::CreateRequest(void *RequestParams)
{
  CallToken *call = NULL;
  if(RequestParams == NULL) {
    // Create a new call with new request
    uint32_t my_req_idx = ++n_r_create;
    call = SmRequests[my_req_idx];
    call->call_id = NULL;
  } else {
    // Recyle a call and save stats 
    printf("Recycling a call. Did you want this to happen?\n");
    call = (CallToken*)RequestParams;
    if(StatsCollector && (call->warmup != true)) {
      StatsCollector->recordEvent("sojournTime",
            (call->completionNs - call->dispatchNs));
    }
  }
  call->call_state = R_STATES::R_CREATE;
  return static_cast<void*>(call);
}

// TX the request
// Schedules the next request based on index
// (or) schedules a specific request
void P2pRpcAppClient::SchedRequest(void *reqTag) 
{
  uint32_t my_req_idx = ++n_r_sent;
  if(my_req_idx > maxRequests)
    return;

  CallToken *call = NULL;
  if(reqTag == NULL) {
    call = SmRequests[my_req_idx];
  } else {
    call = static_cast<CallToken*>(reqTag);
  }

  if(my_req_idx <= n_r_warmup)
    call->warmup = true;
  else
    call->warmup = false;
  call->call_state = R_STATES::R_SENT;

  /*
   * Steps to send a request
   * 1. Allocate the necessary transport mbufs
   * 2. Set the header for each mbuf
   * 3. Copy the request to the mbuf (scatter)
   * 4. Set dispatch time-stamp
   * 5. Transmit the mbufs
   */
  p2p_rpc_rr *rr_item = rr_pool.consume_rr(); 
  rr_item->req_token = (uint64_t)(void*)call;
  // Instead of loading data everytime, just set the size - the payload wouldn't have gone away
  rr_item->req_size = max_req_size; 
  if(rr_alloc_mbufs(ctx, rr_item, rr_item->req_size) == 0) {
    spdlog::error("Error allocating request {}", (void*)call);
    exit(1);
  }
  if(rr_set_hdr(conn_info, rr_item, rr_item->req_size) == 0) {
    spdlog::error("Error setting header request {}", (void*) call);
    exit(1);
  }
  if(rr_req_to_bufs(ctx, rr_item) != rr_item->req_size) {
    spdlog::error("Error in copying payload {}", (void*) call);
    exit(1);
  }
  call->dispatchNs = getCurNs();
#ifdef PROFILE_MODE
  s_startNs = getCurNs();
#endif
  rr_send_request(ctx, rr_item, rr_pool);
#ifdef PROFILE_MODE
  SendDelay.push_back(getCurNs() - s_startNs);
#endif
}

// Loop while listening for completed responses.
void P2pRpcAppClient::CompleteReqListener() 
{
  struct p2p_rpc_rr *new_rr;
  uint64_t complendns;
  spdlog::info("P2pRpcAppClient Starting listener on core: {}, socket: {}", rte_lcore_id(), rte_socket_id()); 
  /*
   * Once we get a packet
   * 1. Check if it is a valid token
   * 2. Check if it is in valid state
   * 3. (optional) Check integrity of reply if verify_run is set
   * 4. Free the mbufs
   * 5. Reap the rr
   * 6. Trigger the next req if closedLoop
   */
  while(ACCESS_ONCE(exit_flag) == 0) 
  {
    TRACE_PRINTF("Waiting for resp, Recvd %u so far\n", n_r_recv.load());
    if(unlikely(rr_recv_response(ctx, &new_rr, rr_pool, exit_flag) == 0))
      continue;
#ifdef PROFILE_MODE
  r_startNs = getCurNs();
#endif

    TRACE_PRINTF("Got resp: %p, Token: %ld, Size: %ld, time: %ld\n", 
        (void*)new_rr, new_rr->req_token, new_rr->resp_size, complendns);

    void *got_req_token = (void *)new_rr->req_token;
    CallToken *call = NULL;

    if (isValidToken(got_req_token)) {
      call = reinterpret_cast<CallToken *>(got_req_token);
      if (call->call_state != R_STATES::R_SENT) {
          TRACE_PRINTF("Resp: %p, token: %ld is in invalid state: %d\n", (void*)new_rr, new_rr->req_token, call->call_state);
          spdlog::error("CompleteReqListener Request incorrect state, call-token: {}, State: {}",
                        (void *)call, call->call_state);
      } else {
        call->completionNs = getCurNs();
#ifdef PROFILE_MODE
  RecvDelay.push_back(getCurNs() - r_startNs);
#endif
        call->call_state = R_STATES::R_RECV;
        ++n_r_recv;
        if(verify_run) {
          if (unlikely(rr_bufs_to_resp(ctx, new_rr) != new_rr->resp_size))
            spdlog::error("Request {} gathered incorrectly", (void *)call);
          resp_cb((void*)(new_rr->resp_payload), new_rr->resp_size);
        }
        rr_release_mbufs(ctx, new_rr);
        new_rr->resp_size = 0;
        rr_pool.reap_rr(new_rr);

        if (closedLoop)
          SchedRequest(NULL);
      }
    } else {
      TRACE_PRINTF("Resp: %p is invalid token: %ld\n", (void*)new_rr, new_rr->req_token);
        spdlog::error("Non-request, call-token: {}", got_req_token);
        rr_release_mbufs(ctx, new_rr);
    }

    if (n_r_recv >= maxRequests)
      break;
  }
  StatsCollector->endExp();
  exit_flag = true;
  printf("P2pRpcAppClient listener terminating, Sent: %d, Recvd: %d\n", n_r_sent.load(), n_r_recv.load());
  spdlog::info("P2pRPCAppClient Listener terminating {} calls received...", n_r_recv.load());
}

void P2pRpcAppClient::PreProcessRequests() {
  spdlog::info("P2pRpcAppClient: Creating RR pool, with data on host");
  rr_pool.setup_and_init_rr_pool(conn_info, get_req_size(), get_resp_size(), -1);

  struct p2p_rpc_rr **all_rrs = rr_pool.get_rr_pool();
  for(int i = 0 ; i < rr_pool.get_pool_size(); i++) {
    // Setup the request payload and callback to get payload
    if (req_cb((void *)all_rrs[i]->req_payload, all_rrs[i]->req_size) != 0) {
      spdlog::error("P2pRpcAppClient Creating AppSpecific payload FAILED!!!");
    }
  }
}

void P2pRpcAppClient::PostProcessResponses() {
  spdlog::info("P2pRpcAppClient Postprocessing responses");
  if(verify_run) {
    // Validate the responses received for all rrs
    struct p2p_rpc_rr **all_rrs = rr_pool.get_rr_pool();
    int rr_errs = 0;
    for(int i = 0 ; i < rr_pool.get_pool_size(); i++) {
      if(resp_cb((void*)(all_rrs[i]->resp_payload), all_rrs[i]->max_resp_size) != 0) {
        rr_errs++;
      }
    }
    spdlog::error("P2pRpcAppClient RRs, Total: {},  Errors: {}", rr_pool.get_pool_size(), rr_errs);
  } else {
    spdlog::info("P2pRpcAppClient verify_run not set, RRs, Total: {}", rr_pool.get_pool_size());
  }

  // Call the base-class
  BaseClient::PostProcessResponses();
  
  PROF_PRINT("P2pRpc-SEND", SendDelay);
  PROF_PRINT("p2pRpc-RECV", RecvDelay);
}
