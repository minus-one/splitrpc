// Copyright (c) Adithya Kumar, The Pennsylvania State University. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "base_client.h"
#include <bits/stdc++.h>

#include "transport/udp_transport.h"

// A UDP-RPC Client for all applications
/**
 * The RPC client is a wrapper that gets raw bytes that need to be sent
 * from the application, slaps certain header information (to identify the
 * request) along with some timestamps, uses sendto to send it as a UDP packet.
 * The receiver RXs incoming packet, verifies it is on a predefined port, and matches the
 * request based on the identifier that was sent and calculates latency
 * statistics.
 */
class UdpAppClient : public BaseClient {
  private:
    //struct dpdk_ctx *ctx;
    //struct p2p_rpc_conn_info *conn_info;

    std::string src_mac_str, dst_mac_str,
      src_ip_str, dst_ip_str,
      src_port_str, dst_port_str;
    int src_port, dst_port;
    mac_addr src_mac, dst_mac;

    struct sockaddr_in si_me, si_server;
    int udp_sock;

  UdpRrPool rr_pool;
#ifdef PROFILE_MODE
  uint64_t s_startNs, r_startNs;
  int numMetrics;
  std::vector<uint64_t> SendDelay, RecvDelay;
#endif

  public:
    explicit 
      UdpAppClient(std::string m_uri, StatsManager* stats) 
      : BaseClient(stats) 
      {
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

        memset((char *)&si_me, 0, sizeof(si_me));
        si_me.sin_family = AF_INET;
        si_me.sin_port = htons(src_port);
        if (inet_aton(src_ip_str.c_str(), &si_me.sin_addr) == 0) {
          std::cout<<"inet_aton() failed to parse src_ip_str\n";
          exit(1);
        }

        memset((char *)&si_server, 0, sizeof(si_server));
        si_server.sin_family = AF_INET;
        si_server.sin_port = htons(dst_port);
        if (inet_aton(dst_ip_str.c_str(), &si_server.sin_addr) == 0) {
          std::cout<<"inet_aton() failed to parse dst_ip_str\n";
          exit(1);
        }

        udp_sock = initUdpSock(&si_me, src_port);
        spdlog::info("Initialized client on sock: {}", udp_sock);
        if(udp_sock == -1) {
          spdlog::error("Socket error: {}", udp_sock);
          exit(1);
        }
      }

    ~UdpAppClient() { }

    virtual std::string getName() {
      return std::string("UdpRpcClient");
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
void* UdpAppClient::CreateRequest(void *RequestParams)
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
void UdpAppClient::SchedRequest(void *reqTag) 
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

  TRACE_PRINTF("Call: %p, Warmup: %d, State: %d\n", 
      (void*)call, call->warmup, call->call_state);

  /*
   * Steps to send a request
   * 1. Use sendto and send chunks of requests
   * 2. Set dispatch time-stamp
   * 3. Transmit the bufs
   * 4. Reap the rr
   */
  UdpRr *rr_item = rr_pool.consume_rr(); 
  TRACE_PRINTF("Call: %p, rr_item: %p\n", (void*)call, (void*)rr_item);
  rr_item->req_token = (uint64_t)(void*)call;
  //rr_item->req_token = (uint64_t)((uintptr_t)call - token_start_addr);

  rr_item->si_other = si_server;
  // Instead of loading data everytime, just set the size - the payload wouldn't have gone away
  rr_item->req_size = max_req_size;

  if(rr_item->req_to_bufs() != rr_item->req_size) {
    printf("Error in copying payload for call: %p\n", (void*)call);
    exit(1);
  }
  call->dispatchNs = getCurNs();
#ifdef PROFILE_MODE
  s_startNs = getCurNs();
#endif
  udp_rr_send_req(udp_sock, rr_item);
#ifdef PROFILE_MODE
  SendDelay.push_back(getCurNs() - s_startNs);
#endif
  rr_pool.reap_rr(rr_item);
}

// Loop while listening for completed responses.
void UdpAppClient::CompleteReqListener() 
{
  UdpRr *new_rr = NULL;
  uint64_t complendns;
  spdlog::info("UdpAppClient Starting listener for port: {}", src_port); 
  /*
   * Once we get a packet
   * 1. Check if it is a valid token
   * 2. Check if it is in valid state
   * 3. (optional) Check integrity of reply if verify_run is set
   * 4. Reap the rr
   * 5. Trigger the next req if closedLoop
   */
  while(ACCESS_ONCE(exit_flag) == 0) 
  {
    TRACE_PRINTF("Waiting for resp, Recvd %u so far\n", n_r_recv.load());
    if(unlikely(udp_rr_recv_resp(udp_sock, &new_rr, &rr_pool) == 0))
      continue;

#ifdef PROFILE_MODE
  r_startNs = getCurNs();
#endif

    TRACE_PRINTF("Got resp: %p, Token: %ld, Size: %ld, time: %ld\n", 
        (void*)new_rr, new_rr->req_token, new_rr->resp_size, complendns);

    //void *got_req_token = (void *)((uintptr_t)new_rr->req_token + token_start_addr);
    void *got_req_token = (void *)(new_rr->req_token); 
    CallToken *call = NULL;

    if (isValidToken(got_req_token)) {
      call = reinterpret_cast<CallToken *>(got_req_token);
      if (call->call_state != R_STATES::R_SENT) {
          TRACE_PRINTF("Resp: %p, token: %ld is in invalid state: %d\n", (void*)new_rr, new_rr->req_token, call->call_state);
          spdlog::error("CompleteReqListener Request incorrect state, call: {}, State: {}",
                        (void *)call, call->call_state);
      } else {
        call->completionNs = getCurNs();
#ifdef PROFILE_MODE
  RecvDelay.push_back(getCurNs() - r_startNs);
#endif
        call->call_state = R_STATES::R_RECV;
        ++n_r_recv;
        if(verify_run) {
          if(unlikely(new_rr->bufs_to_resp() != new_rr->resp_size))
            spdlog::error("Request {} gathered incorrectly", (void *)call);
          resp_cb((void*)(new_rr->resp_payload), new_rr->resp_size);
        }
        new_rr->release_resp_bufs();
        rr_pool.reap_rr(new_rr);
        if (closedLoop)
          SchedRequest(NULL);
      }
    } else {
      TRACE_PRINTF("Resp: %p is invalid token: %ld\n", (void*)new_rr, new_rr->req_token);
        spdlog::error("Non-request, call-token: {}", got_req_token);
        new_rr->release_resp_bufs();
        rr_pool.reap_rr(new_rr);
    }

    if (n_r_recv >= maxRequests)
      break;
  }
  StatsCollector->endExp();
  exit_flag = true;
  printf("UdpAppClient listener terminating, Sent: %d, Recvd: %d\n", n_r_sent.load(), n_r_recv.load());
  spdlog::info("UdpAppClient Listener terminating {} calls received...", n_r_recv.load());
}

void UdpAppClient::PreProcessRequests() {
  size_t req_size = get_req_size();
  size_t resp_size = get_resp_size();
  spdlog::info("Creating UdpRrPool, with data on host, req_size: {}, resp_size: {}", req_size, resp_size);
  rr_pool.setup_and_init_rr_pool(req_size, resp_size);
  UdpRr **all_rrs = rr_pool.get_rr_pool();

  for(int i = 0 ; i < rr_pool.get_pool_size(); i++) {
    // Setup the request payload and callback to get payload
    if (req_cb((void *)all_rrs[i]->req_payload, all_rrs[i]->req_size) != 0) {
      spdlog::error("UdpAppClient Creating AppSpecific payload FAILED!!!");
    } else {
      all_rrs[i]->alloc_req_bufs();
    }
  }
}

void UdpAppClient::PostProcessResponses() {
  spdlog::info("UdpAppClient Postprocessing responses");
  if(verify_run) {
    // Validate the responses received for all rrs
    UdpRr **all_rrs = rr_pool.get_rr_pool();
    int rr_errs = 0;
    for(int i = 0 ; i < rr_pool.get_pool_size(); i++) {
      if(resp_cb((void*)(all_rrs[i]->resp_payload), all_rrs[i]->max_resp_size) != 0) {
        rr_errs++;
      }
    }
    spdlog::error("UdpAppClient RRs, Total: {},  Errors: {}", rr_pool.get_pool_size(), rr_errs);
  } else {
    spdlog::info("UdpAppClient verify_run not set, RRs, Total: {}", rr_pool.get_pool_size());
  }

  // Call the base-class
  BaseClient::PostProcessResponses();
  
  PROF_PRINT("UDP-Send", SendDelay);
  PROF_PRINT("UDP-Recv", RecvDelay);
}
