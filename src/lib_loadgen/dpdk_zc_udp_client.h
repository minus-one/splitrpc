// Copyright (c) Adithya Kumar, The Pennsylvania State University. All rights reserved.
// Licensed under the MIT License.

#ifndef DPDK_ZC_UDP_CLIENT_H
#define DPDK_ZC_UDP_CLIENT_H

#include "base_client.h"
#include <bits/stdc++.h>

#include "p2p_rpc.h"
#include "p2p_rpc_conn_info.h"
#include "transport/dpdk_utils.h"
#include "transport/dpdk_init.h"
#include "transport/dpdk_transport.h"
#include "p2p_rpc_rr.h"

#ifdef PROFILE_MODE
#include <nvToolsExt.h>
#endif

//const uint16_t RECV_PORT = 50052u;

// A DPDK based UDP client for all applications
/**
 * The DPDK client is a simple wrapper that gets raw bytes that need to be sent
 * from the application, slaps certain header information (to identify the
 * request) along with some timestamps, crafts it as a UDP packet and TXs it.
 * The receiver RXs incoming packet, verifies it is on a predefined port, and matches the
 * request based on the identifier that was sent and calculates latency
 * statistics.
 */
class DpdkZcUdpClient : public BaseClient {
  private:
    struct dpdk_ctx *ctx;
    struct p2p_rpc_conn_info *conn_info;

    std::string src_mac_str, dst_mac_str,
      src_ip_str, dst_ip_str,
      src_port_str, dst_port_str;
    int src_port, dst_port;
    mac_addr src_mac, dst_mac;

  public:
    explicit 
      DpdkZcUdpClient(std::string m_uri, StatsManager* stats) 
      : BaseClient(stats) 
      {
        std::stringstream ss_tmp(m_uri);
        std::getline(ss_tmp, src_mac_str, ',');
        std::getline(ss_tmp, src_ip_str, ',');
        std::getline(ss_tmp, src_port_str, ',');
        std::getline(ss_tmp, dst_mac_str, ',');
        std::getline(ss_tmp, dst_ip_str, ',');
        std::getline(ss_tmp, dst_port_str, ',');

        ctx = new struct dpdk_ctx;
        if(init_dpdk_ctx(ctx, get_dpdk_port(), 0) == 0) {
          spdlog::error("Cannot setup DPDK port");
          return;
        }

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
      }

    ~DpdkZcUdpClient()
    {
      //conn_info_teardown
      release_conn_info(conn_info);
      //dpdk_teardown
      stop_dpdk(ctx);
      delete ctx;
    }

    virtual void* CreateRequest(void *RequestParams);
    virtual void SchedRequest(void *reqTag);

    // Loop while listening for completed responses.
    // Prints out the response from the server.
    virtual void CompleteReqListener();

    void PreProcessRequests() override ;
    void PostProcessResponses() override;

};

// Threadsafe
// If request params already contains a call_id re-use that
// instead of recreating new requests
void* DpdkZcUdpClient::CreateRequest(void *RequestParams)
{
  CallToken *call = NULL;
  p2p_rpc_rr *call_id = NULL;
  if(RequestParams == NULL) {
    // Create a new call with new request
    uint32_t my_req_idx = ++n_r_create;
    call = SmRequests[my_req_idx];

    call_id = rr_alloc(conn_info, (uint64_t)(uintptr_t)call, max_req_size, max_resp_size);

    // Setup the request payload and callback to get payload
    if(req_cb((void*)call_id->req_payload, call_id->req_size) != 0) {
      printf("Creating AppSpecific payload FAILED!!!\n");
      return NULL;
    } 
    call->call_id = call_id;
    call->size = call_id->req_size;
  } else {
    // Recyle a call and save stats 
    call = (CallToken*)RequestParams;
    if(StatsCollector && (call->warmup != true)) {
      StatsCollector->recordEvent("sojournTime",
            (call->completionNs - call->dispatchNs));
    }
    call_id = (struct p2p_rpc_rr*)call->call_id;
    call_id->resp_size = 0;
  }
  call->call_state = R_STATES::R_CREATE;
  return static_cast<void*>(call);
}

// TX the request
// Schedules the next request based on index
// (or) schedules a specific request
void DpdkZcUdpClient::SchedRequest(void *reqTag) 
{
  uint32_t my_req_idx = ++n_r_sent;
  if(my_req_idx > maxRequests)
    return;
#ifdef PROFILE_MODE
  nvtxRangePush("dpdk-sched-req");
#endif
  CallToken *call = NULL;
  if(reqTag == NULL) {
    call = SmRequests[my_req_idx];
  } else {
    call = static_cast<CallToken*>(reqTag);
  }

  if(my_req_idx <= n_r_warmup)
    call->warmup = true;
  call->call_state = R_STATES::R_SENT;

  /*
   * Steps to send a request
   * 1. Allocate the necessary transport mbufs
   * 2. Set the header for each mbuf
   * 3. Copy the request to the mbuf (scatter)
   * 4. Set dispatch time-stamp
   * 5. Transmit the mbufs
   */
  p2p_rpc_rr *call_id = (struct p2p_rpc_rr*)(call->call_id);
  if(rr_alloc_mbufs(ctx, call_id, call_id->req_size) == 0) {
    spdlog::error("Error allocating request {}", (void*)call);
  }
  if(rr_set_hdr(call_id, call_id->req_size) == 0) {
    spdlog::error("Error sending request {}", (void*) call);
  }
  if(rr_req_to_bufs(ctx, call_id) != call_id->req_size) {
    spdlog::error("Error in copying payload {}", (void*) call);
  }
  call->dispatchNs = getCurNs();
  if(send_requests_zc(ctx, call_id->transport_mbufs) != call_id->hdr_bufs->num_items) {
    spdlog::error("TX failed for {}", (void*)call);
  }
#ifdef PROFILE_MODE
  nvtxRangePop();
#endif
}

// Loop while listening for completed responses.
void DpdkZcUdpClient::CompleteReqListener() 
{
  struct p2p_hbufs *dpdk_mbufs = new p2p_hbufs;
  struct p2p_hbufs *hdr_bufs = new p2p_hbufs;
  struct p2p_bufs *payload_bufs = new p2p_bufs;
#ifdef PROFILE_MODE
  uint64_t startNs, endNs;
#endif
  spdlog::info("Starting DPDK listener on core: {}, socket: {}", rte_lcore_id(), rte_socket_id()); 
  while(1) {
#ifdef PROFILE_MODE
    //nvtxRangePush("dpdk-udp-recv");
    //nvtxMark("dpdk-udp-recv");
    startNs = getCurNs();
#endif
    // Wait for RX
    while(get_requests_zc(ctx, dpdk_mbufs, hdr_bufs, payload_bufs) == 0);
    uint64_t complendns = getCurNs();

    for(int i = 0 ; i < dpdk_mbufs->num_items; i++) {
      struct p2p_rpc_hdr *rpc_hdr = 
        (struct p2p_rpc_hdr*)(hdr_bufs->burst_items[i]);
      void *got_tag = (void*)get_req_token(rpc_hdr);
      CallToken *call = NULL;

      if(isValidToken(got_tag)) {
        call = reinterpret_cast<CallToken*>(got_tag);
        if(call->call_state != R_STATES::R_SENT) {
          spdlog::error("Request incorrect state, call-token: {}, State: {}",
              (void*)call, call->call_state);
        } else {
          struct p2p_rpc_rr *call_id = (struct p2p_rpc_rr*)call->call_id;
          rr_append_resp_mbuf(call_id, 
              dpdk_mbufs->burst_items[i], hdr_bufs->burst_items[i], 
              payload_bufs->burst_items[i], payload_bufs->item_size[i]);
          //printf("Call: %p, dpdk_bufs: %d, Num payload bufs: %d\n",
          //    call_id, call_id->transport_mbufs->num_items, call_id->payload_bufs->num_items);
          // Check if the call is complete
          if(call_id->resp_size == call_id->max_resp_size) {
            call->completionNs = complendns;
            call->call_state = R_STATES::R_RECV;
            if(rr_bufs_to_resp(ctx, call_id) == call_id->resp_size) {
              rr_release_mbufs(ctx, call_id);
              ++n_r_recv;
              if(closedLoop) {
                SchedRequest(NULL);
              }
            } else {
              spdlog::error("Request {} gathered incorrectly", (void*)call);
            }
          } else {
            //spdlog::info("Waiting to receive more packets for call: {}, got {} out of {}", 
            //    (void*)call, call_id->resp_size, call_id->max_resp_size);
          }
        }
      } else {
        spdlog::error("Non-request, call-token: {}", got_tag);
        rte_pktmbuf_free((struct rte_mbuf*)(dpdk_mbufs->burst_items[i])); 
      }
    }
    if(complendns > endTimeNs)
      endTimeNs = complendns;
    if(n_r_recv >= maxRequests)
      break;
#ifdef PROFILE_MODE
    //nvtxMark("dpdk-udp-recv-end");
    //nvtxRangePop();
    endNs = getCurNs();
    printf("EchoResponse Lat: %ldns, P.T: %ldns\n", endNs - startNs, endNs - complendns);
#endif
  }
  spdlog::info("DPDK Listener terminating..."); 
}

void DpdkZcUdpClient::PreProcessRequests() {
  spdlog::info("DpdkPreprocessing Requests and setting up mbufs");
  for(int i = 0; i < maxRequests; i++) {
    CreateRequest(NULL);
  }
}

void DpdkZcUdpClient::PostProcessResponses() {
  // FIXME: Assumes only a single packet response
  for(int i = 0 ; i < maxRequests; i++) {
    CallToken *call = SmRequests[i];
    if(call == NULL)
      continue;
    if(call->call_state == R_STATES::R_RECV) {
      struct p2p_rpc_rr *call_id = (p2p_rpc_rr*)call->call_id;
      if(resp_cb((void*)(call_id->resp_payload), call_id->resp_size) != 0) {
        spdlog::error("Response error for req, call-token: {}", (void*)call);
      } else if(StatsCollector && (call->warmup != true)) {
        StatsCollector->recordEvent("sojournTime",
            (call->completionNs - call->dispatchNs));
      }
    } else {
      spdlog::error("Req: {} in incorrect state: {}", (void*)call, call->call_state);
    }
  }
}

#endif /* DPDK_ZC_UDP_CLIENT_H */
