// Copyright (c) Adithya Kumar, The Pennsylvania State University. All rights reserved.
// Licensed under the MIT License.

#ifndef BASE_UDP_CLIENT_H
#define BASE_UDP_CLIENT_H

#include "base_client.h"
#include <bits/stdc++.h>
#include <arpa/inet.h>
#include <sys/socket.h>

#ifdef PROFILE_MODE
#include <nvToolsExt.h>
#endif

const uint16_t RECV_PORT = 50052u;

// A common linux based UDP client for all applications
/**
 * A UDP client is a simple wrapper that gets raw bytes that need to be sent
 * from the application, slaps certain header information (to identify the
 * request) along with some timestamps, and send it across the wire to the 
 * (IP,Port).
 * The receiver listens to response on a predefined port, and matches the
 * request based on the identifier that was sent and calculates latency
 * statistics.
 */
class BaseUdpClient : public BaseClient {
  private:
    // End-points of UDP server
    std::string uriIp, uriPort;

    // Connection information
    int send_sockfd, recv_sockfd;
    struct sockaddr_in send_servaddr, recv_servaddr, si_other;
    socklen_t recv_sa_len;

  public:
    explicit 
      BaseUdpClient(std::string m_uri, StatsManager* stats) 
      : BaseClient(stats) {
        // Create a RX, TX pair of communication channels 
        std::stringstream ss_tmp(m_uri);
        std::getline(ss_tmp, uriIp, ':');
        std::getline(ss_tmp, uriPort, ':');
        // Creating socket file descriptors
        if ( (send_sockfd = socket(AF_INET, SOCK_DGRAM, 0)) < 0 ) {
          spdlog::error("socket creation failed");
          exit(EXIT_FAILURE);
        }
	
        ///* Open RAW socket to send on */
        //if ((sockfd = socket(AF_PACKET, SOCK_RAW, IPPROTO_RAW)) == -1) {
        //  perror("socket");
        //}

        if((recv_sockfd = socket(AF_INET, SOCK_DGRAM, 0)) < 0) {
          spdlog::error("Socket creation failed");
          exit(EXIT_FAILURE);
        }
        memset(&send_servaddr, 0, sizeof(send_servaddr));
        memset(&recv_servaddr, 0, sizeof(recv_servaddr));
        memset(&si_other, 0, sizeof(si_other));

        // Filling server information
        send_servaddr.sin_family = AF_INET;
        send_servaddr.sin_port = htons(static_cast<uint16_t>(std::stoul(uriPort)));
        if (inet_aton(uriIp.c_str(), &send_servaddr.sin_addr) == 0) {
          spdlog::error("inet_aton() failed");
          exit(1);
        }
        recv_servaddr.sin_family = AF_INET;
        recv_servaddr.sin_port = htons(RECV_PORT);
        recv_servaddr.sin_addr.s_addr = htonl(INADDR_ANY);
        if( bind(recv_sockfd , (struct sockaddr*)&recv_servaddr, sizeof(recv_servaddr) ) == -2) {
          spdlog::error("Bind failed");
	      }
        int disable = 1;
        if(setsockopt(recv_sockfd, SOL_SOCKET, SO_NO_CHECK, (void*)&disable, sizeof(disable)) != 0) {
          spdlog::error("Error setting sock options");
        }
        struct timeval tv;
        tv.tv_sec = 0;
        tv.tv_usec = 100000;
        if (setsockopt(recv_sockfd, SOL_SOCKET, SO_RCVTIMEO,&tv,sizeof(tv)) < 0) {
              perror("Error");
        }

        si_other.sin_family = AF_INET;
        si_other.sin_addr.s_addr = htonl(INADDR_ANY);
        recv_sa_len = sizeof(si_other);
      }

    ~BaseUdpClient()
    {
      close(send_sockfd);
      close(recv_sockfd);
    }

    virtual void* CreateRequest(void *RequestParams);
    virtual void SchedRequest(void *reqTag);
    //virtual void CompleteReply(void *reqTag);

    // Loop while listening for completed responses.
    // Prints out the response from the server.
    virtual void CompleteReqListener();
    virtual std::string getName() {
      return std::string("UdpClient");
    }
};

// Threadsafe
void* BaseUdpClient::CreateRequest(void *RequestParams)
{
  CallToken *call = new CallToken();
  assert(call != nullptr);

  call->req_buf = new (std::nothrow) uint8_t[max_req_size];
  call->resp_buf = new (std::nothrow) uint8_t[max_req_size]; 
  
  // Store token inside request
  assert(call->req_buf != NULL);
  memcpy(call->req_buf, &call, sizeof(call));

  // Construct the request - callback to get payload
  if(req_cb(call->req_buf + sizeof(call), call->size) != 0) {
    spdlog::error("Creating AppSpecific payload FAILED!!!");
    return NULL;
  }
  
  call->size += sizeof(call);
  AllRequests[call] = R_STATES::R_CREATE;
  return static_cast<void*>(call);
}

// TX the request
void BaseUdpClient::SchedRequest(void *reqTag) 
{
#ifdef PROFILE_MODE
  nvtxRangePush("udp-sched-req");
#endif
  if(++n_r_sent > maxRequests)
    return;

  CallToken* call = static_cast<CallToken*>(reqTag);
  if(n_r_sent <= n_r_warmup)
    call->warmup = true;
  AllRequests[call] = R_STATES::R_SENT;

  call->dispatchNs = getCurNs();

  if(sendto(send_sockfd, (const char *)(call->req_buf), call->size,
        MSG_CONFIRM, (const struct sockaddr *)&send_servaddr,
        sizeof(send_servaddr)) == -1) {
    spdlog::error("Sendto failed, SchedRequest failed");
    exit(1);           
  }
  delete[] call->req_buf;
#ifdef PROFILE_MODE
  nvtxRangePop();
#endif
}

// Loop while listening for completed responses.
void BaseUdpClient::CompleteReqListener() 
{
  void *got_tag;
  void *next_req;
  if(closedLoop) {
    next_req = CreateRequest(NULL);
  }
  uint8_t *recvBuf = new (std::nothrow) uint8_t[max_req_size];
  assert(recvBuf != NULL);

  spdlog::info("Starting listener"); 
  while(ACCESS_ONCE(exit_flag) == 0) 
  {
#ifdef PROFILE_MODE
    nvtxRangePush("udp-recv");
#endif
    if(recvfrom(recv_sockfd, recvBuf, max_req_size, 0,\
          (struct sockaddr *) &si_other, &recv_sa_len) == -1) {
      //spdlog::error("recvfrom failed");
      if(n_r_recv >= maxRequests)
        break;
      continue;
    }
    uint64_t complendns = getCurNs();
#ifdef PROFILE_MODE
    nvtxRangePop();
#endif
    if(complendns > endTimeNs)
      endTimeNs = complendns;

    // Assume the first set of bytes contain the req-token
    uintptr_t tmp = *(reinterpret_cast<uintptr_t*>(recvBuf));
    got_tag = reinterpret_cast<void*>(tmp);

    CallToken* call = static_cast<CallToken*>(got_tag);
    if(AllRequests.find(call) == AllRequests.end()) {
      spdlog::error("Recvd a response for a non-request, call-token: {}", got_tag);
    } else if(AllRequests[call] != R_STATES::R_SENT) {
      spdlog::error("Recvd a response for req in incorrect state, call-token: {}, State: {}", got_tag, AllRequests[call]);
      //continue;
    } else {
      call->completionNs = complendns;
      AllRequests[call] = R_STATES::R_RECV;
      memcpy(call->resp_buf, recvBuf + sizeof(call), max_req_size - sizeof(call));
    }

    if(++n_r_recv >= maxRequests)
      break;
    // If this is a closed loop client, requests are sent once again
    if(closedLoop) {
      SchedRequest(next_req);
      next_req = CreateRequest(NULL);
    }
  }
  spdlog::info("Listener completed..."); 
  delete[] recvBuf;
}

#endif /* BASE_UDP_CLIENT_H */
