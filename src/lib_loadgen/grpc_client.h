// Copyright (c) Adithya Kumar, The Pennsylvania State University. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <vector>
#include <fstream>
#include <iostream>

#include "grpc_common.h"
#include "time_utils.h"
#include "config_utils.h"
#include "stats_utils.h"

// A common GRPC client for all applications
class GrpcClient : public BaseClient {
  private:
    // struct for keeping state and data information
    struct AsyncClientCall {
      // Container for the data we expect from the server.
      ScheduleReply reply;
      ScheduleRequest request;

      // Context for the client. It could be used to convey extra information to
      // the server and/or tweak certain RPC behaviors.
      ClientContext context;

      // Storage for the status of the RPC upon completion.
      Status status;

      std::unique_ptr<ClientAsyncResponseReader<ScheduleReply>> response_reader;
    };

    // Stub comes out of the passed in Channel giving a view of the server's exposed services.
    std::unique_ptr<Scheduler::Stub> stub_;
    std::string ServerSockString_;

    // The producer-consumer queue to communicate asynchronously with the gRPC runtime.
    CompletionQueue cq_;

    char *i_buf, *o_buf;

  public:
#ifdef PROFILE_MODE
  uint64_t startNs, endNs;
  int numMetrics;
  std::vector<uint64_t> SendDelay, RecvDelay;
#endif
    explicit GrpcClient(std::string m_uri, StatsManager *stats)
    : BaseClient(stats) {

      std::string src_mac_str, dst_mac_str, src_ip_str, dst_ip_str, src_port_str, dst_port_str;

      std::stringstream ss_tmp(m_uri);
      std::getline(ss_tmp, src_mac_str, ',');
      std::getline(ss_tmp, src_ip_str, ',');
      std::getline(ss_tmp, src_port_str, ',');
      std::getline(ss_tmp, dst_mac_str, ',');
      std::getline(ss_tmp, dst_ip_str, ',');
      std::getline(ss_tmp, dst_port_str, ',');


      ServerSockString_ = dst_ip_str + std::string(":") + dst_port_str;
      spdlog::info("GrpcClient init: server-uri: {}", ServerSockString_);
      grpc::ChannelArguments channel_args;
      channel_args.SetMaxReceiveMessageSize(MAX_MSG_SIZE);
      channel_args.SetMaxSendMessageSize(MAX_MSG_SIZE);

      std::shared_ptr<grpc::Channel> my_channel = 
        grpc::CreateCustomChannel(ServerSockString_, grpc::InsecureChannelCredentials(), channel_args);
      stub_ = Scheduler::NewStub(my_channel);
      // Shared buf between all requests
      i_buf = new char[max_req_size];
      o_buf = new char[max_resp_size];
      std::memset(i_buf, 1, max_req_size);
      std::memset(o_buf, 0, max_resp_size);
    }

    virtual ~GrpcClient()
    {
      PROF_PRINT("GRPC-Send", SendDelay);
      PROF_PRINT("GPRC-Recv", RecvDelay);
#ifdef PROFILE_MODE
      printPbStats();
#endif
      delete i_buf;
      delete o_buf;
    }

    void PrintOutputBuf() {
      std::cout<<"Output len: "<<max_resp_size<<"\n";
      for(int i = 0 ; i < 5 ; i++) {
        printf("Int: o_buf[%d] = %d\n", i, ((uint8_t*)o_buf)[i]);
      }
      printf("Int: o_buf[-2] = %d\n", ((uint8_t*)o_buf)[max_resp_size-2]);
      printf("Int: o_buf[-1] = %d\n", ((uint8_t*)o_buf)[max_resp_size-1]);
      for(int i = 0 ; i < 5 ; i++) {
        printf("Float: o_buf[%d] = %f\n", i, ((float*)o_buf)[i]);
      }
    }

    virtual std::string getName() {
      return std::string("GrpcClient");
    }

    virtual void* CreateRequest(void *RequestParams);
    virtual void SchedRequest(void *reqTag);

    // Loop while listening for completed responses.
    // Prints out the response from the server.
    virtual void CompleteReqListener();

    void PreProcessRequests() override ;
    void PostProcessResponses() override;
};


void* GrpcClient::CreateRequest(void *RequestParams)
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

void GrpcClient::SchedRequest(void *reqTag)
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

  AsyncClientCall* rpc_item = new AsyncClientCall();
  assert(rpc_item != nullptr);
  rpc_item->request.set_guid(my_req_idx);
  if(verify_run) {
    if (req_cb((void *)i_buf, max_req_size) != 0) {
      spdlog::error("GrpcClient Creating AppSpecific payload FAILED!!!");
    }
  }
  rpc_item->request.set_payload(copyBufToString(i_buf, max_req_size));
  call->dispatchNs = getCurNs();
#ifdef PROFILE_MODE
  startNs = getCurNs();
#endif
  rpc_item->response_reader = stub_->AsyncrunJob(&rpc_item->context, rpc_item->request, &cq_);
#ifdef PROFILE_MODE
  SendDelay.push_back(getCurNs() - startNs);
#endif
  rpc_item->response_reader->Finish(&rpc_item->reply, &rpc_item->status, (void*)rpc_item);
}

void GrpcClient::CompleteReqListener()
{
  void* got_tag;
  bool ok = false;
  int ret = 1;
  while(ACCESS_ONCE(exit_flag) == 0) 
  {
    ret = 1;
    ok = false;
    cq_.Next(&got_tag, &ok);
#ifdef PROFILE_MODE
    startNs = getCurNs();
#endif
    AsyncClientCall* rpc_item = static_cast<AsyncClientCall*>(got_tag);
    assert(ok);
    assert(rpc_item->status.ok());

    if(rpc_item->request.guid() != rpc_item->reply.guid())
      ret = 0;
    if(copyStringToBuf(rpc_item->reply.payload(), o_buf) != max_resp_size)
      ret = 0;
    if(ret == 1) {
      CallToken *call = SmRequests[rpc_item->request.guid()];
      call->completionNs = getCurNs();
#ifdef PROFILE_MODE
    RecvDelay.push_back(getCurNs() - startNs);
#endif
      call->call_state = R_STATES::R_RECV;
      if(verify_run) {
        resp_cb((void*)o_buf, max_resp_size);
      }
    } 
    ++n_r_recv;
    delete rpc_item;
    if (n_r_recv >= maxRequests)
      break;
    if (closedLoop)
      SchedRequest(NULL);
  }
  StatsCollector->endExp();
  exit_flag = true;
  printf("GrpcClient listener terminating, Sent: %d, Recvd: %d\n", 
      n_r_sent.load(), n_r_recv.load());
  spdlog::info("GrpcClient Listener terminating {} calls received...", 
      n_r_recv.load());
  PrintOutputBuf();
}

void GrpcClient::PreProcessRequests() {
  spdlog::info("GrpcClient: PreProcessRequests");
}

void GrpcClient::PostProcessResponses() {
  spdlog::info("GrpcClient Postprocessing responses");
  BaseClient::PostProcessResponses();
}

//int main()
//{
//  std::string m_uri = get_server_ip() + std::string(":") + get_server_port();
//  size_t max_req_size = get_req_size();
//  size_t max_resp_size = get_resp_size();
//  std::cout<<"InputLen: "<<max_req_size<<"OutputLen: "<< max_resp_size << "\n";
//
//  GrpcClient client(m_uri, max_req_size, max_resp_size);
//
//  printf("Warming up...\n");
//  for(int i = 0 ; i < 10 ; i++) {
//    client.CreateAndSendRequest();
//    if(client.GetResponse() == 0) {
//      printf("Invalid response, warm up fail!\n");
//    }
//  }
//  client.PrintOutputBuf();
//
//  uint64_t startNs, endNs;
//  int numMetrics = 20000;
//  std::vector<uint64_t> metricValues(numMetrics, 0);
//  int metricNum = 0;
//  int failedRuns = 0;
//
//  printf("Starting work...\n");
//  for(int i = 0; i < numMetrics; i++) {
//    startNs = getCurNs();
//
//    client.CreateAndSendRequest();
//    if(client.GetResponse() == 1) {
//      endNs = getCurNs();
//      metricNum = (metricNum + 1) % numMetrics;
//      metricValues[metricNum] = endNs - startNs;
//    } else {
//      failedRuns++;
//      i--;
//    }
//  }
//
//  printf("Total failed runs: %d\n", failedRuns);
//  printf("Exec time stats(ns) [N, Mean, p90, p95, p99]: %d, %0.2f, %0.2f, %0.2f, %0.2f\n", 
//      numMetrics, getMean(metricValues), getPercentile(metricValues, 0.90), 
//      getPercentile(metricValues, 0.95), getPercentile(metricValues, 0.99)); 
//
//  
//#ifdef PROFILE_MODE
//  printPbStats();
//#endif
//
//  return 0;
//}
