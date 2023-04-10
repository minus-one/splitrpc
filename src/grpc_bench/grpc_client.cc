// Copyright (c) Adithya Kumar, The Pennsylvania State University. All rights reserved.
// Licensed under the MIT License.

#include "grpc_common.h"

#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <vector>
#include <fstream>
#include <iostream>

#include "time_utils.h"
#include "config_utils.h"
#include "stats_utils.h"

// A common GRPC client for all applications
class GrpcClient {
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
    int64_t numReqCreated;

    size_t inputLen, outputLen;

  public:
#ifdef PROFILE_MODE
  uint64_t startNs, endNs;
  int numMetrics;
  std::vector<uint64_t> SendDelay, RecvDelay;
#endif
    explicit GrpcClient(std::string m_uri, size_t _inputLen, size_t _outputLen) {
      ServerSockString_ = m_uri;
      inputLen = _inputLen;
      outputLen = _outputLen;
      grpc::ChannelArguments channel_args;
      channel_args.SetMaxReceiveMessageSize(MAX_MSG_SIZE);
      channel_args.SetMaxSendMessageSize(MAX_MSG_SIZE);

      std::shared_ptr<grpc::Channel> my_channel = grpc::CreateCustomChannel(ServerSockString_, grpc::InsecureChannelCredentials(), channel_args);
      stub_ = Scheduler::NewStub(my_channel);
      i_buf = new char[inputLen];
      o_buf = new char[outputLen];
      std::memset(i_buf, 1, inputLen);
      std::memset(o_buf, 0, outputLen);
      numReqCreated = 0;
    }

    virtual ~GrpcClient()
    {
#ifdef PROFILE_MODE
      printf("SendDelay stats(ns) [N, Mean, p90, p95, p99]: %d, %0.2f, %0.2f, %0.2f, %0.2f\n", 
          SendDelay.size(), getMean(SendDelay), getPercentile(SendDelay, 0.90), 
          getPercentile(SendDelay, 0.95), getPercentile(SendDelay, 0.99)); 
      printf("RecvDelay stats(ns) [N, Mean, p90, p95, p99]: %d, %0.2f, %0.2f, %0.2f, %0.2f\n", 
          RecvDelay.size(), getMean(RecvDelay), getPercentile(RecvDelay, 0.90), 
          getPercentile(RecvDelay, 0.95), getPercentile(RecvDelay, 0.99)); 
#endif
      delete i_buf;
      delete o_buf;
    }

    void PrintOutputBuf() {
      std::cout<<"Output len: "<<outputLen<<"\n";
      for(int i = 0 ; i < 5 ; i++) {
        printf("Int: o_buf[%d] = %d\n", i, ((uint8_t*)o_buf)[i]);
      }
      for(int i = 0 ; i < 5 ; i++) {
        printf("Float: o_buf[%d] = %f\n", i, ((float*)o_buf)[i]);
      }
    }

    void CreateAndSendRequest();
    int GetResponse();
};

void GrpcClient::CreateAndSendRequest()
{
#ifdef PROFILE_MODE
  startNs = getCurNs();
#endif
  AsyncClientCall* call = new AsyncClientCall();
  assert(call != nullptr);
  call->request.set_guid(numReqCreated++);
  call->request.set_payload(copyBufToString(i_buf, inputLen));
#ifdef PROFILE_MODE
  SendDelay.push_back(getCurNs() - startNs);
#endif
  call->response_reader = stub_->AsyncrunJob(&call->context, call->request, &cq_);
  call->response_reader->Finish(&call->reply, &call->status, (void*)call);
}

int GrpcClient::GetResponse()
{
  void* got_tag;
  bool ok = false;
  int ret = 1;
  cq_.Next(&got_tag, &ok);
#ifdef PROFILE_MODE
  startNs = getCurNs();
#endif
  AsyncClientCall* call = static_cast<AsyncClientCall*>(got_tag);
  GPR_ASSERT(ok);
  assert(call->status.ok());

  if(call->request.guid() != call->reply.guid())
    ret = 0;
  if(copyStringToBuf(call->reply.payload(), o_buf) != outputLen)
    ret = 0;
#ifdef PROFILE_MODE
  RecvDelay.push_back(getCurNs() - startNs);
#endif
  delete call;
  return ret;
}

int main()
{
  std::string m_uri = get_server_ip() + std::string(":") + get_server_port();
  size_t inputLen = get_req_size();
  size_t outputLen = get_resp_size();
  std::cout<<"InputLen: "<<inputLen<<"OutputLen: "<< outputLen << "\n";

  GrpcClient client(m_uri, inputLen, outputLen);

  printf("Warming up...\n");
  for(int i = 0 ; i < 10 ; i++) {
    client.CreateAndSendRequest();
    if(client.GetResponse() == 0) {
      printf("Invalid response, warm up fail!\n");
    }
  }
  client.PrintOutputBuf();

  uint64_t startNs, endNs;
  int numMetrics = 20000;
  std::vector<uint64_t> metricValues(numMetrics, 0);
  int metricNum = 0;
  int failedRuns = 0;

  printf("Starting work...\n");
  for(int i = 0; i < numMetrics; i++) {
    startNs = getCurNs();

    client.CreateAndSendRequest();
    if(client.GetResponse() == 1) {
      endNs = getCurNs();
      metricNum = (metricNum + 1) % numMetrics;
      metricValues[metricNum] = endNs - startNs;
    } else {
      failedRuns++;
      i--;
    }
  }

  printf("Total failed runs: %d\n", failedRuns);
  printf("Exec time stats(ns) [N, Mean, p90, p95, p99]: %d, %0.2f, %0.2f, %0.2f, %0.2f\n", 
      numMetrics, getMean(metricValues), getPercentile(metricValues, 0.90), 
      getPercentile(metricValues, 0.95), getPercentile(metricValues, 0.99)); 

  
#ifdef PROFILE_MODE
  printPbStats();
#endif

  return 0;
}
