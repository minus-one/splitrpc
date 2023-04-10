// Copyright (c) Adithya Kumar, The Pennsylvania State University. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <grpc++/grpc++.h>
#include <infer.grpc.pb.h>
#include <grpc/support/log.h>

#define MAX_MSG_SIZE 2 * 1024 * 1024

using grpc::Channel;
using grpc::Server;
using grpc::ServerBuilder;
using grpc::ServerContext;
using grpc::ClientContext;
using grpc::Status;
using grpc::ClientAsyncResponseReader;
using grpc::ServerAsyncResponseWriter;
using grpc::ServerCompletionQueue;
using grpc::CompletionQueue;

using doinference::ScheduleRequest;
using doinference::ScheduleReply;
using doinference::Scheduler;

static size_t copyStringToBuf(std::string srcString, char* destBuf)
{
  for(size_t j=0;j<srcString.size();j++)
    destBuf[j] = srcString[j];
  return srcString.size();
}

static std::string copyBufToString(char* srcBuf, int len)
{
  return std::string(srcBuf, len);
}
