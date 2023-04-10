// Copyright (c) Adithya Kumar, The Pennsylvania State University. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <thread>

#include "grpc_common.h"
#include "config_utils.h"
#include "stats_utils.h"
#include "time_utils.h"
#include "concurrentqueue.h"

extern size_t appRun(int);
extern void appCopyIn(void*, int);
extern size_t appCopyOut(void*, int);

class BatchedGrpcAsyncRequestHandler  {
  int nServerThreads;                         
  std::string MyServerSockString;               // IP:Port of the server

  std::unique_ptr<ServerCompletionQueue> cq_; // To store the events
  Scheduler::AsyncService service_;           // To inform the GRPC

  volatile bool force_quit = 0;
  
  std::vector<std::thread> serverThreads;
  
  int batch_size;

#ifdef PROFILE_MODE
  static uint64_t startNs;
  static std::vector<uint64_t> metricValues;
#endif
  // Has to be thread-local
  char *h_out;
  public:
  BatchedGrpcAsyncRequestHandler(int nThreads=1, std::string MyServerAddr="localhost:50052", int bs=1) 
    : nServerThreads(nThreads), MyServerSockString(MyServerAddr)
    {
      h_out = new char[MAX_MSG_SIZE];
      batch_size = bs;
    }

  ~BatchedGrpcAsyncRequestHandler() {
    delete h_out;
  }

  Scheduler::AsyncService* getService() {
    return &service_;
  }

  void setCompletionQueue(std::unique_ptr<ServerCompletionQueue> cq) {
    cq_ = std::move(cq);
  }
  
  void quit()
  {
    server->Shutdown();
    // Always shutdown the completion queue after the server.
    cq_->Shutdown();
    
    spdlog::info("Shutdown server and cq, waiting for workers to join");
    ACCESS_ONCE(force_quit) = 1;

    for(auto& t_worker : serverThreads) {
      t_worker.join();
    }
    //for(int i=0; i < nServerThreads + 1; i++) {
    //  serverThreads[i].join();
    //}
#ifdef PROFILE_MODE
    PROF_PRINT("RPC Routing stats", metricValues); 
#endif
  }

  // Initializes the server
  void Init()
  {
  }

  // There is no shutdown handling in this code.
  void Run(bool blocking = false)
  {
    // Proceed to the server's main loop.
    // Has to be single threaded if it is a M/M/1 type of a system
    for(int i=0; i< nServerThreads; i++) {
      serverThreads.push_back(std::thread(&BatchedGrpcAsyncRequestHandler::HandleRpcs, this));
    }
    serverThreads.push_back(std::thread(&BatchedGrpcAsyncRequestHandler::AppWorker, this));

    if(blocking) {
      // And now we wait ... Forever!
      for(int i=0; i< nServerThreads; i++) {
        serverThreads[i].join();
      }
    }
  }

  void BuildServer()
  {
    ServerBuilder builder;
    // Listen on the given address without any authentication mechanism.
    builder.AddListeningPort(MyServerSockString, grpc::InsecureServerCredentials());
    builder.SetMaxReceiveMessageSize(MAX_MSG_SIZE);

    // Register "service_" as the instance through which we'll communicate with
    // clients. In this case it corresponds to an *asynchronous* service.
    builder.RegisterService(getService());

    // Get hold of the completion queue used for the async comm with gRPC
    setCompletionQueue(std::move(builder.AddCompletionQueue()));
    spdlog::info("RequestHandler service started, listening on: {}", MyServerSockString);

    // Finally assemble the server.
    server = builder.BuildAndStart();
  }

  // Blocking call, won't return
  void WaitForServer()
  {
    server->Wait();
  }

  private:
  std::unique_ptr<Server> server;

  // Class encompasing the state and logic needed to serve a request.
  class RequestInternal {
    char *h_out;
    public:
      // Take in the "service" instance (in this case representing an asynchronous
      // server) and the completion queue "cq" used for asynchronous communication
      // with the gRPC runtime.
      RequestInternal(Scheduler::AsyncService* service, ServerCompletionQueue* cq, char* _h_out)
        : service_(service), cq_(cq), responder_(&ctx_), status_(CREATE) {
          // Invoke the serving logic right away.
          h_out = _h_out;
          Proceed();
        }

      // Main Loop which runs the state machine which processes
      // the request data and transitions to various states
      void Proceed(int bs_idx=-1)
      {
        if (status_ == CREATE) {
          //status_ = PROCESS;
          status_ = RECVD;
          service_->RequestrunJob(&ctx_, &request_, &responder_, cq_, cq_, this);
        } else if(status_ == RECVD) {
          new RequestInternal(service_, cq_, h_out);
          reply_.set_guid(request_.guid());
          status_ = COPYIN;
        } else if(status_ == COPYIN) {
          // Copy stuff from request to reply - for verification
          void* h_in = (void*)request_.payload().c_str();
          appCopyIn(h_in, bs_idx);
          status_ = COPYOUT;
        } else if(status_ == COPYOUT) {
          size_t len = appCopyOut(h_out, bs_idx);
          reply_.set_payload(copyBufToString(h_out, len));
          if(len == 0) {
            spdlog::critical("SERVER-> Req: {} ERROR!!!", request_.guid());
          }
          status_ = FINISH;
          responder_.Finish(reply_, Status::OK, this);
          spdlog::debug("SERVER-> Req: {} Reply-Complete", request_.guid());
        } else if (status_ == PROCESS) {
//          new RequestInternal(service_, cq_, h_out);
//#ifdef PROFILE_MODE
//          startNs = getCurNs();
//#endif
//          // Copy stuff from request to reply - for verification
//          reply_.set_guid(request_.guid());
//
//          // Call the application
//          void *h_in = (void*)request_.payload().c_str();
//
//          size_t len = appRun(h_in, h_out);
//          reply_.set_payload(copyBufToString(h_out, len));
//
//          if(len == 0) {
//            spdlog::critical("SERVER-> Req: {} ERROR!!!", request_.guid());
//          }
//
//          status_ = FINISH;
//          responder_.Finish(reply_, Status::OK, this);
//          spdlog::debug("SERVER-> Req: {} Reply-Complete", request_.guid());
//#ifdef PROFILE_MODE
//          metricValues.push_back(getCurNs() - startNs);
//#endif
        } else {
          GPR_ASSERT(status_ == FINISH);
          // Once in the FINISH state, deallocate ourselves (RequestInternal).
          delete this;
        }
      }
      
      // The means of communication with the gRPC runtime for an asynchronous
      // server.
      Scheduler::AsyncService* service_;

      // The producer-consumer queue where for asynchronous server notifications.
      ServerCompletionQueue* cq_;
      // Context for the rpc, allowing to tweak aspects of it such as the use
      // of compression, authentication, as well as to send metadata back to the
      // client.
      ServerContext ctx_;

      // What we get from the client.
      ScheduleRequest request_;
      // What we send back to the client.
      ScheduleReply reply_;

      // The means to get back to the client.
      ServerAsyncResponseWriter<ScheduleReply> responder_;

      // Let's implement a tiny state machine with the following states.
      enum CallStatus { CREATE, RECVD, COPYIN, COPYOUT, PROCESS, FINISH };
      CallStatus status_;  // The current serving state.
  };
  
  moodycamel::ConcurrentQueue<RequestInternal*> in_flight;

  void AppWorker()
  {
    printf("Spawned App Worker...\n");

    std::vector<RequestInternal*> curr_batch;
    RequestInternal *req;
    while(ACCESS_ONCE(force_quit) == 0)
    {
      while(in_flight.try_dequeue(req))
      {
        curr_batch.push_back(req);
        if(curr_batch.size() == batch_size)
          break;
      }
      if(curr_batch.size() > 0) {
        // CopyIn
        int bs_idx = 0;
        for(auto req : curr_batch) {
          req->Proceed(bs_idx++); // RECVD -> COPYIN
        }
        // Do Work
        appRun(curr_batch.size());
        // CopyOut
        bs_idx = 0;
        for(auto req : curr_batch) {
          req->Proceed(bs_idx++); // COPYIN -> COPYOUT
        }
        curr_batch.clear();
      }
    }
    printf("Terminating App Worker...\n");
  }

  // This can be run in multiple threads if needed.
  void HandleRpcs()
  {
    printf("Spawned handler...\n");
    // Spawn a new RequestInternal instance to serve new clients.
    for(int i = 0 ; i < 16 ; i++)
      new RequestInternal(&service_, cq_.get(), h_out);
    void* tag;  // uniquely identifies a request.
    bool ok;
    //uint64_t cq_timeout_ms = 50;
    //in_flight.reserve(batch_size);

    while (ACCESS_ONCE(force_quit) == 0) {
      ::grpc::CompletionQueue::NextStatus cq_status = cq_->AsyncNext(&tag, &ok, \
          std::chrono::system_clock::now() + std::chrono::milliseconds(0));
      while(cq_status == ::grpc::CompletionQueue::GOT_EVENT) {
        if(cq_status == ::grpc::CompletionQueue::GOT_EVENT && ok) {
          RequestInternal* nextReq = static_cast<RequestInternal*>(tag);
          if(nextReq->status_ == RequestInternal::FINISH) {
            // If this is a completion, let it go
            nextReq->Proceed();
          } else {
#ifdef PROFILE_MODE
            startNs = getCurNs();
#endif
            nextReq->Proceed(); // CREATE -> RECVD
            in_flight.enqueue(nextReq);
#ifdef PROFILE_MODE
            metricValues.push_back(getCurNs() - startNs);
#endif
          }
        }
        cq_status = cq_->AsyncNext(&tag, &ok, \
            std::chrono::system_clock::now() + std::chrono::milliseconds(0));
      }
      if(cq_status == ::grpc::CompletionQueue::SHUTDOWN) {
        spdlog::critical("Server has been shutdown!!!");
        break;
      }

      //::grpc::CompletionQueue::NextStatus cq_status = cq_->AsyncNext(&tag, &ok, std::chrono::system_clock::now() + std::chrono::milliseconds(cq_timeout_ms));
      //::grpc::CompletionQueue::NextStatus cq_status = cq_->AsyncNext(&tag, &ok, std::chrono::system_clock::now() + std::chrono::milliseconds(0));
      //if(cq_status == ::grpc::CompletionQueue::GOT_EVENT) {
      //  if(!ok) {
      //    spdlog::critical("Failed request");
      //    continue;
      //  }
      //  RequestInternal* nextReq = static_cast<RequestInternal*>(tag);
      //  spdlog::debug("SERVER-> Req: {} Received", nextReq->request_.guid());
      //  if(nextReq->status_ == RequestInternal::FINISH) {
      //    nextReq->Proceed();
      //  } else {
      //    nextReq->Proceed(); // CREATE -> RECVD
      //    in_flight.push_back(nextReq);
      //    nextReq->Proceed(in_flight.size() - 1); // RECVD -> COPYIN
      //    if(in_flight.size() == batch_size) {
      //      appRun(batch_size);
      //      int bs_idx = 0;
      //      for(auto req : in_flight)
      //        req->Proceed(bs_idx++); // COPYIN -> COPYOUT
      //      in_flight.clear();
      //    }
      //  }
      //} else if(cq_status == ::grpc::CompletionQueue::SHUTDOWN) {
      //  spdlog::critical("Server has been shutdown!!!");
      //  break;
      //}
    }
    spdlog::info("Handler stopped");
  }
};

#ifdef PROFILE_MODE
  uint64_t BatchedGrpcAsyncRequestHandler::startNs;
  std::vector<uint64_t> BatchedGrpcAsyncRequestHandler::metricValues;
#endif
