// Copyright (c) Adithya Kumar, The Pennsylvania State University. All rights reserved.
// Licensed under the MIT License.

#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <vector>
#include <iostream>
#include <signal.h>

#include "time_utils.h"
#include "config_utils.h"
#include "stats_utils.h"

#include "ort_app.h"
#include "batched_grpc_handler.h"

size_t inputLen = get_req_size();
size_t outputLen = get_resp_size();
#ifdef PROFILE_MODE
    uint64_t SStartNs, GStartNs, AppStartNs;
    std::vector<uint64_t> SDelay, GDelay, AppDelay;
#endif

volatile bool force_quit = 0;
static void 
signal_handler(int signum) {                                                                          
  if (signum == SIGINT) {        
      printf("\n\nSignal %d received, preparing to exit...\n", signum);      
      ACCESS_ONCE(force_quit) = 1;
    }
}

// Globally shared
OrtFacade AppServer;
int max_bs = get_ort_batch_size();

void appCopyIn(void *req, int in_bs_idx)
{
  assert(in_bs_idx >= 0);
#ifdef PROFILE_MODE
  GStartNs = getCurNs();
#endif
  AppServer.copyInputs(req, in_bs_idx);
#ifdef PROFILE_MODE
  AppServer.sync();
  GDelay.push_back(getCurNs() - GStartNs);
#endif
}

size_t appCopyOut(void *resp, int out_bs_idx)
{
  assert(out_bs_idx >= 0);
#ifdef PROFILE_MODE
  SStartNs = getCurNs();
#endif
  AppServer.copyOutputs(resp, out_bs_idx);
#ifdef PROFILE_MODE
  AppServer.sync();
  SDelay.push_back(getCurNs() - SStartNs);
#endif
  return outputLen;
}

size_t appRun(int bs)
{
  assert(bs >= 1 && bs <= max_bs);
#ifdef PROFILE_MODE
  AppStartNs = getCurNs();
#endif
  AppServer.run_on_loaded_data(bs);
#ifdef PROFILE_MODE
  AppServer.sync();
  AppDelay.push_back(getCurNs() - AppStartNs);
#endif
  //AppServer.predict_with_io_binding();
  return outputLen;
}

void run_grpc()
{
  std::string model_path = getDatasetBasePath() + std::string("/models/") + get_ort_model_name();
  AppServer.loadModel(model_path);
  //AppServer.printModelInfo();
  
  printf("Initializing ort with model: %s, max_batch_size: %d\n", model_path.c_str(), max_bs);

  std::vector<void*> d_inp_data, d_out_data;
  //AppServer.setup_io_binding(d_inp_data, d_out_data, max_bs);
  AppServer.do_device_mem_allocations(max_bs);
  printf("Warming up for all batch-sizes...\n");
  for(int i = 1 ; i <= max_bs ; i++)
    AppServer.run_on_loaded_data(i);
  printf("App warmed up...\n");

  std::string m_uri = get_server_ip() + std::string(":") + get_server_port();
  BatchedGrpcAsyncRequestHandler reqH(1, m_uri, max_bs);
  reqH.BuildServer();
  reqH.Run();
  while(ACCESS_ONCE(force_quit) == 0);
  printf("Stopping Async Handler\n");
  reqH.quit();
}

int main()
{
  // Install signal handlers to quit
  signal(SIGINT, signal_handler);

  printf("GrpcBench, Starting server (remember to start client)...\n");
  run_grpc();
  
#ifdef PROFILE_MODE
  PROF_PRINT("H2D", GDelay);
  PROF_PRINT("D2H", SDelay);
  PROF_PRINT("App", AppDelay);
  printPbStats();
#endif
  return 0;
}

