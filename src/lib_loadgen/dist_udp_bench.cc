// Copyright (c) Adithya Kumar, The Pennsylvania State University. All rights reserved.
// Licensed under the MIT License.

/**
 * Benchmark tool to test peformance of application. 
 */
#include "stats_factory.h"
#include "load_generator.h"
#include "common_defs.h"
#include "spdlog/cfg/env.h"
#include "spdlog/sinks/basic_file_sink.h"

#include "base_udp_client.h"
#include "dpdk_zc_udp_client.h"

extern size_t app_req_size;
extern size_t app_resp_size;
extern AppReq_cb app_req_cb;
extern AppResp_cb app_resp_cb;

uint64_t prevReqTimeNs;
uint32_t nextReqIndex = 0;
uint32_t total_requests;

BaseClient *Client;

// List of list of <request, start-time> (one list per thread)
std::vector<std::vector<std::pair<void*, uint64_t>>> ReqInRam;

int
UdpRxListener(void *c) {
  Client->CompleteReqListener();
  return 0;
}

int 
genReq(void *args) {
  int tid = *(int *)(args);
  if(tid < 0 || tid >= ReqInRam.size()){
    spdlog::critical("tid out of range of ReqInRam for tid: {}", tid);
    return -1;
  } else {
    spdlog::info("Starting genReq tid: {}", tid);
  }
  std::vector<std::pair<void*, uint64_t>> &myQueue = ReqInRam[tid];
  for(const auto p : myQueue) {
    sleepUntil(p.second);
    Client->SchedRequest(p.first);
  }
  return 0;
}

int main(int argc, char** argv) {
  // Setup logging
  try { 
    spdlog::set_default_logger(spdlog::basic_logger_mt("P2pRpc", "p2p_rpc_bench.log"));
    spdlog::cfg::load_env_levels();
    spdlog::flush_every(std::chrono::seconds(3));
  } catch(const spdlog::spdlog_ex &ex) {
    std::cout << "Log init failed: " << ex.what() << std::endl;
  }

  /*** Read all configurations ***/
  seedGenerator(readEnvInfo<uint32_t>("P2P_RPC_RANDOM_SEED", 12345));
  std::string exp_name = readEnvInfo<std::string>\
                         ("P2P_RPC_BENCH_EXP_NAME", std::string("RPC_BENCH"));
  std::string rpc_uri  = readEnvInfo<std::string>\
                         ("P2P_RPC_URI", "");
  if(rpc_uri.compare("") == 0) {
    spdlog::critical("P2P_RPC_URI needs to be set");
    exit(1);
  }
  total_requests        = readEnvInfo<uint32_t>\
                                ("P2P_RPC_BENCH_TOTAL_REQUESTS", 100);
  // For open-loop this represents lambda, for closed-loop this represents N
  double arrival_rate   = readEnvInfo<double>\
                          ("P2P_RPC_BENCH_ARR_RATE", 1.0);
  uint16_t req_gen_type = readEnvInfo<uint16_t>\
                          ("P2P_RPC_REQ_GEN_TYPE", 0); 
  uint32_t num_warmup   = readEnvInfo<uint32_t>\
                          ("P2P_RPC_NUM_WARMUP",(total_requests * 10)/100);
  uint16_t n_gen_thr      = readEnvInfo<uint16_t>\
                            ("P2P_RPC_N_GEN_THR", 1);
  uint16_t n_listener_thr = readEnvInfo<uint16_t>\
                            ("P2P_RPC_N_LISTENER_THR", 1); 
  uint16_t client_type = readEnvInfo<uint16_t>\
                            ("P2P_RPC_CLIENT_TYPE", 0);

  StatsManager *stats = new StatsManager(exp_name);
  LoadGenerator *LoadGen = new LoadGenerator((1.0/arrival_rate));

  if(client_type == 0) {
    spdlog::info("Creating a linux based udp client");
    Client = new BaseUdpClient(rpc_uri, stats);
  } else if(client_type == 1) {
    spdlog::info("creating dpdk based udp client");
    int ret = rte_eal_init(argc, argv);
    if(ret < 0)
      rte_exit(EXIT_FAILURE, "Error with EAL initialization");
    argc -= ret;
    argv += ret;
    //Client = new DpdkUdpClient(rpc_uri, stats);
    Client = new DpdkZcUdpClient(rpc_uri, stats);
  }

  Client->SetMaxPayloadSize(app_req_size, app_resp_size);
  Client->SetReqRespCb(app_req_cb, app_resp_cb);
  Client->SetWarmupRequests(num_warmup);
  Client->SetMaxRequests(total_requests + num_warmup);
  if(req_gen_type == 0)
    Client->SetClosedLoop();

  std::vector<std::thread> req_listener_threads, req_gen_threads;
  std::vector<unsigned int> req_listener_cores, req_gen_cores;
  if(client_type == 1) {
    unsigned int curr_core_id = rte_lcore_id();
    for(int i=0; i < n_listener_thr; i++) {
      curr_core_id = rte_get_next_lcore(curr_core_id, true, false);
      curr_core_id = rte_get_next_lcore(curr_core_id, true, false);
      if(curr_core_id == RTE_MAX_LCORE)
        rte_exit(EXIT_FAILURE, "Not enough LCORES\n");
      req_listener_cores.push_back(curr_core_id);
    }
    for(int i=0; i < n_gen_thr; i++) {
      curr_core_id = rte_get_next_lcore(curr_core_id, true, false);
      curr_core_id = rte_get_next_lcore(curr_core_id, true, false);
      if(curr_core_id == RTE_MAX_LCORE)
        rte_exit(EXIT_FAILURE, "Not enough LCORES\n");
      req_gen_cores.push_back(curr_core_id);
    }
  }

  if(client_type == 0) {
    for(int i=0; i < n_listener_thr; i++)
      req_listener_threads.push_back(std::thread(&BaseClient::CompleteReqListener, Client));
  } else if(client_type == 1) {
    for(int i=0; i < n_listener_thr; i++) {
      if(rte_eal_remote_launch(UdpRxListener, (void*)Client, req_listener_cores[i]) !=0)
        rte_exit(EXIT_FAILURE, "Unable to launch listener\n");
    }
  }

  if(req_gen_type == 1) {
    for(int i = 0 ; i < num_warmup; i++) {
      Client->SchedRequest(Client->CreateRequest(NULL));
    }
    for(int i = 0 ; i < n_gen_thr; i++) {
      ReqInRam.push_back(std::vector<std::pair<void*, uint64_t>>());
    }

    // Partition arrivals to all threads
    // Start sometime in the future
    prevReqTimeNs = getCurNs() + (total_requests * 1E4);
    stats->startExp(prevReqTimeNs);
    uint16_t thr_idx = 0;
    for(int i = 0 ; i < total_requests; i++) {
      void *req = Client->CreateRequest(NULL);
      prevReqTimeNs = LoadGen->nextReqArr(prevReqTimeNs, false);
      ReqInRam[thr_idx].push_back(std::make_pair(req, prevReqTimeNs));
      thr_idx = (thr_idx + 1) % n_gen_thr;
    }

    spdlog::info("Starting open-loop generator, Arrival Rate: {} RPS", arrival_rate);
    int tids[n_gen_thr];
    if(client_type == 0) {
      for(int i=0; i< n_gen_thr; i++) {
        tids[i] = i;
        req_gen_threads.push_back(std::thread(genReq, &tids[i]));
      }
      for(int i=0; i< n_gen_thr; i++) {
        req_gen_threads[i].join();
      }
    } else if (client_type == 1) {
      for(int i=0; i< n_gen_thr; i++) {
        tids[i] = i;
        if(rte_eal_remote_launch(genReq, &tids[i], req_gen_cores[i]) != 0)
          rte_exit(EXIT_FAILURE, "Unable to launch generator\n");
      }
      for(int i=0; i< n_gen_thr; i++) {
        rte_eal_wait_lcore(req_gen_cores[i]);
      }
    }
    spdlog::info("Generated {} requests, took... {} ms", total_requests, (getCurNs() - stats->expStartTimeNs)/1E6);
  }

  if(req_gen_type == 0) {
    spdlog::info("Starting closed-loop generator");
    stats->startExp();
    if(client_type == 0) { 
      for(int i = 0; i < arrival_rate; i++) {
        void *tag = Client->CreateRequest(NULL);
        if(tag != NULL)
          Client->SchedRequest(tag);
      }
    } else if(client_type == 1) {
      Client->PreProcessRequests();
      for(int i = 0; i < arrival_rate; i++) {
        Client->SchedRequest(NULL);
      }
    }
  }

  // Wait for listeners to be done 
  if(client_type == 0) {
    for(int i=0; i < n_listener_thr; i++)
      req_listener_threads[i].join();
  } else if (client_type == 1) {
    for(int i=0; i < n_listener_thr; i++)
      rte_eal_wait_lcore(req_listener_cores[i]);
  }

  stats->endExp();
  Client->PostProcessResponses();
  spdlog::info("Benchmark Complete...");

  // Save stats
  double execTimeS = (stats->expEndTimeNs - stats->expStartTimeNs) / 1E9;
  double throughput = static_cast<double>(total_requests) / execTimeS; 
  stats->trackStatInfo("exp_name", exp_name);
  stats->trackStatInfo("num_warmup", static_cast<uint64_t>(num_warmup));
  stats->trackStatInfo("arrival_rate", static_cast<double>(arrival_rate));
  stats->trackStatInfo("total_req", static_cast<uint64_t>(total_requests));
  stats->trackStatInfo("achieved_throughput", throughput);
  stats->saveStats({"sojournTime"}, readEnvInfo<bool>("P2P_RPC_DUMP_RAW_STATS", false));

  delete Client;
  delete LoadGen;
  delete stats;
  return 0;
}
