// Copyright (c) Adithya Kumar, The Pennsylvania State University. All rights reserved.
// Licensed under the MIT License.

#include <cmath>
#include <algorithm>
#include <signal.h>

#include "time_utils.h"
#include "config_utils.h"

// Transport related
#include "transport/dpdk_init.h"
#include "transport/dpdk_transport_ng.h"
#include "p2p_rpc_rr_ng.h"
#include "p2p_rpc_rr_pool_ng.h"
P2pRpcRRPool *rpc_rr_pool = NULL;
struct p2p_rpc_conn_info *server_conn_info = NULL;

#include "p2p_rpc_sg_engine.h"
P2pRpcSgEngine *copy_engine = NULL;

// App related
#include "p2p_rpc_app_rr.h"
#include "p2p_rpc_async_app_server.h"
extern AppInitCB AppInit_cb;
extern AppRunCB AppRun_cb;
extern AppCleanupCB AppCleanup_cb;
extern AppRunAsyncCB AppRunAsync_cb;
extern AppRunWaitCB AppRunWait_cb;
extern AppCompleteCB AppComplete_cb;
P2pRpcAppInfo *app_info = NULL;
P2pRpcAsyncAppServer *app_server = NULL;

volatile bool force_quit = 0;
static void 
signal_handler(int signum) {                                                                          
  if (signum == SIGINT) {        
      printf("\n\nSignal %d received, preparing to exit...\n", signum);      
      ACCESS_ONCE(force_quit) = 1;
      if(app_server)
        app_server->quit();
    }
}

int rx_tx_loop(void *conf)
{
  struct dpdk_ctx *_dpdk_ctx = (dpdk_ctx *)conf;
  printf("rx_tx_loop: Starting Core %u [Ctrl+C to quit]\n",
      rte_lcore_id());
  uint32_t num_tx = 0, num_rx = 0;
  while(ACCESS_ONCE(force_quit) == 0) 
  {
    // RX
    if(_dpdk_ctx->zerocopy_mode == 0) {
      rr_recv_ng(_dpdk_ctx, rpc_rr_pool, force_quit);
    } else {
      rr_recv(_dpdk_ctx, rpc_rr_pool, force_quit);
    }

    // TX
    //P2pRpcRr *tx_rpc_rr = rpc_rr_pool->consume_tx_ready_rr();
    //if(tx_rpc_rr != NULL) {
    //  if (unlikely(rr_send(_dpdk_ctx, tx_rpc_rr, *rpc_rr_pool) == 0))
    //    printf("TX Error RPC server\n");
    //  num_tx++;
    //}
  }
  printf("Ending RX/TX loop thread, num_rx: %d, num_tx: %d\n", num_rx, num_tx);
  return 1;
}

int shunter_loop(void *conf)
{
  struct dpdk_ctx *_dpdk_ctx = (dpdk_ctx *)conf;
  printf("shunter loop: Starting Core %u [Ctrl+C to quit]\n",
      rte_lcore_id());

  TRACE_PRINTF("Setting up CopyEngine\n");
  if(_dpdk_ctx->zerocopy_mode == 1) {
    copy_engine = new P2pRpcSgZcEngine();
  } else {
    if(_dpdk_ctx->mem_alloc_type == MEM_ALLOC_TYPES::HOST_MEM_ONLY) {
      copy_engine = new P2pRpcSgCpuEngine();
    } else {
      copy_engine = new P2pRpcSgGpuEngine(get_cuda_device_id());
    }
  }
#ifdef PROFILE_MODE
  uint64_t rxStartNs, txStartNs;
  std::vector<uint64_t> RxDelay, TxDelay;
#endif

  P2pRpcRr *new_rpc_rr = NULL, *tx_rpc_rr = NULL;
  P2pRpcAppRr *new_app_rr = NULL, *tx_app_rr = NULL;

  uint32_t num_rx = 0, num_tx = 0;

  int rr_ci_idx = 0, rr_pi_idx = 0;

  P2pRpcAppRrPool *app_rr_pool = app_info->app_rr_pool;
  int app_rr_pool_size = app_rr_pool->get_pool_size();
  
  // FIXME: This assumes the first item is pre-created
  new_app_rr = app_rr_pool->get_app_rr(rr_pi_idx);
  tx_app_rr = app_rr_pool->get_app_rr(rr_ci_idx);

  printf("Starting busy loop to poll for requests\n");
  while(ACCESS_ONCE(force_quit) == 0) 
  {
    // RX_COMPLETEr
    // 1. Get the RPC_RR
    // 2. Allocate an APP_RR
    // 3. Copy the transport buffers into payload
    // 4. Notify so that app-worker can pick it up
    // 5. Release the transport buffers (Skip this step in case of ZC)
    // The first check is basically the flow-control
    if(((rr_pi_idx + 1) % app_rr_pool_size != rr_ci_idx) &&
        rpc_rr_pool->get_next_rx_ready_rr(&new_rpc_rr) != 0) {
      TRACE_PRINTF("RX: rpc_rr: %p, app_rr: %p src-ip %s, dst-ip %s, rr_token: %lu, req_size: %ld\n",
          (void*)new_rpc_rr, (void*)new_app_rr, ipv4_to_string(get_ip_header(&(new_rpc_rr->_client_conn_info.hdr_template))->src_ip).c_str(),
          ipv4_to_string(get_ip_header(&(new_rpc_rr->_client_conn_info.hdr_template))->dst_ip).c_str(),
          new_rpc_rr->req_token, new_rpc_rr->payload_size);
      //NVTX_R("RX");
#ifdef PROFILE_MODE
      rxStartNs = getCurNs();
#endif
      num_rx++;
      new_app_rr->rpc_rr = (void*)new_rpc_rr;
      copy_engine->gather(new_rpc_rr, new_app_rr->h_stub);
      *ACCESS_ONCE(new_app_rr->h_state) = APP_RR_STATUS::RX_COMPLETE;

      if(_dpdk_ctx->zerocopy_mode == 0) {
        rr_release_mbufs(_dpdk_ctx, new_rpc_rr);
        //new_rpc_rr->payload_size = app_rr_pool->resp_size;
        rr_alloc_mbufs(_dpdk_ctx, new_rpc_rr, app_rr_pool->resp_size);
        rr_set_hdr(&new_rpc_rr->_client_conn_info, new_rpc_rr, app_rr_pool->resp_size);
      } else { 
        rr_swap_hdr(_dpdk_ctx, new_rpc_rr, app_rr_pool->resp_size);
      }
      TRACE_PRINTF("RX Complete rr_pi_idx: %d, app_rr: %p\n", rr_pi_idx, (void*)new_app_rr);
      rr_pi_idx = (rr_pi_idx + 1) % app_rr_pool_size;
      app_rr_pool->get_next();
      new_app_rr = app_rr_pool->get_app_rr(rr_pi_idx);
      TRACE_PRINTF("Next APP_RR rr_pi_idx: %d, app_rr: %p, state: %d\n", 
          rr_pi_idx, (void*)new_app_rr, *(ACCESS_ONCE(new_app_rr->h_state)));
#ifdef PROFILE_MODE
      RxDelay.push_back(getCurNs() - rxStartNs);
#endif
      //NVTX_P;
    }

    // TX_COMPLETEr
    // 1. Allocate new transport buffers (SKIP if ZC)
    // 2. Set the header for the transport buffers (Do SwapHdr in case of ZC)
    // 3. Copy the payload into transport buffers (SKIP if ZC)
    // 4. Notify App (it is marked as TX_COMPLETE once the data has been copied to the transport mbufs)
    // 5. Do the actual TX
    if(*ACCESS_ONCE(tx_app_rr->h_state) == APP_RR_STATUS::WORK_COMPLETE) {
#ifdef PROFILE_MODE
      txStartNs = getCurNs();
#endif
      NVTX_R("TX");
      tx_rpc_rr = (P2pRpcRr*)tx_app_rr->rpc_rr;
      TRACE_PRINTF("TX RPC_RR: %p, my-ip %s, sender-ip %s, rr_token: %lu, resp_size: %ld\n",
          (void*)tx_rpc_rr, ipv4_to_string(get_ip_header(&tx_rpc_rr->_client_conn_info.hdr_template)->src_ip).c_str(),
          ipv4_to_string(get_ip_header(&tx_rpc_rr->_client_conn_info.hdr_template)->dst_ip).c_str(),
          tx_rpc_rr->req_token, tx_rpc_rr->payload_size);

      num_tx++;
      if(_dpdk_ctx->zerocopy_mode == 0) {
        copy_engine->scatter(tx_rpc_rr, tx_app_rr->h_stub);
      } 

      //rpc_rr_pool->mark_tx_ready_rr(tx_rpc_rr);
      *ACCESS_ONCE(tx_app_rr->h_state) = APP_RR_STATUS::FREE;
      if (unlikely(rr_send(_dpdk_ctx, tx_rpc_rr, *rpc_rr_pool) == 0))
        printf("TX Error RPC server\n");
      TRACE_PRINTF("TX Complete rr_ci_idx: %d\n", rr_ci_idx);
      tx_app_rr->rpc_rr = NULL;
      //*ACCESS_ONCE(tx_app_rr->h_state) = APP_RR_STATUS::FREE;
      //_mm_mfence();

      rr_ci_idx = (rr_ci_idx + 1) % app_rr_pool_size;
      tx_app_rr = app_rr_pool->get_app_rr(rr_ci_idx);
#ifdef PROFILE_MODE
      TxDelay.push_back(getCurNs() - txStartNs);
#endif
      NVTX_P;
    }
  }
  printf("Ending shunter loop thread copy_in: %d, copy_out: %d\n", num_rx, num_tx);
  printf("Stopping CopyEngine...\n");
  PROF_PRINT("CopyIn: ", RxDelay);
  PROF_PRINT("CopyOut: ", TxDelay);
  delete copy_engine;
  return 1;
}

int
main(int argc, char *argv[]) {
////////////////////////// Application initializations
  int ret = rte_eal_init(argc, argv);
  if (ret < 0)
    rte_exit(EXIT_FAILURE, "Error with EAL initialization\n");
  argc -= ret;
  argv += ret;
  signal(SIGINT, signal_handler);

/////////////////////////// RPC setup

  struct dpdk_ctx *_dpdk_ctx = new dpdk_ctx;
  _dpdk_ctx->nic_port = get_dpdk_port();
  _dpdk_ctx->queue_id = 0;
  _dpdk_ctx->mem_alloc_type = get_dpdk_mem_alloc_type();
  _dpdk_ctx->device_id = get_cuda_device_id();
  _dpdk_ctx->zerocopy_mode = is_zerocopy_mode(); 
  if(init_dpdk_ctx(_dpdk_ctx) == 0) {
    printf("Failed to init dpdk\n");
    exit(1);
  }

  std::string src_uri = get_server_mac() + std::string(",") 
    + get_server_ip() + std::string(",") 
    + get_server_port();

  server_conn_info = initialize_src_conn_info(src_uri);
  TRACE_PRINTF("Setting up P2pRpcRRPool with src_uri: %s, server_conn_info: %p\n", src_uri.c_str(), server_conn_info);
  rpc_rr_pool = new P2pRpcRRPool;
  rpc_rr_pool->setup_and_init_rr_pool(server_conn_info, get_req_size()); 

////////////////////////////// App Setup

  TRACE_PRINTF("Setting up P2pRpcAppInfo\n");
  app_info = new P2pRpcAppInfo(AppInit_cb, AppRun_cb, AppCleanup_cb, AppComplete_cb, 
      get_cuda_device_id(), get_req_size(), get_resp_size());
  if(app_info == NULL) {
    printf("Failed to create app info...\n");
    exit(1);
  }

  TRACE_PRINTF("Setting up P2pRpcAsyncAppServer\n");
  app_server = new P2pRpcAsyncAppServer(); 
  if(app_server == NULL) {
    printf("Failed to create app server...\n");
    exit(1);
  }
  app_info->appIdx = app_server->register_app(app_info);

////////////////////////// Sample run of app
  
  int max_batch_size = get_ort_batch_size();
  int server_mode = get_server_mode();
  P2pRpcAppRr *t_all_app_rrs[max_batch_size];
  for(int i = 0 ; i < max_batch_size;i++)
    t_all_app_rrs[i] = app_info->app_rr_pool->get_next();

  printf("Server mode: %d, warmup...\n", server_mode);
  for(int i = max_batch_size ; i >= 1;i--) {
    if(server_mode == 2)
      app_server->do_batch_work_sync(app_info->appIdx, 0, i);
    else
      app_server->do_work_sync(app_info->appIdx, 0);
  }
  
  //printf("Server warmed up...\n"); 
  //printf("Test run... batchsize: 1\n");
  //SetDummyData(t_all_app_rrs[0]->h_stub->resp, t_all_app_rrs[0]->resp_size, 1);
  //SetDummyData(t_all_app_rrs[1]->h_stub->resp, t_all_app_rrs[1]->resp_size, 1);
  //SetDummyData(t_all_app_rrs[0]->h_stub->req, t_all_app_rrs[0]->req_size, 5);
  //SetDummyData(t_all_app_rrs[1]->h_stub->req, t_all_app_rrs[1]->req_size, 5);
  //printf("Before running inference...\n");
  //g_floatDump(t_all_app_rrs[0]->h_stub->resp, t_all_app_rrs[0]->resp_size);
  //g_floatDump(t_all_app_rrs[1]->h_stub->resp, t_all_app_rrs[1]->resp_size);
  //g_intDump(t_all_app_rrs[0]->h_stub->resp, t_all_app_rrs[0]->resp_size);
  //g_intDump(t_all_app_rrs[1]->h_stub->resp, t_all_app_rrs[1]->resp_size);
  //app_server->do_batch_work_sync(app_info->appIdx, 0, 1);
  //printf("After running inference...\n");
  //g_floatDump(t_all_app_rrs[0]->h_stub->resp, t_all_app_rrs[0]->resp_size);
  //g_floatDump(t_all_app_rrs[1]->h_stub->resp, t_all_app_rrs[1]->resp_size);
  //g_intDump(t_all_app_rrs[0]->h_stub->resp, t_all_app_rrs[0]->resp_size);
  //g_intDump(t_all_app_rrs[1]->h_stub->resp, t_all_app_rrs[1]->resp_size);

  //printf("Test run... batchsize: 2\n");
  //SetDummyData(t_all_app_rrs[0]->h_stub->resp, t_all_app_rrs[0]->resp_size, 1);
  //SetDummyData(t_all_app_rrs[1]->h_stub->resp, t_all_app_rrs[1]->resp_size, 1);
  //SetDummyData(t_all_app_rrs[0]->h_stub->req, t_all_app_rrs[0]->req_size, 5);
  //SetDummyData(t_all_app_rrs[1]->h_stub->req, t_all_app_rrs[1]->req_size, 5);
  //printf("Before running inference...\n");
  //g_floatDump(t_all_app_rrs[0]->h_stub->resp, t_all_app_rrs[0]->resp_size);
  //g_floatDump(t_all_app_rrs[1]->h_stub->resp, t_all_app_rrs[1]->resp_size);
  //g_intDump(t_all_app_rrs[0]->h_stub->resp, t_all_app_rrs[0]->resp_size);
  //g_intDump(t_all_app_rrs[1]->h_stub->resp, t_all_app_rrs[1]->resp_size);
  //app_server->do_batch_work_sync(app_info->appIdx, 0, 2);
  //printf("After running inference...\n");
  //g_floatDump(t_all_app_rrs[0]->h_stub->resp, t_all_app_rrs[0]->resp_size);
  //g_floatDump(t_all_app_rrs[1]->h_stub->resp, t_all_app_rrs[1]->resp_size);
  //g_intDump(t_all_app_rrs[0]->h_stub->resp, t_all_app_rrs[0]->resp_size);
  //g_intDump(t_all_app_rrs[1]->h_stub->resp, t_all_app_rrs[1]->resp_size);

////////////////////////// Start server 
  
  printf("Starting workers...\n");
  unsigned int main_lcore_id = rte_lcore_id();
  // Skip 1 because of HT
  unsigned int rx_tx_core_id = rte_get_next_lcore(main_lcore_id, true, false);
  rx_tx_core_id = rte_get_next_lcore(rx_tx_core_id, true, false);
  if(rx_tx_core_id == RTE_MAX_LCORE) {
    rte_exit(EXIT_FAILURE, "Not enough LCORES\n");
  }
  rte_eal_remote_launch(rx_tx_loop, (void*)_dpdk_ctx, rx_tx_core_id);

  unsigned int shunter_core_id = rte_get_next_lcore(rx_tx_core_id, true, false);
  shunter_core_id = rte_get_next_lcore(shunter_core_id, true, false);
  if(shunter_core_id == RTE_MAX_LCORE) {
    rte_exit(EXIT_FAILURE, "Not enough LCORES\n");
  }
  rte_eal_remote_launch(shunter_loop, (void*)_dpdk_ctx, shunter_core_id);

  if(get_server_mode() == 0)
    app_server->sync_worker_loop(app_info->appIdx);
  else if(get_server_mode() == 1)
    app_server->async_worker_loop(app_info->appIdx);
  else if(get_server_mode() == 2)
    app_server->dynamic_batching_sync_worker_loop(app_info->appIdx, get_ort_batch_size());

//////////////////////// Cleanup

  rte_eal_wait_lcore(rx_tx_core_id);
  rte_eal_wait_lcore(shunter_core_id);

  printf("Stopping app...\n");
  delete app_server;
  printf("Cleaning up AppInfo\n");
  delete app_info;
  printf("Cleaning up P2pRpcRrPool\n");
  delete rpc_rr_pool;
  printf("Stopping dpdk...\n");
  stop_dpdk(_dpdk_ctx);
  delete _dpdk_ctx;
  printf("Cleaning up EAL...\n");
  rte_eal_cleanup();
  printf("Exiting app cleanly\n");
  return 0;
}
