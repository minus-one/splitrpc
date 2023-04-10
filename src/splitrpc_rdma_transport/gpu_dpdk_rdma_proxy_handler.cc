// Parts of source taken from Animesh Trivedi's Github repo rdma-example and suitably modified
#include <vector>
//#include <thread>

#include "utils/config_utils.h"
#include "gpu_rdma_common.h"
#include "p2p_rpc_bf_rr_pool.h"
#include "utils/debug_utils.h"
#include "utils/time_utils.h"

// Transport related
#include "transport/dpdk_init.h"
#include "transport/dpdk_transport_ng.h"
#include "p2p_rpc_rr_ng.h"
#include "p2p_rpc_rr_pool_ng.h"
P2pRpcRRPool *rpc_rr_pool = NULL;
struct p2p_rpc_conn_info *server_conn_info = NULL;
#include "p2p_rpc_sg_engine.h"
P2pRpcSgEngine *copy_engine = NULL;

#include <signal.h>
volatile bool force_quit = 0;
static void 
signal_handler(int signum) {                                                                          
  if (signum == SIGINT) {        
      printf("\n\nSignal %d received, preparing to exit...\n", signum);      
      ACCESS_ONCE(force_quit) = 1;
    }
}

static int client_connect(struct r_context *rctx)
{
  struct rdma_cm_event *cm_event = NULL;
  int ret = -1;
///////////////////////////////////////////// Connect to Server
	struct rdma_conn_param conn_param;
	ret = -1;
	bzero(&conn_param, sizeof(conn_param));
	conn_param.initiator_depth = 3; // FIXME
	conn_param.responder_resources = 3; // FIXME
	conn_param.retry_count = 3; // if fail, then how many times to retry
  conn_param.rnr_retry_count = 3;
	ret = rdma_connect(rctx->cm_id, &conn_param);
	if (ret) {
		rdma_error("Failed to connect to remote host , errno: %d\n", -errno);
		return -errno;
	}
	ret = process_rdma_cm_event(rctx->cm_event_channel, 
			RDMA_CM_EVENT_ESTABLISHED,
			&cm_event);
	if (ret) {
		rdma_error("Failed to get cm event, ret = %d \n", ret);
	       return ret;
	}
	ret = rdma_ack_cm_event(cm_event);
	if (ret) {
		rdma_error("Failed to acknowledge cm event, errno: %d\n", 
			       -errno);
		return -errno;
	}
	printf("The client is connected successfully \n");
  return 0;
}

static int exchange_bufs_with_server(struct r_context *rctx, struct ibv_mr *client_metadata_mr, struct ibv_mr *server_metadata_mr)
{
/////////////////////////////////////////// PRE-POST Recv Buffer
  if(pre_post_recv_mr(rctx, server_metadata_mr)) {
    printf("Pre-Post Recv failed\n");
    return -1;
  }
////////////////////////////////////////////// Exchange metadata with server
  if(post_send_mr(rctx, client_metadata_mr)) {
    printf("Post send failed\n");
    return -1;
  }
////////////////////////////////////////////////// Wait for completions
  if(wait_on_cq(rctx->send_cq, 1)) {
    rdma_error("Failed when waiting to complete sending the request to server\n");
    return -1;
  }
  if(wait_on_cq(rctx->recv_cq, 1)) {
    rdma_error("Failed when waiting to receive the buf info from server\n");
    return -1;
  }

  return 0;
}

/* This function disconnects the RDMA connection from the server and cleans up 
 * all the resources.
 */
static int client_disconnect(struct r_context *rctx)
{
	struct rdma_cm_event *cm_event = NULL;
	int ret = -1;
	/* active disconnect from the client side */
	ret = rdma_disconnect(rctx->cm_id);
	if (ret) {
		rdma_error("Failed to disconnect, errno: %d \n", -errno);
		//continuing anyways
	}
	ret = process_rdma_cm_event(rctx->cm_event_channel, 
			RDMA_CM_EVENT_DISCONNECTED,
			&cm_event);
	if (ret) {
		rdma_error("Failed to get RDMA_CM_EVENT_DISCONNECTED event, ret = %d\n",
				ret);
		//continuing anyways 
	}
	ret = rdma_ack_cm_event(cm_event);
	if (ret) {
		rdma_error("Failed to acknowledge cm event, errno: %d\n", 
			       -errno);
		//continuing anyways
	}
  printf("Client disconnected...\n");
  return 0;
}

void usage() {
	printf("Usage:\n");
	printf("rdma_client: [-a <server_addr>] [-p <server_port>] -s string (required)\n");
	printf("(default IP is 127.0.0.1 and port is %d)\n", DEFAULT_RDMA_PORT);
	exit(1);
}

#ifdef PROFILE_MODE
uint64_t rxStartNs, txStartNs;
uint64_t copyInStartNs, copyOutStartNs;
std::vector<uint64_t> RxDelay, TxDelay, CopyInDelay, CopyOutDelay;
#endif

struct r_context *rctx = NULL;
P2pRpcBfRrPool *bf_rr_pool = NULL;
int bf_pi_idx = 0;
int bf_ci_idx = 0;

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
    copy_engine = new P2pRpcSgCpuEngine();
  } 

#ifdef PROFILE_MODE
  uint64_t rxStartNs, txStartNs;
  std::vector<uint64_t> RxDelay, TxDelay;
#endif

  P2pRpcRr *new_rpc_rr = NULL, *tx_rpc_rr = NULL;
  uint32_t num_rx = 0, num_tx = 0;

  p2p_rpc_bf_wi *rx_bf_wi = bf_rr_pool->get_next();

  struct req_mon_rr req_mon_resp;
  req_mon_resp.rr_idx = 1000;
  struct ibv_mr *req_mon_resp_mr = rdma_buffer_register(rctx->pd, &req_mon_resp, sizeof(req_mon_resp), 
      (ibv_access_flags)(IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ));
  // Pre-post to recv notification
  for(int i = 0 ; i < MAX_WR - 1; i++) {
    if(pre_post_recv_mr(rctx, req_mon_resp_mr)) {
      printf("Pre-posting response for reqmon failed\n");
      return 0;
    }
  }
  struct ibv_wc work_compl_wr;
  int resp_ci_idx = 0;
 
  while(ACCESS_ONCE(force_quit) == 0) 
  {
    if(((bf_pi_idx + 1) % bf_rr_pool->pool_size != bf_ci_idx) &&
        rpc_rr_pool->get_next_rx_ready_rr(&new_rpc_rr) != 0) {
      TRACE_PRINTF("RX: rpc_rr: %p, rx_bf_wi: %p src-ip %s, dst-ip %s, rr_token: %lu, req_size: %ld\n",
          (void*)new_rpc_rr, (void*)rx_bf_wi, ipv4_to_string(get_ip_header(&(new_rpc_rr->_client_conn_info.hdr_template))->src_ip).c_str(),
          ipv4_to_string(get_ip_header(&(new_rpc_rr->_client_conn_info.hdr_template))->dst_ip).c_str(),
          new_rpc_rr->req_token, new_rpc_rr->payload_size);

#ifdef PROFILE_MODE
      rxStartNs = getCurNs();
#endif
      num_rx++;
      copy_engine->gather(new_rpc_rr, rx_bf_wi->local_req_stub);
      rx_bf_wi->rpc_rr = (void*)new_rpc_rr;

      // WRITE the request payload and NOTIF
      if(post_send_wr(rctx, rx_bf_wi->write_wr)) {
        printf("Req Write failed\n");
        return 0;
      }
      // SYNC
      if(wait_on_cq(rctx->send_cq, 2)) {
        fprintf(stderr, "Failed when waiting for write + NOTIFY to complete\n");
        return 0;
      }
      //*ACCESS_ONCE(new_app_rr->h_state) = APP_RR_STATUS::RX_COMPLETE;

      if(_dpdk_ctx->zerocopy_mode == 0) {
        rr_release_mbufs(_dpdk_ctx, new_rpc_rr);
        rr_alloc_mbufs(_dpdk_ctx, new_rpc_rr, bf_rr_pool->resp_size);
        rr_set_hdr(&new_rpc_rr->_client_conn_info, new_rpc_rr, bf_rr_pool->resp_size);
      } else { 
        rr_swap_hdr(_dpdk_ctx, new_rpc_rr, bf_rr_pool->resp_size);
      }
      TRACE_PRINTF("Write + NOTIF complete, Will wait for rr_idx: %ld\n", rx_bf_wi->idx);
      rx_bf_wi->print_wi();
      rx_bf_wi = bf_rr_pool->get_next();
      bf_pi_idx = (bf_pi_idx + 1) % bf_rr_pool->pool_size;

#ifdef PROFILE_MODE
      RxDelay.push_back(getCurNs() - rxStartNs);
#endif
    }

    // Check for completions 
    if(poll_on_cq_and_get_imm_data(rctx->recv_cq, &work_compl_wr, resp_ci_idx)) {
#ifdef PROFILE_MODE
      txStartNs = getCurNs();
#endif
      if(pre_post_recv_mr(rctx, req_mon_resp_mr)) {
        printf("Pre-posting response for reqmon failed\n");
        return 0;
      }
      if(bf_ci_idx != resp_ci_idx)
        fprintf(stderr, "Received response for an unexpected idx, exp: %d, got: %d", bf_ci_idx, resp_ci_idx);
      p2p_rpc_bf_wi *tx_bf_wi = bf_rr_pool->get_bf_wi(bf_ci_idx); 

#ifdef PROFILE_MODE
        copyOutStartNs = getCurNs();
#endif
      // Copy the response back
      if(post_send_wr(rctx, tx_bf_wi->read_wr)) {
        rdma_error("Resp Read failed\n");
        return 0;
      }
      if(wait_on_cq(rctx->send_cq, 1)) {
        rdma_error("Failed when waiting for read to complete\n");
        return 0;
      }

      tx_rpc_rr = (P2pRpcRr*)tx_bf_wi->rpc_rr;
      TRACE_PRINTF("TX RPC_RR: %p, my-ip %s, sender-ip %s, rr_token: %lu, resp_size: %ld\n",
          (void*)tx_rpc_rr, ipv4_to_string(get_ip_header(&tx_rpc_rr->_client_conn_info.hdr_template)->src_ip).c_str(),
          ipv4_to_string(get_ip_header(&tx_rpc_rr->_client_conn_info.hdr_template)->dst_ip).c_str(),
          tx_rpc_rr->req_token, tx_rpc_rr->payload_size);

      num_tx++;
      if(_dpdk_ctx->zerocopy_mode == 0) {
        copy_engine->scatter(tx_rpc_rr, tx_bf_wi->local_resp_stub);
      } 
#ifdef PROFILE_MODE
      CopyOutDelay.push_back(getCurNs() - copyOutStartNs);
#endif
      //rpc_rr_pool->mark_tx_ready_rr(tx_rpc_rr);
      if (unlikely(rr_send(_dpdk_ctx, tx_rpc_rr, *rpc_rr_pool) == 0))
        printf("TX Error RPC server\n");
      TRACE_PRINTF("TX Complete rr_ci_idx: %d\n", rr_ci_idx);
      tx_bf_wi->rpc_rr = NULL;
      //*ACCESS_ONCE(tx_bf_wi->h_state) = APP_RR_STATUS::FREE;
      //_mm_mfence();

      tx_bf_wi->rpc_rr = NULL;
      bf_ci_idx = (bf_ci_idx + 1) % bf_rr_pool->pool_size;
#ifdef PROFILE_MODE
      TxDelay.push_back(getCurNs() - txStartNs);
#endif
    }
  }
  printf("Ending shunter loop thread copy_in: %d, copy_out: %d\n", num_rx, num_tx);
  printf("Stopping CopyEngine...\n");
  PROF_PRINT("CopyIn: ", RxDelay);
  PROF_PRINT("CopyOut: ", TxDelay);
  delete copy_engine;
  rdma_buffer_deregister(req_mon_resp_mr);
  return 1;
}

int main(int argc, char **argv) {
  signal(SIGINT, signal_handler);
	struct sockaddr_in server_sockaddr;
	int ret;
	bzero(&server_sockaddr, sizeof server_sockaddr);
	server_sockaddr.sin_family = AF_INET;
	server_sockaddr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  ret = get_addr("192.168.25.1", (struct sockaddr*) &server_sockaddr);
  if (ret) {
    rdma_error("Invalid IP \n");
    return ret;
  }
  server_sockaddr.sin_port = htons(DEFAULT_RDMA_PORT);

//////////////// Initialize and setup the ctx 

  if(r_context_init(&rctx)) {
    rdma_error("Failed to init context\n");
    return 0;
  }
  if(resolv_ctx(rctx, &server_sockaddr)) {
    rdma_error("Failed to resolv the server, is the network down?\n");
    return 0;
  }

  printf("Created new rctx: %p with ibv_context: %p\n", (void*)rctx, (void*)rctx->ctx);
  if(setup_ctx_resources(rctx)) {
    rdma_error("Failed to setup ctx resources\n");
    return 0;
  }

////////////////////// Make a BufMon request and setup buffers
  
  struct buf_mon_rr buf_mon_msg;
  struct ibv_mr *buf_mon_mr = rdma_buffer_register(rctx->pd, &buf_mon_msg, sizeof(buf_mon_msg), 
      (ibv_access_flags)(IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ));
  if(!buf_mon_mr) {
    rdma_error("Failed to register mr\n");
    return 0;
  }
  buf_mon_msg.type = buf_mon_rr::MSG_ALLOC;
  // We only set the function-id. The server
  // will actually set the req_size and resp_size
  buf_mon_msg.func_id = 1234;

  printf("Exchanging BufMon info with: type: %d, func_id: %d\n",
      buf_mon_msg.type, buf_mon_msg.func_id);

  if(client_connect(rctx)) {
    printf("Failed to connect to server\n");
    return 0;
  }
  if(exchange_bufs_with_server(rctx, buf_mon_mr, buf_mon_mr)) {
    printf("Failed to get the BufMon details\n");
    return 0;
  }

  // At this stage we should have got the BufMon details
  printf("BufMon exchange complete...\n");
  printf("Received BufMon info with: type: %d, func_id: %d, req_size: %ld, resp_size: %ld\n",
      buf_mon_msg.type, buf_mon_msg.func_id, buf_mon_msg.req_size, buf_mon_msg.resp_size);

  printf("Req Attributes: \n");
  show_rdma_buffer_attr(&buf_mon_msg.req_buf_attr);
  printf("Resp Attributes: \n");
  show_rdma_buffer_attr(&buf_mon_msg.resp_buf_attr);
  printf("State Attributes: \n");
  show_rdma_buffer_attr(&buf_mon_msg.state_buf_attr);

////////////////////////////// BufMon Rendezvous complete

  bf_rr_pool = new P2pRpcBfRrPool(rctx, P2P_RPC_MAX_QUEUE_SIZE, buf_mon_msg);

////////////////////////////// BufMon Setup Ends

  ret = rte_eal_init(argc, argv);
  if (ret < 0)
    rte_exit(EXIT_FAILURE, "Error with EAL initialization\n");
  argc -= ret;
  argv += ret;
  struct dpdk_ctx *_dpdk_ctx = new dpdk_ctx;
  _dpdk_ctx->nic_port = get_dpdk_port();
  _dpdk_ctx->queue_id = 0;
  _dpdk_ctx->mem_alloc_type = get_dpdk_mem_alloc_type();
  _dpdk_ctx->device_id = -2;
  _dpdk_ctx->zerocopy_mode = is_zerocopy_mode(); 
  if(init_dpdk_ctx(_dpdk_ctx) == 0) {
    printf("Failed to init dpdk\n");
    exit(1);
  }
  std::string src_uri = get_server_mac() + std::string(",") 
    + get_server_ip() + std::string(",") + get_server_port();

  server_conn_info = initialize_src_conn_info(src_uri);
  TRACE_PRINTF("Setting up P2pRpcRRPool with src_uri: %s, server_conn_info: %p\n", src_uri.c_str(), server_conn_info);
  rpc_rr_pool = new P2pRpcRRPool(P2P_RPC_MAX_QUEUE_SIZE);
  rpc_rr_pool->setup_and_init_rr_pool(server_conn_info, buf_mon_msg.req_size); 

  printf("Starting DPDK workers...\n");
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

  if(rte_eal_wait_lcore(rx_tx_core_id) < 0) {
    printf("Failed while joining the rx-tx-core\n");
    ret = -1;
  }
  if(rte_eal_wait_lcore(shunter_core_id) < 0) {
    printf("Failed while joining the shunter core\n");
    ret = -1;
  }

  printf("Workload proxy terminating and cleaning up...\n");

  printf("Cleaning up P2pRpcRrPool\n");
  delete rpc_rr_pool;
  printf("Stopping dpdk...\n");
  stop_dpdk(_dpdk_ctx);
  delete _dpdk_ctx;
  printf("Cleaning up EAL...\n");
  if(rte_eal_cleanup() != 0) {
    printf("Error in cleaning up the EAL\n");
  }

  delete bf_rr_pool;
  rdma_buffer_deregister(buf_mon_mr);
  ret = client_disconnect(rctx);
	if (ret) {
		rdma_error("Failed to cleanly disconnect and clean up resources \n");
	}
  ret = r_context_cleanup(rctx);
	if (ret) {
		rdma_error("Failed to cleanly disconnect and clean up resources \n");
	}

  PROF_PRINT("TotalRx: ", RxDelay);
  PROF_PRINT("TotalTx: ", TxDelay);
  PROF_PRINT("CopyInDelay: ", CopyInDelay);
  PROF_PRINT("CopyOutDelay: ", CopyOutDelay);
  printf("Exiting proxy cleanly...\n");

	return ret;
}
