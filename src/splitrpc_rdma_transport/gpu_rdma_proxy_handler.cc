// Parts of source taken from Animesh Trivedi's Github repo rdma-example and suitably modified
#include "utils/config_utils.h"

#include "gpu_rdma_common.h"
#include "p2p_rpc_bf_rr_pool.h"
//#include "udp_bench/udp_common.h"
#include "transport/udp_transport.h"
#include "udp_rr.h"
#include "eth_common.h"
#include "utils/debug_utils.h"
#include "utils/time_utils.h"

#include <vector>
#include <thread>

#include <signal.h>
volatile bool force_quit = 0;
static void 
signal_handler(int signum) {                                                                          
  if (signum == SIGINT) {        
      printf("\n\nSignal %d received, preparing to exit...\n", signum);      
      ACCESS_ONCE(force_quit) = 1;
    }
}

/* This is our testing function */
static int check_src_dst(char *src, char *dst, size_t buf_len) 
{
	return memcmp((void*) src, (void*) dst, buf_len);
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

int main(int argc, char **argv) {
  signal(SIGINT, signal_handler);
	struct sockaddr_in server_sockaddr;
	int ret, option;
	bzero(&server_sockaddr, sizeof server_sockaddr);
	server_sockaddr.sin_family = AF_INET;
	server_sockaddr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
	/* buffers are NULL */
	/* Parse Command Line Arguments */
  while ((option = getopt(argc, argv, "a:p:")) != -1) {
    switch (option) {
      case 'a':
        /* remember, this overwrites the port info */
        ret = get_addr(optarg, (struct sockaddr*) &server_sockaddr);
        if (ret) {
          rdma_error("Invalid IP \n");
          return ret;
        }
        break;
      case 'p':
        /* passed port to listen on */
        server_sockaddr.sin_port = htons(strtol(optarg, NULL, 0)); 
        break;
      default:
        usage();
        break;
    }
  }
  if (!server_sockaddr.sin_port) {
    /* no port provided, use the default port */
    server_sockaddr.sin_port = htons(DEFAULT_RDMA_PORT);
  }

//////////////// Initialize and setup the ctx 

  struct r_context *rctx = NULL;

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

  P2pRpcBfRrPool *bf_rr_pool = new P2pRpcBfRrPool(rctx, P2P_RPC_MAX_QUEUE_SIZE, buf_mon_msg);

////////////////////////////// BufMon Setup Ends

  struct sockaddr_in si_me;
  memset((char *)&si_me, 0, sizeof(si_me));
  si_me.sin_family = AF_INET;
  si_me.sin_port = htons(std::stoi(get_server_port()));
  if (inet_aton(get_server_ip().c_str(), &si_me.sin_addr) == 0) {
    std::cout<<"inet_aton() failed to parse src_ip_str\n";
    exit(1);
  }
  
  int clientUdpSock = initUdpSock(&si_me);
  printf("Creating UdpRrPool, with data on host, req_size: %ld, resp_size: %ld", buf_mon_msg.req_size, buf_mon_msg.resp_size);
  UdpRrPool *udp_rr_pool = new UdpRrPool(P2P_RPC_MAX_QUEUE_SIZE);
  udp_rr_pool->setup_and_init_rr_pool_with_preallocation((uint8_t*)bf_rr_pool->req_mr->addr, buf_mon_msg.req_size, (uint8_t*)bf_rr_pool->resp_mr->addr, buf_mon_msg.resp_size);
  UdpRr **all_rrs = udp_rr_pool->get_rr_pool();
  for(int i = 0 ; i < udp_rr_pool->get_pool_size(); i++) {
    all_rrs[i]->alloc_resp_bufs();
  }
  UdpRr *rx_rpc_rr = NULL, *tx_rpc_rr = NULL;

  struct req_mon_rr req_mon_resp;
  req_mon_resp.rr_idx = 1000;
  struct ibv_mr *req_mon_resp_mr = rdma_buffer_register(rctx->pd, &req_mon_resp, sizeof(req_mon_resp), 
      (ibv_access_flags)(IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ));

  printf("UDP listener started, Ready to send ReqMons, Clients can now be started...\n");

  // Pre-post to recv notification
  for(int i = 0 ; i < MAX_WR - 1; i++) {
    if(pre_post_recv_mr(rctx, req_mon_resp_mr)) {
      printf("Pre-posting response for reqmon failed\n");
      return 0;
    }
  }
  p2p_rpc_bf_wi *rx_bf_wi = bf_rr_pool->get_next();
  int bf_pi_idx = 0;
  int bf_ci_idx = 0;

  struct ibv_wc work_compl_wr;
#ifdef PROFILE_MODE
  uint64_t rxStartNs, txStartNs;
  std::vector<uint64_t> RxDelay, TxDelay, rttDelay;
#endif

  int resp_ci_idx = 0;

  std::thread t_rx(&udp_rr_recv_req_listener, clientUdpSock, udp_rr_pool, std::ref(force_quit));
  std::thread t_rx_processor(&udp_rr_recv_req_processor, udp_rr_pool, std::ref(force_quit));

  while(ACCESS_ONCE(force_quit) == 0) 
  {

///////////////////////////////// RX PATH

    if(((bf_pi_idx + 1 ) % bf_rr_pool->pool_size != bf_ci_idx) && 
        udp_rr_pool->get_next_rx_ready_rr(&rx_rpc_rr) != 0) {
#ifdef PROFILE_MODE
      rxStartNs = getCurNs();
#endif
      TRACE_PRINTF("RX: udp_rr: %p, udp_rr_idx: %ld, rx_bf_wi: %p, rx_bf_idx: %ld, src-ip %s, rr_token: %lu, req_size: %ld\n",
          (void*)rx_rpc_rr, rx_rpc_rr->rr_idx, (void*)rx_bf_wi, rx_bf_wi->idx,
          ipv4_to_string((uint32_t)rx_rpc_rr->si_other.sin_addr.s_addr).c_str(),
          rx_rpc_rr->req_token, rx_rpc_rr->req_size);
      rx_bf_wi->rpc_rr = (void*)rx_rpc_rr;

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
      TRACE_PRINTF("Write + NOTIF complete, Will wait for func_id: %d, rr_idx: %ld\n",
          req_mon_resp.func_id, rx_bf_wi->idx);
      rx_bf_wi->print_wi();

      rx_bf_wi = bf_rr_pool->get_next();
      bf_pi_idx = (bf_pi_idx + 1) % bf_rr_pool->pool_size;
#ifdef PROFILE_MODE
      RxDelay.push_back(getCurNs() - rxStartNs);
#endif
    }

/////////////////////////////////////// TX PATH

    // Wait for req_mon_resp
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
      tx_rpc_rr = (UdpRr*)tx_bf_wi->rpc_rr;

      // Copy the response back
      if(post_send_wr(rctx, tx_bf_wi->read_wr)) {
        rdma_error("Resp Read failed\n");
        return 0;
      }
      if(wait_on_cq(rctx->send_cq, 1)) {
        rdma_error("Failed when waiting for read to complete\n");
        return 0;
      }
      // Send data to client
      if(tx_rpc_rr->resp_to_bufs() != tx_rpc_rr->resp_size) {
        printf("TX gather error\n");
      }

      if(unlikely(udp_rr_send_resp(clientUdpSock, tx_rpc_rr) == 0))
        fprintf(stderr, "TX Error\n");
      udp_rr_pool->reap_rr(tx_rpc_rr);
      TRACE_PRINTF("TX Complete, rpc_rr: %p, rpc_rr_idx: %ld, bf_wi: %p, bf_ci_idx: %d\n",
          (void*)tx_rpc_rr, tx_rpc_rr->rr_idx, (void*)tx_bf_wi, bf_ci_idx);

      tx_bf_wi->rpc_rr = NULL;
      bf_ci_idx = (bf_ci_idx + 1) % bf_rr_pool->pool_size;
#ifdef PROFILE_MODE
      TxDelay.push_back(getCurNs() - txStartNs);
      rttDelay.push_back(getCurNs() - rxStartNs);
#endif
    }
  }

  t_rx.join();
  t_rx_processor.join();
  printf("Workload proxy terminating and cleaning up...\n");
  PROF_PRINT("TotalRx: ", RxDelay);
  PROF_PRINT("TotalTx: ", TxDelay);
  PROF_PRINT("rttDelay: ", rttDelay);

  delete bf_rr_pool;
  rdma_buffer_deregister(buf_mon_mr);
  rdma_buffer_deregister(req_mon_resp_mr);
  ret = client_disconnect(rctx);
	if (ret) {
		rdma_error("Failed to cleanly disconnect and clean up resources \n");
	}
  ret = r_context_cleanup(rctx);
	if (ret) {
		rdma_error("Failed to cleanly disconnect and clean up resources \n");
	}
	return ret;
}
