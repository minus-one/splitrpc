// Copyright (c) Adithya Kumar, The Pennsylvania State University. All rights reserved.
// Licensed under the MIT License.

// Portions of source have been modified from Animesh Trivedi's Github repo rdma-example

#include "config_utils.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <thread>

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

#include "p2p_rpc_app_rr.h"
#include "gpu_rdma_common.h"

#include <signal.h>
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

struct ibv_mr* rdma_gpu_buffer_alloc(struct ibv_pd *pd, uint32_t size,
    enum ibv_access_flags permission) 
{
	struct ibv_mr *mr = NULL;
	if (!pd) {
		rdma_error("Protection domain is NULL \n");
		return NULL;
	}
  void *d_addr;
  if (cudaMalloc((void **)&d_addr, size * sizeof(uint8_t)) != cudaSuccess) {
    printf("CudaMalloc failed, unable to create tmp_buffers\n");
    d_addr = NULL;
  }
  printf("Allocated cuda buffer: %p len: %d\n", d_addr, size);

	if (!d_addr) {
		rdma_error("failed to allocate buffer, -ENOMEM\n");
		return NULL;
	}
	debug("Buffer allocated: %p , len: %u \n", d_addr, size);
	mr = rdma_buffer_register(pd, d_addr, size, permission);
	if(!mr){
    cuMemFree((CUdeviceptr)d_addr);
	}
	return mr;
}

void rdma_gpu_buffer_free(struct ibv_mr *mr) 
{
	if (!mr) {
		rdma_error("Passed memory region is NULL, ignoring\n");
		return ;
	}
	void *to_free = mr->addr;
	rdma_buffer_deregister(mr);
	debug("Buffer %p free'ed\n", to_free);
  cuMemFree((CUdeviceptr)to_free);
}

/* Starts an RDMA server by allocating basic connection resources */
static int bind_rdma_server(struct r_context *rctx, struct sockaddr_in *server_addr) 
{
	struct rdma_cm_event *cm_event = NULL;
	int ret = -1;

	/* Explicit binding of rdma cm id to the socket credentials */
	ret = rdma_bind_addr(rctx->cm_id, (struct sockaddr*) server_addr);
	if (ret) {
		rdma_error("Failed to bind server address, errno: %d \n", -errno);
		return -errno;
	}
	debug("Server RDMA CM id is successfully binded \n");
	/* Now we start to listen on the passed IP and port. However unlike
	 * normal TCP listen, this is a non-blocking call. When a new client is 
	 * connected, a new connection management (CM) event is generated on the 
	 * RDMA CM event channel from where the listening id was created. Here we
	 * have only one channel, so it is easy. */
	ret = rdma_listen(rctx->cm_id, 8); /* backlog = 8 clients, same as TCP, see man listen*/
	if (ret) {
		rdma_error("rdma_listen failed to listen on server address, errno: %d ",
				-errno);
		return -errno;
	}
	printf("RDMA Server is listening successfully at: %s , port: %d \n",
			inet_ntoa(server_addr->sin_addr),
			ntohs(server_addr->sin_port));
/////////////////////////////////////////////////////////////////////////////////////////////////
	/* now, we expect a client to connect and generate a RDMA_CM_EVNET_CONNECT_REQUEST 
	 * We wait (block) on the connection management event channel for 
	 * the connect event. 
	 */
	ret = process_rdma_cm_event(rctx->cm_event_channel, 
			RDMA_CM_EVENT_CONNECT_REQUEST,
			&cm_event);
	if (ret) {
		rdma_error("Failed to get cm event, ret = %d \n" , ret);
		return ret;
	}
	/* Much like TCP connection, listening returns a new connection identifier 
	 * for newly connected client. In the case of RDMA, this is stored in id 
	 * field. For more details: man rdma_get_cm_event 
	 */
  rctx->cm_server_id = rctx->cm_id;
	rctx->cm_id = cm_event->id;
  rctx->ctx = rctx->cm_id->verbs;
	/* now we acknowledge the event. Acknowledging the event free the resources 
	 * associated with the event structure. Hence any reference to the event 
	 * must be made before acknowledgment. Like, we have already saved the 
	 * client id from "id" field before acknowledging the event. 
	 */
	ret = rdma_ack_cm_event(cm_event);
	if (ret) {
		rdma_error("Failed to acknowledge the cm event errno: %d \n", -errno);
		return -errno;
	}
	return ret;
}

static int accept_client_call(struct r_context *rctx)
{
	struct rdma_cm_event *cm_event = NULL;
	struct sockaddr_in remote_sockaddr; 
  int ret = -1;

  struct rdma_conn_param conn_param;
  memset(&conn_param, 0, sizeof(conn_param));
  conn_param.initiator_depth = 3; 
  conn_param.responder_resources = 3; 
  conn_param.rnr_retry_count = 3;
  ret = rdma_accept(rctx->cm_id, &conn_param);
  if (ret) {
    rdma_error("Failed to accept the connection, errno: %d \n", -errno);
    return -errno;
  }
  printf("Going to wait for : RDMA_CM_EVENT_ESTABLISHED event \n");
  ret = process_rdma_cm_event(rctx->cm_event_channel, 
      RDMA_CM_EVENT_ESTABLISHED,
      &cm_event);
  if (ret) {
    rdma_error("Failed to get the cm event, errnp: %d \n", -errno);
    return -errno;
  }
	ret = rdma_ack_cm_event(cm_event);
	if (ret) {
		rdma_error("Failed to acknowledge the cm event %d\n", -errno);
		return -errno;
	}

	/* Just FYI: How to extract connection information */
	memcpy(&remote_sockaddr /* where to save */, 
			rdma_get_peer_addr(rctx->cm_id) /* gives you remote sockaddr */, 
			sizeof(struct sockaddr_in) /* max size */);
	printf("A new connection is accepted from %s \n", 
			inet_ntoa(remote_sockaddr.sin_addr));

  return 0;
}

/* Pre-posts a receive buffer and accepts an RDMA client connection */
static int accept_client_connection(struct r_context *rctx, struct ibv_mr *client_metadata_mr)
{
	int ret = -1;
	if(!rctx->cm_id || !rctx->qp) {
		rdma_error("Client resources are not properly setup\n");
		return -EINVAL;
	}
  ret = pre_post_recv_mr(rctx, client_metadata_mr);
  ret = accept_client_call(rctx);
  struct ibv_wc wc;
	ret = process_work_completion_events(rctx->comp_channel, &wc, 1);
	if (ret != 1) {
		rdma_error("Failed to receive , ret = %d \n", ret);
		return ret;
	}
	/* if all good, then we should have client's buffer information, lets see */
	printf("Client side buffer information is received...\n");
	return 0;
}

/* This is server side logic. Server passively waits for the client to call 
 * rdma_disconnect() and then it will clean up its resources */
static int disconnect(struct r_context *rctx)
{
	struct rdma_cm_event *cm_event = NULL;
	int ret = -1;
       /* Now we wait for the client to send us disconnect event */
       debug("Waiting for cm event: RDMA_CM_EVENT_DISCONNECTED\n");
       ret = process_rdma_cm_event(rctx->cm_event_channel, 
		       RDMA_CM_EVENT_DISCONNECTED, 
		       &cm_event);
       if (ret) {
	       rdma_error("Failed to get disconnect event, ret = %d \n", ret);
	       return ret;
       }
	/* We acknowledge the event */
	ret = rdma_ack_cm_event(cm_event);
	if (ret) {
		rdma_error("Failed to acknowledge the cm event %d\n", -errno);
		return -errno;
	}
	printf("A disconnect event is received from the client...\n");
	return 0;
}

void usage() 
{
	printf("Usage:\n");
	printf("rdma_server: [-a <server_addr>] [-p <server_port>]\n");
	printf("(default port is %d)\n", DEFAULT_RDMA_PORT);
	exit(1);
}

int main(int argc, char **argv) 
{
  signal(SIGINT, signal_handler);
	int ret, option;
	struct sockaddr_in server_sockaddr;
	bzero(&server_sockaddr, sizeof server_sockaddr);
	server_sockaddr.sin_family = AF_INET; /* standard IP NET address */
	server_sockaddr.sin_addr.s_addr = htonl(INADDR_ANY); /* passed address */
	/* Parse Command Line Arguments, not the most reliable code */
  while ((option = getopt(argc, argv, "a:p:")) != -1) {
    switch (option) {
      case 'a':
        /* Remember, this will overwrite the port info */
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
  if(!server_sockaddr.sin_port) {
    /* If still zero, that mean no port info provided */
    server_sockaddr.sin_port = htons(DEFAULT_RDMA_PORT); /* use default port */
  }

  struct r_context *rctx;
  if(r_context_init(&rctx)) {
    rdma_error("Failed to init context\n");
    return 0;
  }
  // Bind returns once a connection request has arrived
  if(bind_rdma_server(rctx, &server_sockaddr)) {
    rdma_error("Failed to bind\n");
    return 0;
  }
  printf("Got a connection request rctx: %p with ibv_context: %p\n", (void*)rctx, (void*)rctx->ctx);
  if(setup_ctx_resources(rctx)) {
    rdma_error("Failed to setup ctx resources\n");
    return 0;
  }

//////////////////////////////// BUF-MON START 
  struct buf_mon_rr buf_mon_req;
  struct ibv_mr *buf_mon_mr = rdma_buffer_register(rctx->pd, &buf_mon_req, sizeof(buf_mon_req), IBV_ACCESS_LOCAL_WRITE);
  if(!buf_mon_mr) {
    rdma_error("Failed to register mr\n");
    return 0;
  }
  ret = pre_post_recv_mr(rctx, buf_mon_mr); 
  ret = accept_client_call(rctx);

  if(wait_on_cq(rctx->recv_cq, 1)) { 
  //if(wait_for_comp(rctx, 1)) {
    rdma_error("Failed to get a cq notif\n");
    return 0;
  }

  /* if all good, then we should have client's buffer allocation request, lets see */
  printf("BufMon request information is received...\n");
  printf("Type: %d, Function_id: %d\n", 
      buf_mon_req.type, buf_mon_req.func_id);

  struct ibv_mr *req_buf_mr = NULL, *resp_buf_mr = NULL, *state_buf_mr = NULL;

  // server starts setting all the necessary details
  buf_mon_req.queue_size = P2P_RPC_MAX_QUEUE_SIZE;
  buf_mon_req.req_size = get_req_size();
  buf_mon_req.resp_size = get_resp_size();
  P2pRpcAppRrPool *app_rr_pool = NULL;
  size_t app_rr_pool_size = P2P_RPC_MAX_QUEUE_SIZE;

  if(buf_mon_req.type == buf_mon_rr::MSG_ALLOC) {
    TRACE_PRINTF("Setting up P2pRpcAppInfo\n");
    app_info = new P2pRpcAppInfo(AppInit_cb, AppRun_cb, AppCleanup_cb, AppComplete_cb, 
        get_cuda_device_id(), get_req_size(), get_resp_size(), P2P_RPC_MAX_QUEUE_SIZE);
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
    app_rr_pool = app_info->app_rr_pool; 

    req_buf_mr = rdma_buffer_register(rctx->pd, app_rr_pool->rr_mem_pool->get_req_addr_range(), 
        app_rr_pool->rr_mem_pool->get_req_addr_pool_size(),
        (ibv_access_flags)(IBV_ACCESS_LOCAL_WRITE|IBV_ACCESS_REMOTE_READ|IBV_ACCESS_REMOTE_WRITE));
    resp_buf_mr = rdma_buffer_register(rctx->pd, app_rr_pool->rr_mem_pool->get_resp_addr_range(), 
        app_rr_pool->rr_mem_pool->get_resp_addr_pool_size(),
        (ibv_access_flags)(IBV_ACCESS_LOCAL_WRITE|IBV_ACCESS_REMOTE_READ|IBV_ACCESS_REMOTE_WRITE));
    state_buf_mr = rdma_buffer_register(rctx->pd, app_rr_pool->rr_mem_pool->get_state_addr_range(), 
        P2P_RPC_MAX_QUEUE_SIZE * sizeof(uint32_t),
        (ibv_access_flags)(IBV_ACCESS_LOCAL_WRITE|IBV_ACCESS_REMOTE_READ|IBV_ACCESS_REMOTE_WRITE));

    if(!req_buf_mr || !resp_buf_mr || !state_buf_mr) {
      rdma_error("Error in registering mr for the APP_RR\n");
      return -ENOMEM;
    }

    buf_mon_req.req_buf_attr.address = (uint64_t) req_buf_mr->addr;
    buf_mon_req.req_buf_attr.length = (uint32_t) app_rr_pool->rr_mem_pool->get_req_addr_pool_size();
    buf_mon_req.req_buf_attr.stag.local_stag = (uint32_t) req_buf_mr->lkey;
    buf_mon_req.resp_buf_attr.address = (uint64_t) resp_buf_mr->addr;
    buf_mon_req.resp_buf_attr.length = (uint32_t) app_rr_pool->rr_mem_pool->get_resp_addr_pool_size();
    buf_mon_req.resp_buf_attr.stag.local_stag = (uint32_t) resp_buf_mr->lkey;
    buf_mon_req.state_buf_attr.address = (uint64_t) state_buf_mr->addr;
    buf_mon_req.state_buf_attr.length = (uint32_t) state_buf_mr->length;
    buf_mon_req.state_buf_attr.stag.local_stag = (uint32_t) state_buf_mr->lkey;

    printf("Req Attributes: \n");
    show_rdma_buffer_attr(&buf_mon_req.req_buf_attr);
    printf("Resp Attributes: \n");
    show_rdma_buffer_attr(&buf_mon_req.resp_buf_attr);
    printf("State Attributes: \n");
    show_rdma_buffer_attr(&buf_mon_req.state_buf_attr);
  } 
/*** Unimplementing this part for now ****************************** 
  else if(buf_mon_req.type == buf_mon_rr::MSG_RELEASE) {
    // FIXME:Actually look up the func-id and get the MRs and then release
	  rdma_buffer_deregister(req_buf_mr);
	  rdma_buffer_deregister(resp_buf_mr);
	  rdma_buffer_deregister(state_buf_mr);
    buf_mon_req.req_buf_attr.address = (uint64_t) 0;
    buf_mon_req.req_buf_attr.length = (uint32_t) 0;
    buf_mon_req.req_buf_attr.stag.local_stag = (uint32_t) 0;
    buf_mon_req.resp_buf_attr.address = (uint64_t) 0;
    buf_mon_req.resp_buf_attr.length = (uint32_t) 0;
    buf_mon_req.resp_buf_attr.stag.local_stag = (uint32_t) 0;
    buf_mon_req.state_buf_attr.address = (uint64_t) 0;
    buf_mon_req.state_buf_attr.length = (uint32_t) 0;
    buf_mon_req.state_buf_attr.stag.local_stag = (uint32_t) 0;
  }
*********************************************************************/

  // Send the buf_mon_resp back to client
  ret = post_send_mr_sync(rctx, buf_mon_mr);
//////////////////////////////// BUF-MON End

  struct req_mon_rr req_mon_req;
  req_mon_req.func_id = 1234;
  req_mon_req.type = req_mon_rr::MSG_NOTIFY_COMPL;

  struct ibv_mr *req_mon_mr = rdma_buffer_register(rctx->pd, &req_mon_req, sizeof(req_mon_req), IBV_ACCESS_LOCAL_WRITE);
  if(!req_mon_mr) {
    rdma_error("Failed to register mr\n");
    return 0;
  }

  int max_batch_size = get_ort_batch_size();

  // Pre-create the AppRrPool items so that the client can notify them when ready
  for(int16_t i = 0 ; i < max_batch_size; i++) {
    app_rr_pool->get_next();
  }

  // Do a warm-up run so that initializations are all done
  for(int i = 1 ; i <= max_batch_size; i++) {
      app_server->do_batch_work_sync(app_info->appIdx, 0, i);
  }

  std::thread t_worker;
  int server_mode = get_server_mode();

  //if(server_mode == 0)
    //t_worker = std::thread(&rdma_sync_worker_loop, app_info, app_server);
    //t_worker = std::thread(&P2pRpcAsyncAppServer::sync_worker_loop, std::ref(*app_server), app_info->appIdx);
  if(server_mode == 1)
    t_worker = std::thread(&P2pRpcAsyncAppServer::async_worker_loop, std::ref(*app_server), app_info->appIdx);
  else if(server_mode == 2)
    t_worker = std::thread(&P2pRpcAsyncAppServer::dynamic_batching_sync_worker_loop, std::ref(*app_server), app_info->appIdx, max_batch_size);

  printf("RPC Server is ready to service requests....\n");

  if(server_mode == 0) {
    int rr_ci_idx = 0;
    int num_req = 0;
    // Worker Loop
    while(ACCESS_ONCE(force_quit) == 0) 
    {
      if(*ACCESS_ONCE(app_rr_pool->get_app_rr(rr_ci_idx)->h_state) == APP_RR_STATUS::RX_COMPLETE) {
        TRACE_PRINTF("Got RX_COMPLETE AppCtx: %p, rr_ci_idx: %d, AppRr: %p\n", 
            (void*)app_ctx, rr_ci_idx, (void*)app_rr_pool->get_app_rr(rr_ci_idx));
        app_server->do_work_sync(app_info->appIdx, rr_ci_idx);
        num_req++;
        req_mon_req.rr_idx = rr_ci_idx;
        if(post_send_imm_with_inline_mr(rctx, req_mon_mr, rr_ci_idx) != 0)
          fprintf(stderr, "Failed to post reply...idx: %d\n", rr_ci_idx);
        if(wait_on_cq(rctx->send_cq, 1) != 0)
          fprintf(stderr, "Failed to waiting for post reply complete...idx: %d\n", rr_ci_idx);

        //FIXME: This should not be here
        app_rr_pool->get_next();
        *ACCESS_ONCE(app_rr_pool->get_app_rr(rr_ci_idx)->h_state) = APP_RR_STATUS::TX_COMPLETE;
        rr_ci_idx = (rr_ci_idx + 1) % app_rr_pool_size;
      }
    }
    printf("Exited worker + forwarder loop, rr_ci_idx: %d, num_req: %d\n", rr_ci_idx, num_req);
  }

  if(server_mode == 2) {
    int rr_ci_idx = 0;
    int num_req = 0;
    // Worker Loop
    while(ACCESS_ONCE(force_quit) == 0) {
      if(*ACCESS_ONCE(app_rr_pool->get_app_rr(rr_ci_idx)->h_state) == APP_RR_STATUS::WORK_COMPLETE) {
        num_req++;
        req_mon_req.rr_idx = rr_ci_idx;
        if(post_send_imm_with_inline_mr(rctx, req_mon_mr, rr_ci_idx) != 0)
          fprintf(stderr, "Failed to post reply...idx: %d\n", rr_ci_idx);
        if(wait_on_cq(rctx->send_cq, 1) != 0)
          fprintf(stderr, "Failed to waiting for post reply complete...idx: %d\n", rr_ci_idx);

        //FIXME: This should not be here
        app_rr_pool->get_next();
        *ACCESS_ONCE(app_rr_pool->get_app_rr(rr_ci_idx)->h_state) = APP_RR_STATUS::TX_COMPLETE;
        rr_ci_idx = (rr_ci_idx + 1) % app_rr_pool_size;
      }
    }
    printf("Exited forwarder loop, rr_ci_idx: %d, num_req: %d\n", rr_ci_idx, num_req);
    t_worker.join();
  }

  ret = disconnect(rctx);
  if (ret) { 
    rdma_error("Failed to disconnect resources properly, ret = %d \n", ret);
    return ret;
  }
  /* Destroy memory buffers */
  rdma_buffer_deregister(buf_mon_mr);
  rdma_buffer_deregister(req_buf_mr);
  rdma_buffer_deregister(resp_buf_mr);
  rdma_buffer_deregister(state_buf_mr);
  rdma_buffer_deregister(req_mon_mr);
  ret = r_context_cleanup(rctx);
  if (ret) {
    rdma_error("Failed to clean up resources \n");
  }

  printf("Cleaning up Server\n");
  delete app_server;
  printf("Cleaning up AppInfo\n");
  delete app_info;

  printf("App exiting cleanly...\n");
  return 0;
}
