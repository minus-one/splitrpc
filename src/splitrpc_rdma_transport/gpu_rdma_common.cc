// Source heavily lifted from Animesh Trivedi's Github repo rdma-example
#include "gpu_rdma_common.h"

void show_rdma_cmid(struct rdma_cm_id *id)
{
	if(!id){
		rdma_error("Passed ptr is NULL\n");
		return;
	}
	printf("RDMA cm id at %p \n", id);
	if(id->verbs && id->verbs->device)
		printf("dev_ctx: %p (device name: %s) \n", id->verbs, 
				id->verbs->device->name);
	if(id->channel)
		printf("cm event channel %p\n", id->channel);
	printf("QP: %p, port_space %x, port_num %u \n", id->qp, 
			id->ps,
			id->port_num);
}

void show_rdma_buffer_attr(struct rdma_buffer_attr *attr){
	if(!attr){
		rdma_error("Passed attr is NULL\n");
		return;
	}
	printf("---------------------------------------------------------\n");
	printf("buffer attr, addr: %p , len: %u , stag : 0x%x \n", 
			(void*) attr->address, 
			(unsigned int) attr->length,
			attr->stag.local_stag);
	printf("---------------------------------------------------------\n");
}

struct ibv_mr* rdma_buffer_alloc(struct ibv_pd *pd, uint32_t size,
    enum ibv_access_flags permission) 
{
	struct ibv_mr *mr = NULL;
	if (!pd) {
		rdma_error("Protection domain is NULL \n");
		return NULL;
	}
	void *buf = calloc(1, size);
	if (!buf) {
		rdma_error("failed to allocate buffer, -ENOMEM\n");
		return NULL;
	}
	debug("Buffer allocated: %p , len: %u \n", buf, size);
	mr = rdma_buffer_register(pd, buf, size, permission);
	if(!mr){
		free(buf);
	}
	return mr;
}

struct ibv_mr *rdma_buffer_register(struct ibv_pd *pd, 
		void *addr, uint32_t length, 
		enum ibv_access_flags permission)
{
	struct ibv_mr *mr = NULL;
	if (!pd) {
		rdma_error("Protection domain is NULL, ignoring \n");
		return NULL;
	}
	mr = ibv_reg_mr(pd, addr, length, permission);
	if (!mr) {
		rdma_error("Failed to create mr on buffer, errno: %d \n", -errno);
		return NULL;
	}
	debug("Registered: %p , len: %u , stag: 0x%x \n", 
			mr->addr, 
			(unsigned int) mr->length, 
			mr->lkey);
	return mr;
}

void rdma_buffer_free(struct ibv_mr *mr) 
{
	if (!mr) {
		rdma_error("Passed memory region is NULL, ignoring\n");
		return ;
	}
	void *to_free = mr->addr;
	rdma_buffer_deregister(mr);
	debug("Buffer %p free'ed\n", to_free);
	free(to_free);
}

void rdma_buffer_deregister(struct ibv_mr *mr) 
{
	if (!mr) { 
		rdma_error("Passed memory region is NULL, ignoring\n");
		return;
	}
	debug("Deregistered: %p , len: %u , stag : 0x%x \n", 
			mr->addr, 
			(unsigned int) mr->length, 
			mr->lkey);
	ibv_dereg_mr(mr);
}

int process_rdma_cm_event(struct rdma_event_channel *echannel, 
		enum rdma_cm_event_type expected_event,
		struct rdma_cm_event **cm_event)
{
	int ret = 1;
	ret = rdma_get_cm_event(echannel, cm_event);
	if (ret) {
		rdma_error("Failed to retrieve a cm event, errno: %d \n",
				-errno);
		return -errno;
	}
	/* lets see, if it was a good event */
	if(0 != (*cm_event)->status){
		rdma_error("CM event has non zero status: %d\n", (*cm_event)->status);
		ret = -((*cm_event)->status);
		/* important, we acknowledge the event */
		rdma_ack_cm_event(*cm_event);
		return ret;
	}
	/* if it was a good event, was it of the expected type */
	if ((*cm_event)->event != expected_event) {
		rdma_error("Unexpected event received: %s [ expecting: %s ]", 
				rdma_event_str((*cm_event)->event),
				rdma_event_str(expected_event));
		/* important, we acknowledge the event */
		rdma_ack_cm_event(*cm_event);
		return -1; // unexpected event :(
	}
	debug("A new %s type event is received \n", rdma_event_str((*cm_event)->event));
	/* The caller must acknowledge the event */
	return ret;
}

int process_work_completion_events (struct ibv_comp_channel *comp_channel, 
		struct ibv_wc *wc, int max_wc)
{
  struct ibv_cq *cq_ptr = NULL;
  void *context = NULL;
  int ret = -1, i, total_wc = 0;
  /* We wait for the notification on the CQ channel */
  ret = ibv_get_cq_event(comp_channel, /* IO channel where we are expecting the notification */ 
      &cq_ptr, /* which CQ has an activity. This should be the same as CQ we created before */ 
      &context); /* Associated CQ user context, which we did set */
  if (ret) {
    rdma_error("Failed to get next CQ event due to %d \n", -errno);
    return -errno;
  }
  /* Request for more notifications. */
  ret = ibv_req_notify_cq(cq_ptr, 0);
  if (ret){
    rdma_error("Failed to request further notifications %d \n", -errno);
    return -errno;
  }
  /* We got notification. We reap the work completion (WC) element. It is 
   * unlikely but a good practice to write the CQ polling code that 
   * can handle zero WCs. ibv_poll_cq can return zero. Same logic as 
   * MUTEX conditional variables in pthread programming.
   */
  total_wc = 0;
  do {
    ret = ibv_poll_cq(cq_ptr /* the CQ, we got notification for */, 
        max_wc - total_wc /* number of remaining WC elements*/,
        wc + total_wc/* where to store */);
    if (ret < 0) {
      rdma_error("Failed to poll cq for wc due to %d \n", ret);
      /* ret is errno here */
      return ret;
    }
    total_wc += ret;
  } while (total_wc < max_wc); 
  /* Now we check validity and status of I/O work completions */
  for( i = 0 ; i < total_wc ; i++) {
    if (wc[i].status != IBV_WC_SUCCESS) {
      rdma_error("Work completion (WC) has error status: %s at index %d\n", 
          ibv_wc_status_str(wc[i].status), i);
      /* return negative value */
      return -(wc[i].status);
    }
  }
  /* Similar to connection management events, we need to acknowledge CQ events */
  ibv_ack_cq_events(cq_ptr, 
      1 /* we received one event notification. This is not 
           number of WC elements */);
    return total_wc; 
}

int wait_for_comp(struct r_context *rctx, int num_events)
{
  // FIXME
  int ret = -1;
  struct ibv_wc wc[2];
  ret = process_work_completion_events(rctx->comp_channel, wc, num_events);
  if (ret != num_events) {
    rdma_error("Failed to process cq event , exp: %d, ret = %d \n", num_events, ret);
    return ret;
  }
  return 0;
}

/* Code acknowledgment: rping.c from librdmacm/examples */
int get_addr(char *dst, struct sockaddr *addr)
{
	struct addrinfo *res;
	int ret = -1;
	ret = getaddrinfo(dst, NULL, NULL, &res);
	if (ret) {
		rdma_error("getaddrinfo failed - invalid hostname or IP address\n");
		return ret;
	}
	memcpy(addr, res->ai_addr, sizeof(struct sockaddr_in));
	freeaddrinfo(res);
	return ret;
}

int r_context_init(struct r_context **_rctx)
{
  struct r_context *rctx = (struct r_context*)malloc(sizeof(struct r_context));

	int ret = -1;
	/*  Open a channel used to report asynchronous communication event */
	rctx->cm_event_channel = rdma_create_event_channel();
	if (!rctx->cm_event_channel) {
		rdma_error("Creating cm event channel failed, errno: %d \n", -errno);
		return -errno;
	}
	//debug("RDMA CM event channel is created at : %p \n", cm_event_channel);
	/* rdma_cm_id is the connection identifier (like socket) which is used 
	 * to define an RDMA connection. 
	 */
	ret = rdma_create_id(rctx->cm_event_channel, &rctx->cm_id, 
			NULL,
			RDMA_PS_TCP);
	if (ret) {
		rdma_error("Creating cm id failed with errno: %d \n", -errno); 
		return -errno;
	}
  *_rctx = rctx;
  printf("Creating new rctx: %p, with cm_id: %p, ibv_ctx: %p\n", (void*)rctx, (void*)rctx->cm_id, (void*)rctx->cm_id->verbs);
  return 0;
}

int resolv_ctx(struct r_context *rctx, struct sockaddr_in *s_addr)
{
  int ret = -1;
  struct rdma_cm_event *cm_event = NULL;
  /* Resolve destination and optional source addresses from IP addresses  to
   * an RDMA address.  If successful, the specified rdma_cm_id will be bound
   * to a local device. */
  ret = rdma_resolve_addr(rctx->cm_id, NULL, (struct sockaddr*) s_addr, 2000);
  if (ret) {
    rdma_error("Failed to resolve address, errno: %d \n", -errno);
    return -errno;
  }
  //debug("waiting for cm event: RDMA_CM_EVENT_ADDR_RESOLVED\n");
  ret  = process_rdma_cm_event(rctx->cm_event_channel, 
      RDMA_CM_EVENT_ADDR_RESOLVED,
      &cm_event);
  if (ret) {
    rdma_error("Failed to receive a valid event, ret = %d \n", ret);
    return ret;
  }
  /* we ack the event */
  ret = rdma_ack_cm_event(cm_event);
  if (ret) {
    rdma_error("Failed to acknowledge the CM event, errno: %d\n", -errno);
    return -errno;
  }
  //debug("RDMA address is resolved \n");

  /* Resolves an RDMA route to the destination address in order to 
   * establish a connection */
  ret = rdma_resolve_route(rctx->cm_id, 2000);
  if (ret) {
    rdma_error("Failed to resolve route, erno: %d \n", -errno);
    return -errno;
  }
  //debug("waiting for cm event: RDMA_CM_EVENT_ROUTE_RESOLVED\n");
  ret = process_rdma_cm_event(rctx->cm_event_channel, 
      RDMA_CM_EVENT_ROUTE_RESOLVED,
      &cm_event);
  if (ret) {
    rdma_error("Failed to receive a valid event, ret = %d \n", ret);
    return ret;
  }
  /* we ack the event */
  ret = rdma_ack_cm_event(cm_event);
  if (ret) {
    rdma_error("Failed to acknowledge the CM event, errno: %d \n", -errno);
    return -errno;
  }
  printf("Trying to connect to server at : %s port: %d \n", 
      inet_ntoa(s_addr->sin_addr),
      ntohs(s_addr->sin_port));
  rctx->ctx = rctx->cm_id->verbs;
  return 0;
}

int setup_ctx_resources(struct r_context *rctx)
{
  int ret = -1;
  if(!rctx || !rctx->ctx) {
    rdma_error("Cannot setup ctx resources, is ctx set up?\n");
    return -1;
  }

	/* Protection Domain (PD) is similar to a "process abstraction" 
	 * in the operating system. All resources are tied to a particular PD. 
	 * And accessing resourses across PD will result in a protection fault.
	 */
	rctx->pd = ibv_alloc_pd(rctx->ctx);
	if (!rctx->pd) {
		rdma_error("Failed to alloc pd, errno: %d \n", -errno);
		return -errno;
	}
  struct ibv_device_attr_ex attrx;
  if (ibv_query_device_ex(rctx->ctx, NULL, &attrx)) {
    fprintf(stderr, "Couldn't query device for its features\n");
  } else {
    //struct ibv_alloc_dm_attr dm_attr = {};
    printf("Max. device memory available: %ld\n", attrx.max_dm_size);
  }
	//debug("pd allocated at %p \n", pd);

	/* Now we need a completion channel, were the I/O completion 
	 * notifications are sent. Remember, this is different from connection 
	 * management (CM) event notifications. 
	 * A completion channel is also tied to an RDMA device, hence we will 
	 * use cm_client_id->verbs. 
	 */
	//rctx->comp_channel = ibv_create_comp_channel(rctx->ctx);
	//if (!rctx->comp_channel) {
	//	rdma_error("Failed to create IO completion event channel, errno: %d\n",
	//		       -errno);
	//return -errno;
	//}
	//debug("completion event channel created at : %p \n", io_completion_channel);
	/* Now we create a completion queue (CQ) where actual I/O 
	 * completion metadata is placed. The metadata is packed into a structure 
	 * called struct ibv_wc (wc = work completion). ibv_wc has detailed 
	 * information about the work completion. An I/O request in RDMA world 
	 * is called "work" ;) 
	 */
	//rctx->cq = ibv_create_cq(rctx->ctx /* which device*/, 
	//		CQ_CAPACITY /* maximum capacity*/, 
	//		NULL /* user context, not used here */,
	//		rctx->comp_channel /* which IO completion channel */, 
	//		0 /* signaling vector, not used here*/);
	//if (!rctx->cq) {
	//	rdma_error("Failed to create CQ, errno: %d \n", -errno);
	//	return -errno;
	//}
	//debug("CQ created at %p with %d elements \n", client_cq, client_cq->cqe);
  //ret = ibv_req_notify_cq(rctx->cq, 0);
	//if (ret) {
	//	rdma_error("Failed to request notifications, errno: %d\n", -errno);
	//	return -errno;
	//}

  rctx->send_cq = ibv_create_cq(rctx->ctx, CQ_CAPACITY, NULL, NULL, 0);
  rctx->recv_cq = ibv_create_cq(rctx->ctx, CQ_CAPACITY, NULL, NULL, 0);
  if(!rctx->send_cq || !rctx->recv_cq) {
    rdma_error("Failed to create CQs, errno: %d \n", -errno);
    return -errno;
  }

  /* Now the last step, set up the queue pair (send, recv) queues and their capacity.
   * The capacity here is define statically but this can be probed from the 
   * device. We just use a small number as defined in rdma_common.h */
  struct ibv_qp_init_attr qp_init_attr;
  bzero(&qp_init_attr, sizeof qp_init_attr);
  qp_init_attr.cap.max_recv_sge = MAX_SGE; /* Maximum SGE per receive posting */
  qp_init_attr.cap.max_send_sge = MAX_SGE; /* Maximum SGE per send posting */
  qp_init_attr.cap.max_recv_wr = MAX_WR; /* Maximum receive posting capacity */
  qp_init_attr.cap.max_send_wr = MAX_WR; /* Maximum send posting capacity */
  qp_init_attr.qp_type = IBV_QPT_RC; /* QP type, RC = Reliable connection */

  /* We use same completion queue, but one can use different queues */
  //qp_init_attr.recv_cq = rctx->cq; /* Where should I notify for receive completion operations */
  //qp_init_attr.send_cq = rctx->cq; /* Where should I notify for send completion operations */
  qp_init_attr.recv_cq = rctx->recv_cq;
  qp_init_attr.send_cq = rctx->send_cq;

  /*Lets create a QP */
  ret = rdma_create_qp(rctx->cm_id /* which connection id */,
      rctx->pd /* which protection domain*/,
      &qp_init_attr /* Initial attributes */);
  if (ret) {
		rdma_error("Failed to create QP, errno: %d \n", -errno);
	       return -errno;
	}
	rctx->qp = rctx->cm_id->qp;
  return 0;
}

int r_context_cleanup(struct r_context *rctx)
{
	/* Destroy QP */
	rdma_destroy_qp(rctx->cm_id);
	/* Destroy client cm id */
	int ret = rdma_destroy_id(rctx->cm_id);
	if (ret) {
		rdma_error("Failed to destroy client id cleanly, %d \n", -errno);
		// we continue anyways;
	}
	/* Destroy CQ */
	//ret = ibv_destroy_cq(rctx->cq);
	//if (ret) {
	//	rdma_error("Failed to destroy completion queue cleanly, %d \n", -errno);
	//	// we continue anyways;
	//}
	/* Destroy completion channel */
	//ret = ibv_destroy_comp_channel(rctx->comp_channel);
	//if (ret) {
	//	rdma_error("Failed to destroy completion channel cleanly, %d \n", -errno);
	//	// we continue anyways;
	//}
  ret = ibv_destroy_cq(rctx->send_cq);
	if (ret) {
		rdma_error("Failed to destroy completion queue cleanly, %d \n", -errno);
		// we continue anyways;
	}
  ret = ibv_destroy_cq(rctx->recv_cq);
  if (ret) {
    rdma_error("Failed to destroy completion queue cleanly, %d \n", -errno);
    // we continue anyways;
  }

  /* Destroy protection domain */
	ret = ibv_dealloc_pd(rctx->pd);
	if (ret) {
		rdma_error("Failed to destroy client protection domain cleanly, %d \n", -errno);
		// we continue anyways;
	}
	rdma_destroy_event_channel(rctx->cm_event_channel);
	printf("Client resource clean up is complete \n");
	return 0;
}
