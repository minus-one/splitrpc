// Copyright (c) Adithya Kumar, The Pennsylvania State University. All rights reserved.
// Licensed under the MIT License.

// Source heavily lifted from Animesh Trivedi's Github repo rdma-example
#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <errno.h>
#include <getopt.h>

#include <netdb.h>
#include <netinet/in.h>	
#include <arpa/inet.h>
#include <sys/socket.h>

#include <rdma/rdma_cma.h>
#include <infiniband/verbs.h>

/* Error Macro*/
#define rdma_error(msg, args...) do {\
	fprintf(stderr, "%s : %d : ERROR : " msg, __FILE__, __LINE__, ## args);\
}while(0);

#ifdef ACN_RDMA_DEBUG 
/* Debug Macro */
#define debug(msg, args...) do {\
    printf("DEBUG: "msg, ## args);\
}while(0);

#else 

#define debug(msg, args...) 

#endif /* ACN_RDMA_DEBUG */

/* Default port where the RDMA server is listening */
#define DEFAULT_RDMA_PORT (20886)

#define P2P_RPC_MAX_QUEUE_SIZE 512

/* Capacity of the completion queue (CQ) */
#define CQ_CAPACITY (P2P_RPC_MAX_QUEUE_SIZE)
/* MAX SGE capacity */
#define MAX_SGE (2)
/* MAX work requests */
#define MAX_WR (32)


/* 
 * We use attribute so that compiler does not step in and try to pad the structure.
 * We use this structure to exchange information between the server and the client. 
 *
 * For details see: http://gcc.gnu.org/onlinedocs/gcc/Type-Attributes.html
 */
struct __attribute((packed)) rdma_buffer_attr {
  uint64_t address;
  uint32_t length;
  union stag {
	  /* if we send, we call it local stags */
	  uint32_t local_stag;
	  /* if we receive, we call it remote stag */
	  uint32_t remote_stag;
  }stag;
};

// Used to request and respond for buffer allocation requests
// For a specific function-id
// Client passes the req-size, and resp-size
// The server will create a rr-pool for this request and respond
// with the buffer_attributes
struct __attribute((packed)) buf_mon_rr {
  enum {
    MSG_ALLOC,
    MSG_RELEASE
  }type; // Whether this is a request for allocation/releasing the buffers
  uint32_t func_id; // Associate this func-id for this set of buffers
  // To be set by the server
  size_t queue_size;
  size_t req_size;
  size_t resp_size;
  struct rdma_buffer_attr req_buf_attr;
  struct rdma_buffer_attr resp_buf_attr;
  struct rdma_buffer_attr state_buf_attr;
};

// This is for the request monitor
// The request monitor listens to launch requests corresponding to a function-id
struct __attribute((packed)) req_mon_rr {
  enum {
    MSG_NOTIFY_WORK,
    MSG_NOTIFY_COMPL
  }type;
  uint32_t func_id;
  uint32_t rr_idx;
};

/* resolves a given destination name to sin_addr */
int get_addr(char *dst, struct sockaddr *addr);

/* prints RDMA buffer info structure */
void show_rdma_buffer_attr(struct rdma_buffer_attr *attr);

/* 
 * Processes an RDMA connection management (CM) event. 
 * @echannel: CM event channel where the event is expected. 
 * @expected_event: Expected event type 
 * @cm_event: where the event will be stored 
 */
int process_rdma_cm_event(struct rdma_event_channel *echannel, 
		enum rdma_cm_event_type expected_event,
		struct rdma_cm_event **cm_event);

struct ibv_mr* rdma_gpu_buffer_alloc(struct ibv_pd *pd, 
		uint32_t length, 
		enum ibv_access_flags permission);


/* Allocates an RDMA buffer of size 'length' with permission permission. This 
 * function will also register the memory and returns a memory region (MR) 
 * identifier or NULL on error. 
 * @pd: Protection domain where the buffer should be allocated 
 * @length: Length of the buffer 
 * @permission: OR of IBV_ACCESS_* permissions as defined for the enum ibv_access_flags
 */
struct ibv_mr* rdma_buffer_alloc(struct ibv_pd *pd, 
		uint32_t length, 
		enum ibv_access_flags permission);

void rdma_gpu_buffer_free(struct ibv_mr *mr);

/* Frees a previously allocated RDMA buffer. The buffer must be allocated by 
 * calling rdma_buffer_alloc();
 * @mr: RDMA memory region to free 
 */
void rdma_buffer_free(struct ibv_mr *mr);

/* This function registers a previously allocated memory. Returns a memory region 
 * (MR) identifier or NULL on error.
 * @pd: protection domain where to register memory 
 * @addr: Buffer address 
 * @length: Length of the buffer 
 * @permission: OR of IBV_ACCESS_* permissions as defined for the enum ibv_access_flags
 */
struct ibv_mr *rdma_buffer_register(struct ibv_pd *pd, 
		void *addr, 
		uint32_t length, 
		enum ibv_access_flags permission);
/* Deregisters a previously register memory 
 * @mr: Memory region to deregister 
 */
void rdma_buffer_deregister(struct ibv_mr *mr);

/* Processes a work completion (WC) notification. 
 * @comp_channel: Completion channel where the notifications are expected to arrive 
 * @wc: Array where to hold the work completion elements 
 * @max_wc: Maximum number of expected work completion (WC) elements. wc must be 
 *          atleast this size.
 */
int process_work_completion_events(struct ibv_comp_channel *comp_channel, 
		struct ibv_wc *wc, 
		int max_wc);
// Waits for the completion of num_events
int wait_for_comp(struct r_context *rctx, int num_events);

/* prints some details from the cm id */
void show_rdma_cmid(struct rdma_cm_id *id);

struct r_context {
  struct rdma_event_channel *cm_event_channel;
  struct rdma_cm_id *cm_id;
  struct rdma_cm_id *cm_server_id; // Not sure if this will be useful
  struct ibv_context *ctx;
  struct ibv_pd *pd;
  struct ibv_cq *cq;
  struct ibv_cq *send_cq, *recv_cq;
  struct ibv_comp_channel *comp_channel;
  struct ibv_qp *qp; // Shortcut
};

/* This function prepares client side connection resources for an RDMA connection */
int r_context_init(struct r_context **_rctx);

int resolv_ctx(struct r_context *rctx, struct sockaddr_in *s_addr);

// Call once cm_id has been established i.e. cm_id is created, and address is resolved
int setup_ctx_resources(struct r_context *rctx);

int r_context_cleanup(struct r_context *rctx);

// Agnostic of contents, but does require set up of sge and wr
static int pre_post_recv_mr(struct r_context *rctx, struct ibv_mr *mr) 
{
  /* We pre-post this receive buffer on the QP. SGE credentials is where we 
   * receive the metadata from the client */
  struct ibv_sge recv_sge;
  recv_sge.addr = (uint64_t) mr->addr;
  recv_sge.length = mr->length;
  recv_sge.lkey = mr->lkey;
  /* Now we link this SGE to the work request (WR) */
  struct ibv_recv_wr recv_wr, *bad_recv_wr = NULL;
  bzero(&recv_wr, sizeof(recv_wr));
  recv_wr.sg_list = &recv_sge;
  recv_wr.num_sge = 1; // only one SGE

  return ibv_post_recv(rctx->qp /* which QP */,
      &recv_wr /* receive work request*/,
      &bad_recv_wr /* error WRs */);
}

static int post_send_mr(struct r_context *rctx, struct ibv_mr *mr)
{
  int ret = -1;
  struct ibv_sge send_sge;
  send_sge.addr = (uint64_t) mr->addr;
  send_sge.length = mr->length;
  send_sge.lkey = mr->lkey;
  /* now we link this sge to the send request */
  struct ibv_send_wr send_wr, *bad_send_wr = NULL;
  bzero(&send_wr, sizeof(send_wr));
  send_wr.sg_list = &send_sge;
  send_wr.num_sge = 1; // only 1 SGE element in the array 
  send_wr.opcode = IBV_WR_SEND; // This is a send request 
  send_wr.send_flags = IBV_SEND_SIGNALED; // We want to get notification 

  ret = ibv_post_send(rctx->qp, &send_wr, &bad_send_wr);
  if (ret) {
    rdma_error("Posting of server metdata failed, errno: %d \n",
        -errno);
    return -errno;
  }
  return 0;
}

static int post_send_imm_with_inline_mr(struct r_context *rctx, struct ibv_mr *mr, int imm_data)
{
  int ret = -1;
  struct ibv_sge send_sge;
  send_sge.addr = (uint64_t) mr->addr;
  send_sge.length = mr->length;
  send_sge.lkey = mr->lkey;

  struct ibv_send_wr send_wr, *bad_send_wr = NULL;
  bzero(&send_wr, sizeof(send_wr));
  send_wr.sg_list = &send_sge;
  send_wr.num_sge = 1;
  send_wr.opcode = IBV_WR_SEND_WITH_IMM;
  send_wr.send_flags = IBV_SEND_SIGNALED | IBV_SEND_INLINE;
  send_wr.imm_data = imm_data;

  ret = ibv_post_send(rctx->qp, &send_wr, &bad_send_wr);
  if (ret) {
    rdma_error("Posting of server metdata failed, errno: %d \n",
        -errno);
    return -errno;
  }
  return 0;
}

static inline
int poll_on_cq(struct ibv_cq *cq, struct ibv_wc *wc, int max_wc)
{
  int ret = -1;
  ret = ibv_poll_cq(cq, max_wc, wc);
  if(ret < 0) {
    rdma_error("Failed to poll cq for wc due to %d \n", ret);
    /* ret is errno here */
    return ret;
  }
  for(int i = 0 ; i < ret; i++) {
    if (wc[i].status != IBV_WC_SUCCESS) {
      rdma_error("Work completion (WC) has error status: %s at index %d\n", 
          ibv_wc_status_str(wc[i].status), i);
      /* return negative value */
      return -(wc[i].status);
    }
  }
  return ret;
}

static inline
int poll_on_cq_and_get_imm_data(struct ibv_cq *cq, struct ibv_wc *wc, int& imm_data)
{
  int ret = -1;
  ret = ibv_poll_cq(cq, 1, wc);
  if(ret < 0) {
    rdma_error("Failed to poll cq for wc due to %d \n", ret);
    /* ret is errno here */
    return ret;
  }
  if(ret == 1) {
    if(wc[0].status != IBV_WC_SUCCESS) {
      rdma_error("Work completion (WC) has error status: %s at index %d\n", 
          ibv_wc_status_str(wc[0].status), 0);
      /* return negative value */
      return -(wc[0].status);
    }
    imm_data = wc[0].imm_data;
  }
  return ret;
}

static inline 
int wait_on_cq(struct ibv_cq *cq, int max_wc)
{
  int ret = -1;
  struct ibv_wc wc[max_wc];
  int total_wc = 0;
  do {
    ret = ibv_poll_cq(cq /* the CQ, we want notification for */, 
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
  for(int i = 0 ; i < total_wc ; i++) {
    if (wc[i].status != IBV_WC_SUCCESS) {
      rdma_error("Work completion (WC) has error status: %s at index %d\n", 
          ibv_wc_status_str(wc[i].status), i);
      /* return negative value */
      return -(wc[i].status);
    }
  }
  return 0;
}
  
static inline 
int post_send_wr(struct r_context *rctx, struct ibv_send_wr *wr)
{
  struct ibv_send_wr *bad_wr = NULL;
  if (ibv_post_send(rctx->qp, wr, &bad_wr)) {
    rdma_error("Failed to post send_wr, errno: %d \n", -errno);
    return -errno;
  }
  return 0;
}

// Posts a send with the specificed mr
static inline
int post_send_mr_sync(struct r_context *rctx, struct ibv_mr *mr)
{
  if(post_send_mr(rctx, mr))
    return -1;
  return wait_on_cq(rctx->send_cq, 1);
  //return wait_for_comp(rctx, 1);
}
