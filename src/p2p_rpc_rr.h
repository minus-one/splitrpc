// Copyright (c) Adithya Kumar, The Pennsylvania State University. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "p2p_rpc_conn_info.h"
#include <unordered_map>
#include "p2p_bufs.h"

enum RR_STATUS { CREATE, PROCESS, FINISH, UNUSED };

// Holder for a request-response
struct p2p_rpc_rr {
////////////// App specific
  uint64_t req_token;
  // Corresponds to g_params
  uint8_t *req_payload, *resp_payload;
  size_t req_size, resp_size;
  size_t max_req_size, max_resp_size;
  int req_device_id, resp_device_id;
  // Corresponds to door_bells of the actual rr
  volatile uint32_t *state;

//////////// RR_Pool specific
  // This is the internal state in the RR Pool
  uint32_t rr_state;

///////////// Transport specific
  // This is common to both RX/TX
  // Routing info for a request-response 
  //struct p2p_rpc_conn_info *_conn_info; 
  struct p2p_rpc_conn_info _client_conn_info;
  struct p2p_hbufs *transport_mbufs; // Points to the message buffers of the transport used (for DPDK -> rte_mbuf) 
  struct p2p_hbufs *hdr_bufs;
  struct p2p_bufs *payload_bufs;
  // Corresponds to g_copy_params
  struct p2p_sk_buf *payload_sk_bufs;
};

static inline struct p2p_rpc_rr *
rr_base_alloc()
{
  struct p2p_rpc_rr *rr = new p2p_rpc_rr;
  rr->transport_mbufs = new p2p_hbufs;
  rr->hdr_bufs = new p2p_hbufs;
  rr->payload_bufs = new p2p_bufs;
  rr->payload_sk_bufs = new p2p_sk_buf;
  // Depending on whether it is zc or not, we need to allocate
  // Also needs to depend on whether this is on device/host
  rr->req_payload = NULL; 
  rr->resp_payload = NULL; 
  rr->req_device_id = -2;
  rr->resp_device_id = -2;
  rr->state = NULL;
  rr->rr_state = RR_STATUS::CREATE;

  // Stats that need to be reset across re-use of rr
  rr->transport_mbufs->num_items = 0;
  rr->hdr_bufs->num_items = 0;
  rr->payload_bufs->num_items = 0;
  rr->req_size = 0;
  rr->resp_size = 0;

  return rr;
}

static inline void
rr_base_free(struct p2p_rpc_rr *rr)
{
  delete rr->transport_mbufs;
  delete rr->hdr_bufs;
  delete rr->payload_bufs;
  delete rr->payload_sk_bufs;
  delete rr;
}

// Proto alloc does not allocate a 
// conn or buffers 
static inline struct p2p_rpc_rr *
rr_proto_alloc(struct p2p_rpc_conn_info *conn_info, 
    size_t max_req_size, size_t max_resp_size)
{
  struct p2p_rpc_rr *rr = rr_base_alloc();
  rr->max_req_size = max_req_size;
  rr->max_resp_size = max_resp_size;
 
  rr->req_token = 0;
  //rr->_conn_info = conn_info;
  // Pre-initializing the client-conn information
  memcpy((void*)&rr->_client_conn_info.hdr_template, (void*)&conn_info->hdr_template, RPC_HEADER_LEN);

  return rr;
}

// Clones an rr including conn_info - does not copy data/bufs
static inline struct p2p_rpc_rr*
rr_clone_alloc(const struct p2p_rpc_rr *rr)
{
  struct p2p_rpc_rr *new_rr = rr_base_alloc();
  new_rr->max_req_size = rr->max_req_size;
  new_rr->max_resp_size = rr->max_resp_size;
  new_rr->req_device_id = rr->req_device_id;
  new_rr->resp_device_id = rr->resp_device_id;
  new_rr->req_token = 0;
  //new_rr->_conn_info = rr->_conn_info;
  memcpy((void*)&new_rr->_client_conn_info.hdr_template, (void*)&rr->_client_conn_info.hdr_template, RPC_HEADER_LEN);

  return new_rr;
}

// FIXME: Remove this API entirely
//static inline struct p2p_rpc_rr *
//rr_alloc(struct p2p_rpc_conn_info *conn_info,
//         uint64_t req_token,
//         size_t max_req_size,
//         size_t max_resp_size)
//{
//  struct p2p_rpc_rr *rr = rr_base_alloc();
//  rr->max_req_size = max_req_size;
//  rr->max_resp_size = max_resp_size;
//
//  // Depending on whether it is zc or not, we need to allocate
//  // Also needs to depend on whether this is on device/host
//  rr->req_payload = new uint8_t[max_req_size];
//  rr->resp_payload = new uint8_t[max_resp_size];
//  rr->req_device_id = -1;
//  rr->resp_device_id = -1;
//
//  // Setting up the rr
//  rr->req_token = req_token;
//  rr->_conn_info = conn_info;
//
//  return rr;
//}

inline __attribute__((always_inline)) static void
rr_merge_mbuf(struct p2p_rpc_rr *rr,
    uintptr_t transport_mbuf,
    uintptr_t hdr_buf,
    uintptr_t payload_buf,
    size_t item_size,
    int idx)
{
  rr->transport_mbufs->burst_items[idx] = transport_mbuf;
  rr->transport_mbufs->num_items++;

  rr->hdr_bufs->burst_items[idx] = hdr_buf;
  rr->hdr_bufs->num_items++;

  // Here we can ideally coalesce into SG lists
  rr->payload_bufs->burst_items[idx] = payload_buf;
  rr->payload_bufs->item_size[idx] = item_size;

  rr->payload_bufs->num_items++;
}

// Appends a transport mbuf (with hdr and payload info) into this rr's response and inc size
// This does not check the frame-id, but just appends it to the end
inline __attribute__((always_inline)) static void
rr_append_resp_mbuf(struct p2p_rpc_rr *rr,
    uintptr_t transport_mbuf,
    uintptr_t hdr_buf,
    uintptr_t payload_buf,
    size_t item_size)
{
  rr_merge_mbuf(rr, transport_mbuf, hdr_buf, payload_buf, item_size, rr->transport_mbufs->num_items);
  rr->resp_size += item_size;
}

// Appends a transport mbuf (with hdr and payload info) into this rr's request and inc size
// This does not check the frame-id, but just appends it to the end
inline __attribute__((always_inline)) static void
rr_append_req_mbuf(struct p2p_rpc_rr *rr,
    uintptr_t transport_mbuf,
    uintptr_t hdr_buf,
    uintptr_t payload_buf,
    size_t item_size)
{
  rr_merge_mbuf(rr, transport_mbuf, hdr_buf, payload_buf, item_size, rr->transport_mbufs->num_items);
  rr->req_size += item_size;
}

// Appends a transport mbuf (with hdr and payload info) into this rr's response and inc size
inline __attribute__((always_inline)) static void
rr_merge_resp_mbuf(struct p2p_rpc_rr *rr,
    uintptr_t transport_mbuf,
    uintptr_t hdr_buf,
    uintptr_t payload_buf,
    size_t item_size,
    int seq_num)
{
  rr_merge_mbuf(rr, transport_mbuf, hdr_buf, payload_buf, item_size, seq_num);
  rr->resp_size += item_size;
}

// Appends a transport mbuf (with hdr and payload info) into this rr's request and inc size
inline __attribute__((always_inline)) static void
rr_merge_req_mbuf(struct p2p_rpc_rr *rr,
    uintptr_t transport_mbuf,
    uintptr_t hdr_buf,
    uintptr_t payload_buf,
    size_t item_size,
    int seq_num)
{
  rr_merge_mbuf(rr, transport_mbuf, hdr_buf, payload_buf, item_size, seq_num);
  rr->req_size += item_size;
}

static void
rr_print_bufs(p2p_rpc_rr *rr)
{
  printf("==========================================================================\n");
  printf("Req: %p, num_bufs: %d\n", (void*)rr, rr->payload_bufs->num_items);

  for(int i = 0 ; i < rr->payload_bufs->num_items ; i++) {
    void *start_addr = (void*)rr->payload_bufs->burst_items[i];
    size_t item_size = rr->payload_bufs->item_size[i];
    void *end_addr = (void*)((uint8_t*)start_addr + item_size);
    int seq_num = (((struct p2p_rpc_hdr*)rr->hdr_bufs->burst_items[i])->seq_num);
    printf("Req: %p, start_addr: %p, end_addr: %p, seq_num: %d\n", (void*)rr, start_addr, end_addr, seq_num);
  }
  printf("==========================================================================\n");
}
