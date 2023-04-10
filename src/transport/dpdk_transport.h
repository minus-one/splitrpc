// Copyright (c) Adithya Kumar, The Pennsylvania State University. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "p2p_rpc.h"
#include "p2p_rpc_conn_info.h"
#include "dpdk_utils.h"
#include "dpdk_init.h"
#include "dpdk_rx_tx.h"
#include "p2p_rpc_rr.h"
#include "p2p_rpc_rr_pool.h"

#include "utils/debug_utils.h"

#ifdef __cplusplus
extern "C" { 
#endif

/********************* RAW APIs **********************************************/
// Reads a bunch of transport mbufs and sets the appropriate buf-ptrs
__rte_always_inline int 
get_requests_zc(struct dpdk_ctx *_dpdk_ctx,
    struct p2p_hbufs *transport_mbufs,
    struct p2p_hbufs *hdr_bufs,
    struct p2p_bufs *payload_bufs)
{
  uint16_t nb = 0;
  struct rte_mbuf **bufs = (struct rte_mbuf**)transport_mbufs->burst_items;
  nb = rx_pkts(_dpdk_ctx, bufs);
  transport_mbufs->num_items = nb;
  nb = pkts_to_buf_ptrs(_dpdk_ctx, transport_mbufs, hdr_bufs, payload_bufs);
  return nb;
}

__rte_always_inline int
get_requests_zc_blocking(struct dpdk_ctx *_dpdk_ctx,
    struct p2p_hbufs *transport_mbufs,
    struct p2p_hbufs *hdr_bufs,
    struct p2p_bufs *payload_bufs,
    int burst_size,
    volatile bool& cancel)
{
  int nb_rx = 0;
  struct rte_mbuf **bufs = (struct rte_mbuf**)transport_mbufs->burst_items;
  while(ACCESS_ONCE(cancel) == 0) {
    nb_rx += rx_pkts(_dpdk_ctx, &bufs[nb_rx]);
    if(nb_rx >= burst_size)
      break;
  }
  if(unlikely(nb_rx == 0))
      return 0;
  transport_mbufs->num_items = nb_rx;

  return pkts_to_buf_ptrs(_dpdk_ctx, transport_mbufs, hdr_bufs, payload_bufs);  
}

// Sends a bunch of transport mbufs directly
__rte_always_inline int
send_requests_zc(struct dpdk_ctx *_dpdk_ctx,
    struct p2p_hbufs *transport_mbufs)
{
  struct rte_mbuf **bufs = (struct rte_mbuf**)transport_mbufs->burst_items;
  int nb_tx = tx_pkts(_dpdk_ctx, bufs, transport_mbufs->num_items);
  transport_mbufs->num_items -= nb_tx;
  return nb_tx;
}

/*******************************************************************************/
/************************Transport APIs implemented in CPU*************************/

inline int 
swap_hdrs_on_cpu(struct p2p_hbufs *transport_mbufs, 
    struct p2p_hbufs *hdr)
{
  //struct rte_mbuf **mbufs = (struct rte_mbuf**)transport_mbufs->burst_items;
  // FIXME: Can this be vectorized or optimized?
  for (int i = 0; i < hdr->num_items; i++)
  {
    struct p2p_rpc_hdr *pkt_hdr = (struct p2p_rpc_hdr*)(hdr->burst_items[i]);
    swap_eth_hdr(pkt_hdr);
    add_udp_cksum(pkt_hdr);
  }
  return 1;
}

inline int 
swap_sockperf_hdrs_on_cpu(struct p2p_hbufs *transport_mbufs, 
    struct p2p_hbufs *hdr)
{
  struct rte_mbuf **mbufs = (struct rte_mbuf**)transport_mbufs->burst_items;
  // FIXME: Can this be vectorized or optimized?
  for (int i = 0; i < hdr->num_items; i++)
  {
    struct p2p_rpc_hdr *pkt_hdr = (struct p2p_rpc_hdr*)(hdr->burst_items[i]);
    swap_eth_hdr(pkt_hdr);
    mbufs[i]->ol_flags |= ( PKT_TX_IPV4 | PKT_TX_IP_CKSUM | PKT_TX_UDP_CKSUM);
    struct udp_hdr *udp_h = get_udp_header(pkt_hdr);
    struct ipv4_hdr *ip_h = get_ip_header(pkt_hdr);
    udp_h->check = 0;
    udp_h->check = rte_ipv4_phdr_cksum((struct rte_ipv4_hdr*)ip_h, mbufs[i]->ol_flags);
    set_sockperf_header(pkt_hdr); 
  }
  return 1;
}

// Assumes mbufs have been allocated
// Returns number of bytes copied
inline int 
scatter_payload_on_cpu(struct p2p_bufs *payload_bufs, 
    uint8_t *payload, 
    size_t payload_size)
{
  uint8_t *payload_buf;
  size_t curr_pkt_size = 0;
  int byte_offset = 0;

  for(int i = 0; i < payload_bufs->num_items; i++) {
    curr_pkt_size = (payload_size <= RPC_MTU) ? payload_size : RPC_MTU;
    payload_size -= curr_pkt_size;

    // Scatter payload into payload_bufs
    payload_buf = (uint8_t*)payload_bufs->burst_items[i];
    memcpy(payload_buf, 
        &(payload[byte_offset]), 
        curr_pkt_size);
    payload_bufs->item_size[i] = curr_pkt_size;
    byte_offset += curr_pkt_size;
  }

  return byte_offset;
}

/*
 * Gathers a bunch of payload_bufs into a single payload
 * Takes offsets based on item_size in the payload_bufs
 * Returns the number of bytes copied.
 */
inline int 
gather_payload_on_cpu(struct p2p_bufs *payload_bufs,
    uint8_t *payload)
{
  uint8_t *payload_buf;
  int byte_offset = 0;

  for(int i = 0 ; i < payload_bufs->num_items; i++) {
    payload_buf = (uint8_t*)payload_bufs->burst_items[i];
    memcpy(&payload[byte_offset],
        payload_buf,
        payload_bufs->item_size[i]);
    byte_offset += payload_bufs->item_size[i];
  }

  return byte_offset;
}

static int 
sg_on_cpu(p2p_sk_buf *skb)
{
  for(size_t i = 0 ; i < skb->num_items; i++) {
    memcpy((void*)skb->o_buf[i], (void*)skb->i_buf[i], skb->len[i]);
  }
  return 1;
}

/************************** APIs for rpc_rr management ******************************/

// This is basically like a blocking recvfrom()
// Not multi-thread safe
__rte_always_inline int
rr_recv_request(struct dpdk_ctx *_dpdk_ctx,
    struct p2p_rpc_rr **new_call_id, P2pRpcRRPool &rr_pool, volatile bool& cancel)
{
  if(_dpdk_ctx->num_ready_rrs > 0) {
    *new_call_id = _dpdk_ctx->ready_rr[--_dpdk_ctx->num_ready_rrs];
    return 1;
  }

  bool got_rr = false;
  struct p2p_hbufs *dpdk_mbufs = _dpdk_ctx->dpdk_mbufs; 
  struct p2p_hbufs *hdr_bufs = _dpdk_ctx->hdr_bufs;
  struct p2p_bufs *payload_bufs = _dpdk_ctx->payload_bufs;
  
  struct p2p_rpc_rr *call_id;

  while(!got_rr && ACCESS_ONCE(cancel) == 0) {
    while(get_requests_zc(_dpdk_ctx, dpdk_mbufs, hdr_bufs, payload_bufs) == 0 && ACCESS_ONCE(cancel) == 0);

    for(int i = 0 ; i < dpdk_mbufs->num_items; i++) {
      struct p2p_rpc_hdr *rpc_hdr = 
        (struct p2p_rpc_hdr*)(hdr_bufs->burst_items[i]);
      
      // FIXME: First Map function-id to an rr-pool
      // Map to an appropriate p2p_rpc_rr
      call_id = rr_pool.get_rr(rpc_hdr);
      
      if(likely(rpc_hdr->seq_num >= 0 && rpc_hdr->seq_num < MAX_BI_SIZE)) {
        rr_merge_req_mbuf(call_id, 
            dpdk_mbufs->burst_items[i], hdr_bufs->burst_items[i], 
            payload_bufs->burst_items[i], payload_bufs->item_size[i],
            rpc_hdr->seq_num);
      } else {
        release_dpdk_mbufs((struct rte_mbuf**)&dpdk_mbufs->burst_items[i], 1);
      }
      
      if(call_id->req_size == call_id->max_req_size) {
        // At this point we know that whole request is received
        _dpdk_ctx->ready_rr[_dpdk_ctx->num_ready_rrs++] = call_id;
        got_rr = true;
        TRACE_PRINTF("New Req: %p is ready\n", (void*)call_id);
      } 
    }
  }
  if(got_rr) {
    *new_call_id = _dpdk_ctx->ready_rr[--_dpdk_ctx->num_ready_rrs];
    return 1;
  }
  return 0;
}

// This is basically like a blocking recvfrom()
__rte_always_inline int
rr_recv_response(struct dpdk_ctx *_dpdk_ctx,
    struct p2p_rpc_rr **new_call_id, P2pRpcRRPool &rr_pool, volatile bool& cancel)
{
  if(_dpdk_ctx->num_ready_rrs > 0) {
    *new_call_id = _dpdk_ctx->ready_rr[--_dpdk_ctx->num_ready_rrs];
    return 1;
  }

  bool got_rr = false;
  struct p2p_hbufs *dpdk_mbufs = _dpdk_ctx->dpdk_mbufs; 
  struct p2p_hbufs *hdr_bufs = _dpdk_ctx->hdr_bufs;
  struct p2p_bufs *payload_bufs = _dpdk_ctx->payload_bufs;

  struct p2p_rpc_rr *call_id;

  while(!got_rr && ACCESS_ONCE(cancel) == 0) {
    while(get_requests_zc(_dpdk_ctx, dpdk_mbufs, hdr_bufs, payload_bufs) == 0 && ACCESS_ONCE(cancel) == 0);

    for(int i = 0 ; i < dpdk_mbufs->num_items; i++) {
      struct p2p_rpc_hdr *rpc_hdr = 
        (struct p2p_rpc_hdr*)(hdr_bufs->burst_items[i]);

      // Map to an appropriate p2p_rpc_rr
      call_id = rr_pool.get_rr(rpc_hdr);

      if(likely(rpc_hdr->seq_num >= 0 && rpc_hdr->seq_num < MAX_BI_SIZE)) {
        rr_merge_resp_mbuf(call_id, 
            dpdk_mbufs->burst_items[i], hdr_bufs->burst_items[i], 
            payload_bufs->burst_items[i], payload_bufs->item_size[i],
            rpc_hdr->seq_num);
      } else {
        release_dpdk_mbufs((struct rte_mbuf**)&dpdk_mbufs->burst_items[i], 1);
      }

      if(call_id->resp_size == call_id->max_resp_size) {
        // At this point we know that whole response is received
        _dpdk_ctx->ready_rr[_dpdk_ctx->num_ready_rrs++] = call_id;
        got_rr = true;
      } 
    }
  }
  if(got_rr) {
    *new_call_id = _dpdk_ctx->ready_rr[--_dpdk_ctx->num_ready_rrs];
    return 1;
  }
  return 0;
}

// Assumes payload has been transferred to transport-mbufs
// Calls the transport APIs to send the packet
__rte_always_inline int
rr_send_request(struct dpdk_ctx *_dpdk_ctx,
    struct p2p_rpc_rr *rr, P2pRpcRRPool &rr_pool)
{
  int nb_tx = send_requests_zc(_dpdk_ctx, rr->transport_mbufs);
  // Reap the rr
  if (nb_tx == rr->hdr_bufs->num_items)
  {
    rr->req_size = 0;
    rr_pool.reap_rr(rr); 
    return 1;
  }
  return 0;
}

// Assumes payload has been transferred to transport-mbufs
// Calls the transport APIs to send the packet
__rte_always_inline int
rr_send_response(struct dpdk_ctx *_dpdk_ctx,
    struct p2p_rpc_rr *rr, P2pRpcRRPool &rr_pool)
{
  int nb_tx = send_requests_zc(_dpdk_ctx, rr->transport_mbufs);
  // Reap the rr
  if (nb_tx == rr->hdr_bufs->num_items)
  {
    rr->resp_size = 0;
    rr_pool.reap_rr(rr); 
    return 1;
  }
  return 0;
}

__rte_always_inline int
rr_alloc_mbufs(struct dpdk_ctx *_dpdk_ctx,
    struct p2p_rpc_rr *rr,
    int payload_size)
{
  struct rte_mbuf **bufs = (struct rte_mbuf**)rr->transport_mbufs->burst_items;
  int new_bufs = (payload_size + RPC_MTU - 1) / RPC_MTU;
  rr->transport_mbufs->num_items = alloc_dpdk_mbufs(_dpdk_ctx, bufs, new_bufs, payload_size);
  pkts_to_buf_ptrs(_dpdk_ctx, rr->transport_mbufs, rr->hdr_bufs, rr->payload_bufs);
  return rr->transport_mbufs->num_items;
}

__rte_always_inline void
rr_release_mbufs(struct dpdk_ctx *_dpdk_ctx,
    struct p2p_rpc_rr *rr)
{
  struct rte_mbuf **bufs = (struct rte_mbuf**)rr->transport_mbufs->burst_items;
  release_dpdk_mbufs(bufs, rr->transport_mbufs->num_items);
  rr->transport_mbufs->num_items = 0;
  rr->hdr_bufs->num_items = 0;
  rr->payload_bufs->num_items = 0;
}

__rte_always_inline void
rr_realloc_mbufs(struct dpdk_ctx *_dpdk_ctx,
    struct p2p_rpc_rr *rr, int new_payload_size)
{
  int old_bufs = rr->transport_mbufs->num_items;
  int new_bufs = (new_payload_size + RPC_MTU - 1) / RPC_MTU;

  if(new_bufs < old_bufs) {
    struct rte_mbuf **bufs = (struct rte_mbuf**)rr->transport_mbufs->burst_items;
    int bufs_to_be_released = old_bufs - new_bufs;
    int num_bufs = release_dpdk_mbufs(bufs+new_bufs, bufs_to_be_released);
    rr->transport_mbufs->num_items = new_bufs;
    rr->hdr_bufs->num_items = new_bufs;
    rr->payload_bufs->num_items = new_bufs;
  } else if(new_bufs > old_bufs) {
    // Fallback
    rr_release_mbufs(_dpdk_ctx, rr);
    rr_alloc_mbufs(_dpdk_ctx, rr, new_payload_size);
  }
  // If they are equal, we do nothing
  // FIXME: We still need to update the last mbuf's len if needed
}

// Assumes mbufs have been allocated
// Returns number of header packets
// Assumes the header resides on CPU memory (HOST_MEM or BUFFER_SPLIT)
__rte_always_inline int 
rr_set_hdr(
    struct p2p_rpc_conn_info *conn_info,
    struct p2p_rpc_rr *rr, 
    size_t payload_size)
{
  struct p2p_rpc_hdr *hdr_buf;
  size_t curr_pkt_size = 0;

  for(int i = 0; i < rr->hdr_bufs->num_items; i++) {
    curr_pkt_size = (payload_size <= RPC_MTU) ? payload_size : RPC_MTU;
    payload_size -= curr_pkt_size;

    // Set header from conn for each mbuf
    hdr_buf = (struct p2p_rpc_hdr*)(rr->hdr_bufs->burst_items[i]); 
    memcpy((uint8_t*)hdr_buf, 
        (uint8_t*) &(conn_info->hdr_template), 
        RPC_HEADER_LEN);

    // Routing
    hdr_buf->req_token = rr->req_token;
    hdr_buf->seq_num = i;
    set_sockperf_header(hdr_buf); 
    add_ip_udp_len(hdr_buf, (RPC_HEADER_LEN - UDP_HEADER_LEN) + curr_pkt_size);
    add_udp_cksum(hdr_buf);
  }

  return rr->hdr_bufs->num_items;
}

__rte_always_inline int 
rr_swap_hdr(struct dpdk_ctx *_dpdk_ctx, struct p2p_rpc_rr *rr)
{
  int ret = -1;
  switch (_dpdk_ctx->mem_alloc_type)
  {
    case MEM_ALLOC_TYPES::HOST_MEM_ONLY:
      //ret = swap_hdrs_on_cpu(rr->transport_mbufs, rr->hdr_bufs);
      ret = swap_sockperf_hdrs_on_cpu(rr->transport_mbufs, rr->hdr_bufs);
      break;
    case MEM_ALLOC_TYPES::DEV_MEM_ONLY:
  //    ret = swap_hdrs_on_gpu(rr->transport_mbufs, rr->hdr_bufs); 
      break;
    case MEM_ALLOC_TYPES::BUFFER_SPLIT:
      //ret = swap_hdrs_on_cpu(rr->transport_mbufs, rr->hdr_bufs);
      ret = swap_sockperf_hdrs_on_cpu(rr->transport_mbufs, rr->hdr_bufs);
      break;
  } 
  return ret;
}

// * Copies from payload into a set of payload_bufs
// * Calculates offsets based on RPC_MTU
// * Returns the number of bytes copied.
__rte_always_inline int 
rr_bufs_to_resp(struct dpdk_ctx *_dpdk_ctx, 
    struct p2p_rpc_rr *rr)
{
  int ret = -1;
  switch (_dpdk_ctx->mem_alloc_type)
  {
    case MEM_ALLOC_TYPES::HOST_MEM_ONLY:
      ret = gather_payload_on_cpu(rr->payload_bufs, rr->resp_payload);
      break;
    case MEM_ALLOC_TYPES::DEV_MEM_ONLY:
      //ret = gather_payload_on_gpu(rr->payload_bufs, rr->resp_payload);
      break;
    case MEM_ALLOC_TYPES::BUFFER_SPLIT:
      ret = gather_payload_on_cpu(rr->payload_bufs, rr->resp_payload);
      break;
  } 
  return ret;
}

// Gathers the bufs into Request Payload
__rte_always_inline int 
rr_bufs_to_req(struct dpdk_ctx *_dpdk_ctx, 
    struct p2p_rpc_rr *rr)
{
  int ret = -1;
  switch (_dpdk_ctx->mem_alloc_type)
  {
    case MEM_ALLOC_TYPES::HOST_MEM_ONLY:
      ret = gather_payload_on_cpu(rr->payload_bufs, rr->req_payload);
      break;
    case MEM_ALLOC_TYPES::DEV_MEM_ONLY:
      //ret = gather_payload_on_gpu(rr->payload_bufs, rr->req_payload);
      break;
    case MEM_ALLOC_TYPES::BUFFER_SPLIT:
      ret = gather_payload_on_cpu(rr->payload_bufs, rr->req_payload);
      break;
  } 
  return ret;
}

__rte_always_inline int 
rr_resp_to_bufs(struct dpdk_ctx *_dpdk_ctx, 
    struct p2p_rpc_rr *rr)
{
  int ret = -1;
  switch (_dpdk_ctx->mem_alloc_type)
  {
    case MEM_ALLOC_TYPES::HOST_MEM_ONLY:
      ret = scatter_payload_on_cpu(rr->payload_bufs, rr->resp_payload, rr->resp_size);
      break;
    case MEM_ALLOC_TYPES::DEV_MEM_ONLY:
      //ret = scatter_payload_on_gpu(rr->payload_bufs, rr->resp_payload, rr->resp_size);
      break;
    case MEM_ALLOC_TYPES::BUFFER_SPLIT:
      ret = scatter_payload_on_cpu(rr->payload_bufs, rr->resp_payload, rr->resp_size);
      break;
  } 
  return ret;
}

__rte_always_inline int 
rr_req_to_bufs(struct dpdk_ctx *_dpdk_ctx, 
    struct p2p_rpc_rr *rr)
{
  int ret = -1;
  switch (_dpdk_ctx->mem_alloc_type)
  {
    case MEM_ALLOC_TYPES::HOST_MEM_ONLY:
      ret = scatter_payload_on_cpu(rr->payload_bufs, rr->req_payload, rr->req_size);
      break;
    case MEM_ALLOC_TYPES::DEV_MEM_ONLY:
      //ret = scatter_payload_on_gpu(rr->payload_bufs, rr->req_payload, rr->req_size);
      break;
    case MEM_ALLOC_TYPES::BUFFER_SPLIT:
      ret = scatter_payload_on_cpu(rr->payload_bufs, rr->req_payload, rr->req_size);
      break;
  } 
  return ret;
}

/**********************************************************************************/

#ifdef __cplusplus
}
#endif
