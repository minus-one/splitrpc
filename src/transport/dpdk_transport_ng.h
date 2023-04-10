// Copyright (c) Adithya Kumar, The Pennsylvania State University. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "p2p_rpc.h"
#include "p2p_rpc_conn_info.h"
#include "dpdk_utils.h"
#include "dpdk_init.h"
#include "dpdk_rx_tx.h"
#include "p2p_rpc_rr_ng.h"
#include "p2p_rpc_rr_pool_ng.h"

#ifdef __cplusplus
extern "C" { 
#endif

/***************************** RAW transport APIs ************************************/
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
/************************Transport helper APIs implemented in CPU*************************/

inline int 
swap_hdrs_on_cpu(struct p2p_hbufs*, 
    struct p2p_hbufs *hdr, size_t payload_size)
{
  //struct rte_mbuf **mbufs = (struct rte_mbuf**)transport_mbufs->burst_items;
  // FIXME: Can this be vectorized or optimized?
  for (int i = 0; i < hdr->num_items; i++)
  {
    struct p2p_rpc_hdr *pkt_hdr = (struct p2p_rpc_hdr*)(hdr->burst_items[i]);
    swap_eth_hdr(pkt_hdr);
    add_ip_udp_len(pkt_hdr, (RPC_HEADER_LEN - UDP_HEADER_LEN) + payload_size);
    add_udp_cksum(pkt_hdr);
  }
  return 1;
}

inline int 
swap_sockperf_hdrs_on_cpu(struct p2p_hbufs *transport_mbufs, 
    struct p2p_hbufs *hdr, size_t payload_size)
{
  struct rte_mbuf **mbufs = (struct rte_mbuf**)transport_mbufs->burst_items;
  // FIXME: Can this be vectorized or optimized?
  for (int i = 0; i < hdr->num_items; i++)
  {
    struct p2p_rpc_hdr *pkt_hdr = (struct p2p_rpc_hdr*)(hdr->burst_items[i]);
    swap_eth_hdr(pkt_hdr);
    mbufs[i]->ol_flags |= ( PKT_TX_IPV4 | PKT_TX_IP_CKSUM | PKT_TX_UDP_CKSUM);
    add_ip_udp_len(pkt_hdr, (RPC_HEADER_LEN - UDP_HEADER_LEN) + payload_size);

    struct udp_hdr *udp_h = get_udp_header(pkt_hdr);
    struct ipv4_hdr *ip_h = get_ip_header(pkt_hdr);
    udp_h->check = 0;
    udp_h->check = rte_ipv4_phdr_cksum((struct rte_ipv4_hdr*)ip_h, mbufs[i]->ol_flags);
    set_sockperf_header(pkt_hdr); 
  }
  return 1;
}

// Assumes mbufs have been allocated
// Returns number of header packets
// Assumes the header resides on CPU memory (HOST_MEM or BUFFER_SPLIT)
// This can take any conn_info - if this is a response, you will pass in the rr's _client_conn_info
// If this is a new request, you can send in whatever conn_info you want to send it to
__rte_always_inline int 
rr_set_hdr(
    struct p2p_rpc_conn_info *_tx_conn_info,
    P2pRpcRr *rr, 
    size_t payload_size)
{
  struct p2p_rpc_hdr *hdr_buf = NULL;
  size_t curr_pkt_size = 0;

  for(int i = 0; i < rr->hdr_bufs->num_items; i++) {
    curr_pkt_size = (payload_size <= RPC_MTU) ? payload_size : RPC_MTU;
    payload_size -= curr_pkt_size;

    // Set header from conn for each mbuf
    hdr_buf = (struct p2p_rpc_hdr*)(rr->hdr_bufs->burst_items[i]); 
    memcpy((uint8_t*)hdr_buf, 
        (uint8_t*) &(_tx_conn_info->hdr_template), 
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
rr_swap_hdr(struct dpdk_ctx *_dpdk_ctx, P2pRpcRr *rr, size_t payload_size)
{
  int ret = -1;
  switch (_dpdk_ctx->mem_alloc_type)
  {
    case MEM_ALLOC_TYPES::HOST_MEM_ONLY:
      //ret = swap_hdrs_on_cpu(rr->transport_mbufs, rr->hdr_bufs);
      ret = swap_sockperf_hdrs_on_cpu(rr->transport_mbufs, rr->hdr_bufs, payload_size);
      for(int i = 0 ; i < rr->transport_mbufs->num_items; i++) {
        struct rte_mbuf *mbuf = (struct rte_mbuf*)(rr->transport_mbufs->burst_items[i]);
        update_dpdk_mbuf_len(mbuf, RPC_HEADER_LEN + payload_size);
      }
      break;
    case MEM_ALLOC_TYPES::DEV_MEM_ONLY:
      //ret = swap_hdrs_on_gpu(rr->transport_mbufs, rr->hdr_bufs); 
      break;
    case MEM_ALLOC_TYPES::BUFFER_SPLIT:
      //ret = swap_hdrs_on_cpu(rr->transport_mbufs, rr->hdr_bufs, payload_size);
      ret = swap_sockperf_hdrs_on_cpu(rr->transport_mbufs, rr->hdr_bufs, payload_size);
      for(int i = 0 ; i < rr->transport_mbufs->num_items; i++) {
        struct rte_mbuf *mbuf = (struct rte_mbuf*)(rr->transport_mbufs->burst_items[i]);
        update_dpdk_mbuf_len(mbuf->next, payload_size);
      }
      break;
  } 
  return ret;
}

/************************** APIs for rpc_rr management ******************************/

// Not multi-thread safe
// Use this only if there is split buffer allocation, and header will not be needed (i.e. not zero copy)
__rte_always_inline int
rr_recv_ng(struct dpdk_ctx *_dpdk_ctx, P2pRpcRRPool *rr_pool, volatile bool& cancel)
{
  //bool got_rr = true;
  P2pRpcRr *call_id = NULL;
  void *payload_addr;
  size_t payload_len;
  int seq_num;
  uint16_t nb = 0;

  struct rte_mbuf **bufs = (struct rte_mbuf**)_dpdk_ctx->dpdk_mbufs->burst_items;
  nb = rx_pkts(_dpdk_ctx, bufs);

  for(int i = 0 ; i < nb; i++) {
    // Parse header
    struct p2p_rpc_hdr *rpc_hdr = rte_pktmbuf_mtod(bufs[i], struct p2p_rpc_hdr *);
    payload_addr = rte_pktmbuf_mtod(bufs[i]->next, void*);
    payload_len = bufs[i]->pkt_len - RPC_HEADER_LEN;
    seq_num = rpc_hdr->seq_num;

    if(likely(rpc_hdr->req_token != 0)) {
      // Allocate and Collect info 
      call_id = rr_pool->get_rr(rpc_hdr);
      // Save only the payload info
      call_id->rr_emplace_mbuf(
          (uintptr_t)(bufs[i]->next), (uintptr_t)0, 
          (uintptr_t)payload_addr, payload_len, seq_num);
      // Release header 
      bufs[i]->next = NULL;
      rte_pktmbuf_free(bufs[i]);

      if(call_id->is_rr_ready()) {
        // At this point we know that whole request is received
        rr_pool->mark_rx_ready_rr(call_id);
        TRACE_PRINTF("New Req: %p is ready\n", (void*)call_id);
      }
    } else {
      // Invalid packet, release everything
      rte_pktmbuf_free(bufs[i]);
    }
  }
  return 1;
}

__rte_always_inline int
rr_recv(struct dpdk_ctx *_dpdk_ctx, P2pRpcRRPool *rr_pool, volatile bool& cancel)
{
  struct p2p_hbufs *dpdk_mbufs = _dpdk_ctx->dpdk_mbufs; 
  struct p2p_hbufs *hdr_bufs = _dpdk_ctx->hdr_bufs;
  struct p2p_bufs *payload_bufs = _dpdk_ctx->payload_bufs;
  P2pRpcRr *call_id = NULL;

  get_requests_zc(_dpdk_ctx, dpdk_mbufs, hdr_bufs, payload_bufs);

  for(int i = 0 ; i < dpdk_mbufs->num_items; i++) {
    struct p2p_rpc_hdr *rpc_hdr = 
      (struct p2p_rpc_hdr*)(hdr_bufs->burst_items[i]);

    //if(likely(rpc_hdr->req_token != 0 && rpc_hdr->seq_num >= 0 && rpc_hdr->seq_num < MAX_BI_SIZE)) {
    if(likely(rpc_hdr->req_token != 0)) {
      call_id = rr_pool->get_rr(rpc_hdr);
      call_id->rr_emplace_mbuf(
          dpdk_mbufs->burst_items[i], hdr_bufs->burst_items[i], 
          payload_bufs->burst_items[i], payload_bufs->item_size[i],
          rpc_hdr->seq_num);
      if(call_id->is_rr_ready()) {
        // At this point we know that whole request is received
        rr_pool->mark_rx_ready_rr(call_id);
        TRACE_PRINTF("New Req: %p is ready\n", (void*)call_id);
      }
    } else {
      release_dpdk_mbufs((struct rte_mbuf**)&dpdk_mbufs->burst_items[i], 1);
    }
  }

  return 1;
  }

//__rte_always_inline int
//get_next_rr(P2pRpcRRPool *rr_pool, P2pRpcRr **new_call_id)
//{
//  *new_call_id = rr_pool->consume_rx_ready_rr();
//  if(*new_call_id != NULL)
//    return 1;
//
//  return 0;
//}

// This is basically like a blocking recvfrom()
// Not multi-thread safe
__rte_always_inline int
rr_recv_sync(struct dpdk_ctx *_dpdk_ctx,
    P2pRpcRr **new_call_id, P2pRpcRRPool &rr_pool, volatile bool& cancel)
{
  bool got_rr = false;
  struct p2p_hbufs *dpdk_mbufs = _dpdk_ctx->dpdk_mbufs; 
  struct p2p_hbufs *hdr_bufs = _dpdk_ctx->hdr_bufs;
  struct p2p_bufs *payload_bufs = _dpdk_ctx->payload_bufs;
  P2pRpcRr *call_id = NULL;

  while(!got_rr && ACCESS_ONCE(cancel) == 0) {
    while(get_requests_zc(_dpdk_ctx, dpdk_mbufs, hdr_bufs, payload_bufs) == 0 
        && ACCESS_ONCE(cancel) == 0);

    for(int i = 0 ; i < dpdk_mbufs->num_items; i++) {
      struct p2p_rpc_hdr *rpc_hdr = 
        (struct p2p_rpc_hdr*)(hdr_bufs->burst_items[i]);

      if(likely(rpc_hdr->seq_num >= 0 && rpc_hdr->seq_num < MAX_BI_SIZE)) {
        call_id = rr_pool.get_rr(rpc_hdr);
        call_id->rr_emplace_mbuf(
            dpdk_mbufs->burst_items[i], hdr_bufs->burst_items[i], 
            payload_bufs->burst_items[i], payload_bufs->item_size[i],
            rpc_hdr->seq_num);
        if(call_id->is_rr_ready()) {
          // At this point we know that whole request is received
          rr_pool.mark_rx_ready_rr(call_id);
          got_rr = true;
          TRACE_PRINTF("New Req: %p is ready\n", (void*)call_id);
        }
      } else {
        release_dpdk_mbufs((struct rte_mbuf**)&dpdk_mbufs->burst_items[i], 1);
      }
    }
  }

  *new_call_id = rr_pool.consume_rx_ready_rr();
  if(*new_call_id != NULL)
    return 1;

  return 0;
}

// Assumes payload has been transferred to transport-mbufs
// Calls the transport APIs to send the packet
__rte_always_inline int
rr_send(struct dpdk_ctx *_dpdk_ctx,
    P2pRpcRr *rr, P2pRpcRRPool &rr_pool)
{
  int nb_tx = send_requests_zc(_dpdk_ctx, rr->transport_mbufs);
  // Reap the rr
  if (nb_tx == rr->hdr_bufs->num_items)
  {
    rr->payload_size = 0;
    rr_pool.reap_rr(rr); 
    return 1;
  }
  return 0;
}

__rte_always_inline int
rr_alloc_mbufs(struct dpdk_ctx *_dpdk_ctx,
    P2pRpcRr *rr,
    int payload_size)
{
  struct rte_mbuf **bufs = (struct rte_mbuf**)rr->transport_mbufs->burst_items;
  int new_bufs = (payload_size + RPC_MTU - 1) / RPC_MTU;
  rr->transport_mbufs->num_items = alloc_dpdk_mbufs(_dpdk_ctx, bufs, new_bufs, payload_size);
  pkts_to_buf_ptrs(_dpdk_ctx, rr->transport_mbufs, rr->hdr_bufs, rr->payload_bufs);
  rr->payload_size = payload_size;
  return rr->transport_mbufs->num_items;
}

__rte_always_inline void
rr_release_mbufs(struct dpdk_ctx *,
    P2pRpcRr *rr)
{
  struct rte_mbuf **bufs = (struct rte_mbuf**)rr->transport_mbufs->burst_items;
  release_dpdk_mbufs(bufs, rr->transport_mbufs->num_items);
  rr->transport_mbufs->num_items = 0;
  rr->hdr_bufs->num_items = 0;
  rr->payload_bufs->num_items = 0;
  rr->payload_size = 0;
}

__rte_always_inline void
rr_realloc_mbufs(struct dpdk_ctx *_dpdk_ctx,
    P2pRpcRr *rr, int new_payload_size)
{
  int old_bufs = rr->transport_mbufs->num_items;
  int new_bufs = (new_payload_size + RPC_MTU - 1) / RPC_MTU;

  if(new_bufs < old_bufs) {
    struct rte_mbuf **bufs = (struct rte_mbuf**)rr->transport_mbufs->burst_items;
    int bufs_to_be_released = old_bufs - new_bufs;
    release_dpdk_mbufs(bufs+new_bufs, bufs_to_be_released);
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

/**********************************************************************************/
#ifdef __cplusplus
}
#endif
