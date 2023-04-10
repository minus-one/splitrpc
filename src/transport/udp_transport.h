// Copyright (c) Adithya Kumar, The Pennsylvania State University. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "udp_rx_tx.h"
#include "udp_rr.h"

//#include "/usr/include/mellanox/vma_extra.h"

#ifndef likely
#define likely(x)       __builtin_expect((x),1)
#define unlikely(x)     __builtin_expect((x),0)
#endif

inline int
udp_rr_recv_resp(int udp_sock,
    UdpRr **new_call_id, UdpRrPool *rr_pool)
{
  UdpRr *new_rr = NULL;
  uint8_t *resp_buf = new uint8_t[UDP_RPC_MAX_MTU];
  struct sockaddr_in si_other;
  int resp_buf_len = getData(udp_sock, &si_other, resp_buf, UDP_RPC_MAX_MTU);
  if(resp_buf_len > 0) {
    UdpRpcHdr *rpc_hdr = (UdpRpcHdr*)resp_buf;
    if(likely(rpc_hdr->seq_num >= 0 && rpc_hdr->seq_num < MAX_BI_SIZE)) {
      new_rr = rr_pool->get_rr(rpc_hdr);
      new_rr->si_other = si_other;
      new_rr->rr_merge_resp_buf((uintptr_t)resp_buf, size_t(resp_buf_len), rpc_hdr->seq_num);
      if(new_rr->is_req_ready()) {
        rr_pool->mark_ready_rr(new_rr);
        TRACE_PRINTF("New Resp: %p is ready\n", (void*)new_rr);
      }
    } else {
      delete resp_buf;
    }
  } else {
    delete resp_buf;
  }
  *new_call_id = rr_pool->consume_ready_rr();
  if(*new_call_id != NULL)
    return 1;
  return 0;
}

inline int
udp_rr_recv_req(int udp_sock,
    UdpRr **new_call_id, UdpRrPool *rr_pool)
{
  UdpRr *new_rr = NULL;
  uint8_t *req_buf = new uint8_t[UDP_RPC_MAX_MTU];
  struct sockaddr_in si_other;
  int64_t req_buf_len = getData(udp_sock, &si_other, req_buf, UDP_RPC_MAX_MTU);
  if(req_buf_len > 0) {
    UdpRpcHdr *rpc_hdr = (UdpRpcHdr*)req_buf;
    TRACE_PRINTF("Got pkt, token: %ld, size: %ld\n", rpc_hdr->req_token, req_buf_len);
    if(likely(rpc_hdr->seq_num >= 0 && rpc_hdr->seq_num < MAX_BI_SIZE)) {
      new_rr = rr_pool->get_rr(rpc_hdr);
      new_rr->si_other = si_other;
      new_rr->rr_merge_req_buf((uintptr_t)req_buf, req_buf_len, rpc_hdr->seq_num);
      if(new_rr->is_req_ready()) {
        rr_pool->mark_ready_rr(new_rr);
        TRACE_PRINTF("New req: %p is ready\n", (void*)new_rr);
      } else {
        TRACE_PRINTF("req: %p has got req_size: %ld\n", (void*)new_rr, new_rr->req_size);
      }
    } else {
      delete req_buf;
    }
  } else {
    delete req_buf;
  }
  *new_call_id = rr_pool->consume_ready_rr();
  if(*new_call_id != NULL)
    return 1;
  return 0;
}

void
udp_rr_recv_req_listener(int udp_sock,
    UdpRrPool *rr_pool, volatile bool &force_quit)
{
  UdpRr *new_rr = NULL;
  uint8_t *req_buf = new uint8_t[UDP_RPC_MAX_MTU];
  struct sockaddr_in si_other;

  printf("Starting REQ Listener\n");
  while(ACCESS_ONCE(force_quit) == 0)
  {
    int64_t req_buf_len = getData(udp_sock, &si_other, req_buf, UDP_RPC_MAX_MTU);
    if(req_buf_len > 0) {
      UdpRpcHdr *rpc_hdr = (UdpRpcHdr*)req_buf;
      //TRACE_PRINTF("Got pkt, token: %ld, size: %ld\n", rpc_hdr->req_token, req_buf_len);
      if(likely(rpc_hdr->seq_num >= 0 && rpc_hdr->seq_num < MAX_BI_SIZE)) {
        new_rr = rr_pool->get_rr(rpc_hdr);
        new_rr->si_other = si_other;
        new_rr->rr_merge_req_buf((uintptr_t)req_buf, req_buf_len, rpc_hdr->seq_num);
        if(new_rr->is_req_ready()) {
          //rr_pool->mark_ready_rr(new_rr);
          rr_pool->mark_process_rr(new_rr);
          TRACE_PRINTF("New rr: %p, idx: %ld is ready\n", (void*)new_rr, new_rr->rr_idx);
        } 
        req_buf = new uint8_t[UDP_RPC_MAX_MTU];
      } 
    } 
  }
  printf("Stopping REQ listener\n");
}

void
udp_rr_recv_req_processor(UdpRrPool *rr_pool, volatile bool &force_quit)
{
  UdpRr *next_rr = NULL;

  printf("Starting REQ Processor\n");
  while(ACCESS_ONCE(force_quit) == 0)
  {
    next_rr = rr_pool->consume_process_rr();
    if(next_rr) {
      next_rr->bufs_to_req();
      next_rr->release_req_bufs();
      rr_pool->mark_ready_rr(next_rr);
    }
  }
  printf("Stopping REQ Processor\n");
}

//inline int
//udp_rr_zc_recv_req(int udp_sock,
//    UdpRrPool *rr_pool, volatile &force_quit)
//{
//  uint8_t *tmp_buf = new uint8_t[UDP_RPC_MAX_MTU];
//  sturct sockaddr_in si_other;
//  socklen_t slen = sizeof(si_other);
//  int64_t req_buf_len;
//  int flags = MSG_VMA_ZCOPY_FORCE;
//
//  while(ACCESS_ONCE(force_quit) == 0)
//  {
//    req_buf_len = recvfrom_zcopy(udp_sock, tmp_buf, UDP_RPC_MAX_MTU, &flags, &si_other, &slen);
//    if(flags & MSG_VMA_ZCOPY) {
//      vma_packets_t *t_rx_pkts = (vma_packets_t*)tmp_buf;
//      for(int i = 0 ; i < t_rx_pkts->n_packet_num; i++) {
//        vma_packet_t *t_pkt = &t_rx_pkts->pkts[i];
//        UdpRpcHdr *rpc_hdr = (UdpRpcHdr*)t_pkt->iov[0]->iov_base;
//        size_t buf_len = 
//      }
//    }
//  }
//}


inline int
udp_rr_send_req(int udp_sock, 
    UdpRr *rr)
{
  for(int i = 0 ; i < rr->req_bufs->num_items; i++) {
    sendData(udp_sock, &rr->si_other, (void*)rr->req_bufs->burst_items[i], rr->req_bufs->item_size[i], UDP_RPC_MAX_MTU);
  }
  return rr->req_size;
}

inline int
udp_rr_send_resp(int udp_sock, 
    UdpRr *rr)
{
  for(int i = 0 ; i < rr->resp_bufs->num_items; i++) {
    sendData(udp_sock, &rr->si_other, (void*)rr->resp_bufs->burst_items[i], rr->resp_bufs->item_size[i], UDP_RPC_MAX_MTU);
  }
  return rr->resp_size;
}
