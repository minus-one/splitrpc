// Copyright (c) Adithya Kumar, The Pennsylvania State University. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "p2p_rpc_conn_info.h"
#include <unordered_map>
#include "p2p_bufs.h"

enum P2P_RPC_RR_STATES { RR_CREATE, RR_PROCESS, RR_READY, RR_FINISH, RR_UNUSED };

// Holder for a request-response
class P2pRpcRr {
  public:
//////////////////// App specific
    uint64_t req_token;
    size_t payload_size;
    size_t max_payload_size;
    //struct p2p_rpc_conn_info *_conn_info; 
    struct p2p_rpc_conn_info _client_conn_info;

///////////////// RR_Pool specific
    uint32_t rr_state; // Internal state defined by P2P_RPC_RR_STATES

///////////////// Transport specific
    // This is common to both RX/TX
    struct p2p_hbufs *transport_mbufs; // Points to the message buffers of the transport used (for DPDK -> rte_mbuf) 
    struct p2p_hbufs *hdr_bufs;
    struct p2p_bufs *payload_bufs;

//////////////// Copy engine specific
    struct p2p_sk_buf *payload_sk_bufs;

///////////////////////////////////// END OF STRUCT /////////////////////////////////////

    P2pRpcRr(struct p2p_rpc_conn_info *template_conn_info = NULL,
        size_t _max_payload_size = 0) {
      transport_mbufs = new p2p_hbufs;
      hdr_bufs = new p2p_hbufs;
      payload_bufs = new p2p_bufs;
      payload_sk_bufs = new p2p_sk_buf;

      rr_state = P2P_RPC_RR_STATES::RR_CREATE;
      if(template_conn_info)
        memcpy((void*)&_client_conn_info.hdr_template, (void*)&template_conn_info->hdr_template, RPC_HEADER_LEN);
      //_conn_info = conn_info;

      // Stats that need to be reset across re-use of rr
      transport_mbufs->num_items = 0;
      hdr_bufs->num_items = 0;
      payload_bufs->num_items = 0;
      payload_size = 0;
      max_payload_size = _max_payload_size;

      req_token = 0;
    }

    ~P2pRpcRr() {
      delete transport_mbufs;
      delete hdr_bufs;
      delete payload_bufs;
      delete payload_sk_bufs;
    }

    inline __attribute__((always_inline)) void rr_merge_mbuf(
        uintptr_t transport_mbuf,
        uintptr_t hdr_buf,
        uintptr_t payload_buf,
        size_t item_size,
        int idx)
    {
      transport_mbufs->burst_items[idx] = transport_mbuf;
      transport_mbufs->num_items++;

      hdr_bufs->burst_items[idx] = hdr_buf;
      hdr_bufs->num_items++;

      payload_bufs->burst_items[idx] = payload_buf;
      payload_bufs->item_size[idx] = item_size;
      payload_bufs->num_items++;
    }

    // Appends a transport mbuf (with hdr and payload info) into this rr's and inc size
    // This does not check the frame-id, but just appends it to the end
    inline __attribute__((always_inline)) void rr_append_mbuf(
        uintptr_t transport_mbuf,
        uintptr_t hdr_buf,
        uintptr_t payload_buf,
        size_t item_size)
    {
      rr_merge_mbuf(transport_mbuf, hdr_buf, payload_buf, item_size, transport_mbufs->num_items);
      payload_size += item_size;
    }

    // Places a transport mbuf (with hdr and payload info) into this rr's and inc size
    inline __attribute__((always_inline)) void rr_emplace_mbuf(
        uintptr_t transport_mbuf,
        uintptr_t hdr_buf,
        uintptr_t payload_buf,
        size_t item_size,
        int seq_num)
    {
      rr_merge_mbuf(transport_mbuf, hdr_buf, payload_buf, item_size, seq_num);
      payload_size += item_size;
    }

    inline __attribute__((always_inline)) bool is_rr_ready()
    {
      return payload_size == max_payload_size;
    }

    void rr_print_bufs()
    {
      printf("==========================================================================\n");
      printf("Req: %p, num_bufs: %d\n", (void*)this, payload_bufs->num_items);

      for(int i = 0 ; i < payload_bufs->num_items ; i++) {
        void *start_addr = (void*)payload_bufs->burst_items[i];
        size_t item_size = payload_bufs->item_size[i];
        void *end_addr = (void*)((uint8_t*)start_addr + item_size);
        int seq_num = (((struct p2p_rpc_hdr*)hdr_bufs->burst_items[i])->seq_num);
        printf("Req: %p, start_addr: %p, end_addr: %p, seq_num: %d\n", 
            (void*)this, start_addr, end_addr, seq_num);
      }
      printf("==========================================================================\n");
    }
};

// Clones an rr including conn_info - does not copy data/bufs
static inline P2pRpcRr*
rr_clone_alloc(const P2pRpcRr *rr)
{
  P2pRpcRr *new_rr = new P2pRpcRr();
  new_rr->max_payload_size = rr->max_payload_size;
  new_rr->req_token = 0;
  memcpy((void*)&new_rr->_client_conn_info.hdr_template, (void*)&rr->_client_conn_info.hdr_template, RPC_HEADER_LEN);
  //new_rr->_conn_info = rr->_conn_info;

  return new_rr;
}


