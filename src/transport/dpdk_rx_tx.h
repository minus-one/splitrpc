// Copyright (c) Adithya Kumar, The Pennsylvania State University. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include "p2p_rpc.h"
#include "p2p_rpc_conn_info.h"
#include "p2p_bufs.h"
#include "dpdk_utils.h"
#include "dpdk_init.h"
#include "utils/debug_utils.h"

#ifdef __cplusplus
extern "C" { 
#endif

// Converts an address to mbuf
__rte_always_inline rte_mbuf* 
pktmbuf_atom(uintptr_t addr) {
  return reinterpret_cast<rte_mbuf *>((uint8_t*)addr - RTE_PKTMBUF_HEADROOM -
                                              sizeof(rte_mbuf));
}

// Converts a data-ptr to mbuf
__rte_always_inline rte_mbuf* 
pktmbuf_dtom(uint8_t* addr) {
  return reinterpret_cast<rte_mbuf *>(addr - RTE_PKTMBUF_HEADROOM -
                                              sizeof(rte_mbuf));
}

__rte_always_inline int 
tx_pkts_internal(int nic_port, int queue_id, struct rte_mbuf **bufs, int nb)
{
  //for(int i = 0 ; i < nb;i++) { 
  //  struct p2p_rpc_hdr *rpc_hdr = rte_pktmbuf_mtod(bufs[i], struct p2p_rpc_hdr*);
  //  uint8_t *rpc_payload = rte_pktmbuf_mtod_offset(bufs[i], uint8_t*, RPC_HEADER_LEN);
  //  hexDump("RPC Header", rpc_hdr, RPC_HEADER_LEN);
  //  hexDump("RPC Payload", rpc_payload, bufs[i]->pkt_len - RPC_HEADER_LEN); 
  //}

  uint16_t nb_tx = rte_eth_tx_burst(nic_port, queue_id, bufs, nb);
  if (unlikely(nb_tx != nb)) {
    uint8_t retry_count = 0;
    while (nb_tx != nb) {
      nb_tx += rte_eth_tx_burst(nic_port, queue_id, &bufs[nb_tx],
          nb - nb_tx);
      retry_count++;
      if (unlikely(retry_count == 100)) {
        printf("Stuck in tx_burst for %d/%d pkts\n", nb_tx, nb);
        retry_count = 0;
        break;
      }
    }
    /* Free any unsent packets. */
    if (nb_tx < nb)
      rte_pktmbuf_free_bulk(&bufs[nb_tx], nb - nb_tx);
  } 
  return nb_tx;
}

__rte_always_inline  int
tx_pkts(
    struct dpdk_ctx *_dpdk_ctx,
    struct rte_mbuf **bufs,
    int num_bufs)
{
  return tx_pkts_internal(_dpdk_ctx->nic_port, _dpdk_ctx->queue_id, bufs, num_bufs);
}

__rte_always_inline int 
rx_pkts(struct dpdk_ctx *_dpdk_ctx,
    struct rte_mbuf **bufs)
{
  uint16_t nb = 0;
  nb = rte_eth_rx_burst(_dpdk_ctx->nic_port, _dpdk_ctx->queue_id, bufs, MAX_BI_SIZE);
  if (unlikely(nb == 0)) {
    return 0;
  }
  
  uint16_t nb_rx = 0;
  for (int i = 0; i < nb; i++)
  {
    // FIXME: Filter out non p2p-rpc packets more cleanly
    // This would actually trigger a bug where bufs and other wi don't match
    // We can actually have two ptrs and do it cleanly. Fix it.
    // This is the only way to not look at the data and do something
    if(unlikely(bufs[i]->pkt_len < RPC_HEADER_LEN)) {
      rte_pktmbuf_free(bufs[i]); // Drop packet
      printf("Dropping non-p2p-rpc packet\n");
      continue;
    }

    //if(rte_be_to_cpu_16(get_eth_header(rx_pkts[nb_rx])->eth_type_) != RTE_ETHER_TYPE_IPV4)
    //  continue;

    //if(get_ip_header(rx_pkts[nb_rx])->protocol != IPPROTO_UDP)
    //  continue;

    bufs[nb_rx++] = bufs[i];
  }
  return nb_rx;
}

__rte_always_inline void 
update_dpdk_mbuf_len(
    struct rte_mbuf *buf,
    size_t pkt_len)
{
  buf->pkt_len = pkt_len;
  buf->data_len = pkt_len;
}

// Allocates MBUFs for a payload
// The size and RPC_MTU determine the number of MBUFs 
// and is allocated as per the memory allocation type
__rte_always_inline int
alloc_dpdk_mbufs(
    struct dpdk_ctx *_dpdk_ctx,
    struct rte_mbuf **bufs,
    int new_bufs,
    size_t payload_len)
{
  size_t rem_len = payload_len;
  TRACE_PRINTF("Allocating %d new bufs, payload_len: %ld, RPC_MTU: %ld\n", new_bufs, payload_len, RPC_MTU);
  struct rte_mbuf *payload_mbufs[MAX_BI_SIZE];
  // Allocate mbufs for headers
  if(unlikely(rte_pktmbuf_alloc_bulk(_dpdk_ctx->memseg_info[0]._mbuf_pool, 
          bufs, new_bufs) != 0))
    return -1;
  // Allocate mbufs for payload (if BUFFER_SPLIT)
  if(_dpdk_ctx->mem_alloc_type == MEM_ALLOC_TYPES::BUFFER_SPLIT) {
    if(unlikely(rte_pktmbuf_alloc_bulk(_dpdk_ctx->memseg_info[1]._mbuf_pool, 
            payload_mbufs, new_bufs) != 0))
      return -1;
  }

  // Update the len and other fields
  for(int j = 0; j < new_bufs; j++) {
    int curr_pkt_size = (rem_len <= RPC_MTU) ? rem_len : RPC_MTU;
    rem_len -= curr_pkt_size;
    struct rte_mbuf *new_buf = bufs[j];

    // Flag NIC to do the check-sum
    new_buf->ol_flags |= ( PKT_TX_IPV4 | PKT_TX_IP_CKSUM | PKT_TX_UDP_CKSUM);

    new_buf->l2_len = sizeof(struct eth_hdr);
    new_buf->l3_len = sizeof(struct eth_hdr) + sizeof(struct ipv4_hdr);

    if(_dpdk_ctx->mem_alloc_type == MEM_ALLOC_TYPES::BUFFER_SPLIT) {
      update_dpdk_mbuf_len(new_buf, RPC_HEADER_LEN);
      update_dpdk_mbuf_len(payload_mbufs[j], curr_pkt_size);
      //new_buf->pkt_len = RPC_HEADER_LEN; 
      //new_buf->data_len = RPC_HEADER_LEN;
      //payload_mbufs[j]->pkt_len = curr_pkt_size;
      //payload_mbufs[j]->data_len = curr_pkt_size;

      // Append the split buffer
      rte_pktmbuf_chain(new_buf, payload_mbufs[j]);
    } else {
      update_dpdk_mbuf_len(new_buf, RPC_HEADER_LEN + curr_pkt_size);
      //new_buf->pkt_len = RPC_HEADER_LEN + curr_pkt_size;
      //new_buf->data_len = RPC_HEADER_LEN + curr_pkt_size;
    }
  }
  return new_bufs;
}

__rte_always_inline int
release_dpdk_mbufs(
    struct rte_mbuf **bufs,
    int num_mbufs) 
{
  rte_pktmbuf_free_bulk(bufs, num_mbufs);
  return num_mbufs;
}

__rte_always_inline int
pkts_to_buf_ptrs(struct dpdk_ctx *_dpdk_ctx,
    struct p2p_hbufs *dpdk_mbufs,
    struct p2p_hbufs *hdr_bufs,
    struct p2p_bufs *payload_bufs)
{
  struct rte_mbuf **bufs = (struct rte_mbuf**)dpdk_mbufs->burst_items;
  for (int i = 0; i < dpdk_mbufs->num_items; i++)
  {
    void *hdr_addr = NULL, *payload_addr = NULL;
    switch (_dpdk_ctx->mem_alloc_type)
    {
      case MEM_ALLOC_TYPES::HOST_MEM_ONLY:
        hdr_addr = rte_pktmbuf_mtod(bufs[i], struct p2p_rpc_hdr *);
        payload_addr = rte_pktmbuf_mtod_offset(bufs[i], void *, RPC_HEADER_LEN);
#ifdef BUILD_ON_KEPLER
        // On kepler, the host addr and device addr do not match. So, we need to specifically get the mapping, to use on the device
        void *d_addr = NULL;
        cudaHostGetDevicePointer((void**)&d_addr, payload_addr, 0);
        payload_addr = d_addr;
#endif
        break;
      case MEM_ALLOC_TYPES::DEV_MEM_ONLY:
        hdr_addr = rte_pktmbuf_mtod(bufs[i], struct p2p_rpc_hdr *);
        payload_addr = rte_pktmbuf_mtod_offset(bufs[i], void *, RPC_HEADER_LEN);
        break;
      case MEM_ALLOC_TYPES::BUFFER_SPLIT:
        hdr_addr = rte_pktmbuf_mtod(bufs[i], struct p2p_rpc_hdr *);
        payload_addr = rte_pktmbuf_mtod(bufs[i]->next, void*);
        break;
    }
    hdr_bufs->burst_items[i] = (uintptr_t)hdr_addr;
    payload_bufs->burst_items[i] = (uintptr_t)payload_addr;
    payload_bufs->item_size[i] = bufs[i]->pkt_len - RPC_HEADER_LEN; 

    TRACE_PRINTF("BI: %d, h_start: %p, h_end: %p, h_len: %ld, p_start: %p, p_end: %p, p_len: %ld\n",
        i, hdr_addr, ((char*)hdr_addr + RPC_HEADER_LEN), RPC_HEADER_LEN, 
        payload_addr, ((char*)payload_addr + payload_bufs->item_size[i]), payload_bufs->item_size[i]);
  }
  hdr_bufs->num_items = dpdk_mbufs->num_items;
  payload_bufs->num_items = dpdk_mbufs->num_items;
  return dpdk_mbufs->num_items;
}

#ifdef __cplusplus
}
#endif
