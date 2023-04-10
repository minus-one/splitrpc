// Copyright (c) Adithya Kumar, The Pennsylvania State University. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "p2p_rpc.h"
#include "dpdk_utils.h"
#include "p2p_bufs.h"
#include "p2p_rpc_rr.h"

#ifndef GPU_DISABLED
#include "gdr_mem_manager.h"
#endif

#define RX_RING_SIZE 8192
#define TX_RING_SIZE 8192
// 128 MB max with MTU=8192
#define NUM_MBUFS (1 << 16)
#define MBUF_CACHE_SIZE 250

#define MAX_RX_QUEUE_PER_PORT 1
#define MAX_TX_QUEUE_PER_PORT 1

#ifndef GPU_DISABLED
const size_t DEVICE_PG_SZ = GPU_PAGE_SIZE;
#else
const size_t DEVICE_PG_SZ = RTE_PGSIZE_4K;
#endif

// Contains necessary information about a
// memory segment registered with DPDK
struct dpdk_memseg_info {
    struct rte_pktmbuf_extmem *_ext_mem;
    struct rte_mempool *_mbuf_pool;
    union memsg_info  {
#ifndef GPU_DISABLED
      struct gdr_memseg_info *gdr_memsg;
#endif
      const struct rte_memzone *mz_info;
    }_alloc_info;
    size_t num_mbufs;
    uint16_t elt_size; // Size of buffer. Include RTE_PKTMBUF_HEADROOM in your calculation separately
    int buf_loc; // Location of the buffer -2 => CPU buffer, -1 => CPU CUDA Pinned buffer, >=0 => GPU-x GDR memory
    char memseg_name[RTE_MEMZONE_NAMESIZE];
    size_t pgsz;
};

// Global DPDK info that is returned by setup DPDK
struct dpdk_ctx {
  /* Client provided details */
    int nic_port;
    int queue_id;
    int mem_alloc_type;
    int zerocopy_mode;
    int device_id; // = -1 => CPU, >= 0 => GPU device-id
  /* End of client provided details */
    int nsegs;
    struct dpdk_memseg_info *memseg_info; // Array of nsegs

    //struct rte_ring *init_ring; // Ring for recycling RPC_WIs
    //struct rte_ring *work_ring; // Ring for RX -> Work
    //struct rte_ring *tx_ring; // Ring for Echo -> TX
    //struct rte_ring *echo_ring; // Ring for Work -> Echo

    // Used to get requests
    struct p2p_hbufs *dpdk_mbufs;
    struct p2p_hbufs *hdr_bufs;
    struct p2p_bufs *payload_bufs;

    // One burst might trigger multiple requests
    // We use this to store all of them
    // This is a LIFO for now
    // FIXME: Change this to a FIFO
    p2p_rpc_rr *ready_rr[MAX_BI_SIZE];
    int num_ready_rrs;
};

#ifdef __cplusplus
extern "C" { 
#endif
 
int init_dpdk_ctx(struct dpdk_ctx *ctx);
void stop_dpdk(struct dpdk_ctx *ctx);

#ifdef __cplusplus
}
#endif
