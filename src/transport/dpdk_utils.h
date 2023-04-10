// Copyright (c) Adithya Kumar, The Pennsylvania State University. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include <stdint.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <inttypes.h>
#include <signal.h>

#include <rte_eal.h>
#include <rte_ethdev.h>
#include <rte_cycles.h>
#include <rte_lcore.h>
#include <rte_mbuf.h>
#include <rte_config.h>
#include <rte_byteorder.h>
#include <rte_memory.h>
#include <rte_memzone.h>
#include <rte_launch.h>
#include <rte_tailq.h>
#include <rte_per_lcore.h>
#include <rte_debug.h>
#include <rte_ether.h>
#include <rte_ip.h>
#include <rte_udp.h>
#include <rte_branch_prediction.h>
#include <rte_string_fns.h>
#include <rte_log.h>
#include <rte_ring.h>

#include "utils/config_utils.h"
#include "utils/debug_utils.h"

typedef enum MEM_ALLOC_TYPES {
	HOST_MEM_ONLY = 0,
	DEV_MEM_ONLY = 1 << 0,
	BUFFER_SPLIT = 1 << 1
} MEM_ALLOC_TYPES;

static inline uint16_t get_dpdk_port() {
    return readEnvInfo<uint16_t>("P2P_RPC_DPDK_PORT", 0);
}

static inline int get_dpdk_mem_alloc_type() {
    return readEnvInfo<int>("P2P_RPC_DPDK_MEM_ALLOC_TYPE", MEM_ALLOC_TYPES::HOST_MEM_ONLY);
}

static void print_rx_offloads(uint64_t offloads)
{
	uint64_t single_offload;
	int begin;
	int end;
	int bit;

	if (offloads == 0)
		return;

	begin = __builtin_ctzll(offloads);
	end = sizeof(offloads) * CHAR_BIT - __builtin_clzll(offloads);

	single_offload = 1ULL << begin;
	for (bit = begin; bit < end; bit++)
	{
		if (offloads & single_offload)
			printf(" %s",
				   rte_eth_dev_rx_offload_name(single_offload));
		single_offload <<= 1;
	}
}

/*
 * We can't include arpa/inet.h because our compiler options are too strict
 * for that shitty code. Thus, we have to do this here...
 * ^^
 * Very bossy comment
 */
static void print_pkt(int src_ip, int dst_ip, uint16_t src_port, uint16_t dst_port, int len)
{
    uint8_t     b[12];
    uint16_t    sp,
                dp;

    b[0] = src_ip & 0xFF;
    b[1] = (src_ip >> 8) & 0xFF;
    b[2] = (src_ip >> 16) & 0xFF;
    b[3] = (src_ip >> 24) & 0xFF;
    b[4] = src_port & 0xFF;
    b[5] = (src_port >> 8) & 0xFF;
    sp = ((b[4] << 8) & 0xFF00) | (b[5] & 0x00FF);
    b[6] = dst_ip & 0xFF;
    b[7] = (dst_ip >> 8) & 0xFF;
    b[8] = (dst_ip >> 16) & 0xFF;
    b[9] = (dst_ip >> 24) & 0xFF;
    b[10] = dst_port & 0xFF;
    b[11] = (dst_port >> 8) & 0xFF;
    dp = ((b[10] << 8) & 0xFF00) | (b[11] & 0x00FF);
    printf("rx: %u.%u.%u.%u:%u -> %u.%u.%u.%u:%u (%d bytes)\n",
            b[0], b[1], b[2], b[3], sp,
            b[6], b[7], b[8], b[9], dp,
            len);
}

static int
check_mem(void *addr, rte_iova_t *iova, size_t pgsz, int n_pages)
{
	int i;

	/* check that we can get this memory from EAL now */
	for (i = 0; i < n_pages; i++) {
		const struct rte_memseg_list *msl;
		const struct rte_memseg *ms;
		void *cur = RTE_PTR_ADD(addr, pgsz * i);
		rte_iova_t expected_iova;

		msl = rte_mem_virt2memseg_list(cur);
		if (!msl->external) {
			printf("%s():%i: Memseg list is not marked as external\n",
				__func__, __LINE__);
			return -1;
		}

		ms = rte_mem_virt2memseg(cur, msl);
		if (ms == NULL) {
			printf("%s():%i: Failed to retrieve memseg for external mem\n",
				__func__, __LINE__);
			return -1;
		}
		if (ms->addr != cur) {
			printf("%s():%i: VA mismatch\n", __func__, __LINE__);
			return -1;
		}
		expected_iova = (iova == NULL) ? RTE_BAD_IOVA : iova[i];
		if (ms->iova != expected_iova) {
			printf("%s():%i: IOVA mismatch\n", __func__, __LINE__);
			return -1;
		}
	}
	return 0;
}
