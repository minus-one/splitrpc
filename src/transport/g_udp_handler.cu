// Copyright (c) Adithya Kumar, The Pennsylvania State University. All rights reserved.
// Licensed under the MIT License.

#include <cuda_runtime.h>
#include <helper_functions.h>
#include <helper_cuda.h>

#include <emmintrin.h> 
#include "gdr_mem_manager.h"
#include "g_udp_handler.cuh"
#include "config_utils.h"

#ifdef PROFILE_MODE
#include <nvToolsExt.h>
#endif

#define G_PP_TB_SZ MAX_BI_SIZE

static inline uint16_t get_g_pp_launch_type() {
  return readEnvInfo<uint16_t>("P2P_RPC_G_PP_LAUNCH_TYPE", 2);
}

/**
  We require separate versions of these structures because CUDA does not like bitfields
  Not sure if there is a way around it.
**/
struct g_mac_addr {
  uint8_t addr[6];
}__attribute__((packed));

struct g_eth_hdr {
  struct g_mac_addr dst_mac_;
  struct g_mac_addr src_mac_;
  uint16_t eth_type_;
} __attribute__((packed));

// In network-byte-order
struct g_ipv4_hdr {
  uint8_t ihl_version;
  //uint8_t version : 4;
  uint8_t ecn_dscp;
  //uint8_t dscp : 6;
  uint16_t tot_len;
  uint16_t id;
  uint16_t frag_off;
  uint8_t ttl;
  uint8_t protocol;
  uint16_t check;
  uint32_t src_ip;
  uint32_t dst_ip;
} __attribute__((packed));

struct g_udp_hdr {
  uint16_t src_port;
  uint16_t dst_port;
  uint16_t len;
  uint16_t check;
} __attribute__((packed));

//#define MAKE_INT16(a, b) (signed short)((a << 8) | (unsigned char)b)

#define SWAP16(num) (num>>8) | (num<<8);

/*
 * Compute checksum of IPv4 pseudo-header.
 */
__device__ __forceinline__ uint16_t ipv4_pseudo_csum(struct g_ipv4_hdr *ip)
{
    uint16_t psd_hdr_compressed[8];
    *(uint32_t*)(psd_hdr_compressed + 0) = ip->src_ip;
    *(uint32_t*)(psd_hdr_compressed + 2) = ip->dst_ip;
    *(uint8_t*)(psd_hdr_compressed + 4) = 0;
    *((uint8_t*)psd_hdr_compressed + 9) = ip->protocol;
    uint16_t new_len = SWAP16(ip->tot_len);
    new_len = new_len - (uint16_t)(4u * 4u); 
    *(uint16_t*)((uint8_t*)psd_hdr_compressed + 10) = SWAP16(new_len);
    uint32_t sum = 0;
    uint16_t size = 8 * sizeof(uint16_t);
    uint16_t *p = psd_hdr_compressed;
    
    while (size > 1) {
        sum += *p;
        size -= sizeof(uint16_t);
        p++;
    }
    if (size) {
        sum += *((const uint8_t *) p);
    }
    
    // Fold 32-bit @x one's compliment sum into 16-bit value.
    sum = (sum & 0x0000FFFF) + (sum >> 16);
    sum = (sum & 0x0000FFFF) + (sum >> 16);
    return (uint16_t) sum;
}

__device__ __forceinline__ void swap_header_internal(uintptr_t __restrict pkt_hdr)
{
  struct g_eth_hdr *eth_h = (struct g_eth_hdr *)(pkt_hdr);
  struct g_ipv4_hdr *ip_h = (struct g_ipv4_hdr *)((uint8_t*)pkt_hdr + sizeof(g_eth_hdr));
  struct g_udp_hdr *udp_h = (struct g_udp_hdr *)((uint8_t*)pkt_hdr + sizeof(g_eth_hdr) + sizeof(g_ipv4_hdr));

  // Swap the MAC addr to do the echo
  struct g_mac_addr tmp_mac;
  tmp_mac = eth_h->src_mac_;
  eth_h->src_mac_ = eth_h->dst_mac_;
  eth_h->dst_mac_ = tmp_mac;

  // Swap the IP-Addresses
  uint32_t tmp_ip = ip_h->src_ip;
  ip_h->src_ip = ip_h->dst_ip;
  ip_h->dst_ip = tmp_ip;
  ip_h->check = 0;

  uint16_t tmp_port = udp_h->src_port;  
  udp_h->src_port = udp_h->dst_port;
  udp_h->dst_port = tmp_port;
  udp_h->check = 0;
  udp_h->check = ipv4_pseudo_csum(ip_h);

  //// FIXME: UGLY hack alert
  //uint64_t sock_perf_header = *((uint8_t*)pkt_hdr + sizeof(struct g_eth_hdr) + sizeof(uint64_t));
  //sock_perf_header &= 0xFFFFFFFF00000000;
}

__global__ void persistent_swap_hdrs(pp_params *d_item, volatile uint32_t *door_bell)
{
  // Get thread ID.
  int tid = blockDim.x * blockIdx.x + threadIdx.x;

  uint32_t wait_status;
  __shared__ uint32_t wait_status_shared[1];
  __shared__ uint32_t num_items;
  __shared__ uintptr_t *pkt_addrs;

  __syncthreads();

  while(1) {
    // Wait for work to be ready
    if (tid == 0) {
      while (1) {
        wait_status = ACCESS_ONCE(*(door_bell));
        if(wait_status == 1 || wait_status == 3) {
          wait_status_shared[0] = wait_status;
          num_items = d_item->hdr_bufs.num_items;
          pkt_addrs = d_item->hdr_bufs.burst_items;
          __threadfence_block();
          break;
        }
      }
    } 
    __syncthreads();

    if (wait_status_shared[0] != 1 && wait_status_shared[0] != 2)
      break;

    //Do Work
    if(tid < num_items) {
      swap_header_internal(pkt_addrs[tid]);
    }

    __threadfence();
    __syncthreads();

    // Signal work to be complete
    if (tid == 0) {
      ACCESS_ONCE(*(door_bell)) = 2;
      __threadfence_system();
    }
  }
}

__global__ void swap_hdrs(pp_params *d_item)
{
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  if (tid < d_item->hdr_bufs.num_items) {
    swap_header_internal(d_item->hdr_bufs.burst_items[tid]);
  }
}

// Start PP on GPU as a persistent thread
pp_handler_ctx* setup_g_pp() 
{
  struct pp_handler_ctx *ctx = new pp_handler_ctx;
  ctx->launch_type = get_g_pp_launch_type();

  // Setup the streams for the workload
  checkCudaErrors(cudaStreamCreateWithFlags(&ctx->g_stream, cudaStreamNonBlocking));
  checkCudaErrors(cudaEventCreateWithFlags(&ctx->work_complete, cudaEventDisableTiming));

  // Setup synch
  ctx->_stub = BufItemPool<pp_params>::create_buf_item_pool(1, get_cuda_device_id());
  ctx->door_bell = BufItemPool<uint32_t>::create_buf_item_pool(1, get_cuda_device_id());

  ACCESS_ONCE(*(ctx->door_bell)) = 0;
  _mm_mfence();

  if(ctx->launch_type == 2) {
    dim3 blockSize(G_PP_TB_SZ, 1, 1);
    dim3 gridSize(1, 1);
    pp_params *d_stub = BufItemPool<pp_params>::get_dev_ptr(ctx->_stub);
    uint32_t *d_doorbell = BufItemPool<uint32_t>::get_dev_ptr(ctx->door_bell);

    persistent_swap_hdrs<<<gridSize, blockSize, 0, ctx->g_stream>>>
      (d_stub, d_doorbell);

    printf("Launched persistent echo kernel: GRID: %d, %d, %d ; Block: %d, %d, %d\n", \
        gridSize.x, gridSize.y, gridSize.z, blockSize.x, blockSize.y, blockSize.z);
  } else if(ctx->launch_type == 1) {
    printf("Configured to launch echo kernels\n");
  } else {
    printf("G_PP: Incorrect launch type, defaulting to no work\n");
    ctx->launch_type = 0;
  }
  return ctx;
}

void stop_g_pp(pp_handler_ctx *ctx)
{
  if(ctx->launch_type == 0)
    return;

  ACCESS_ONCE(*(ctx->door_bell)) = 3;
  _mm_mfence();
  checkCudaErrors(cudaStreamSynchronize(ctx->g_stream));
  //FIXME: Commenting this as with persistent threads cuMemFree blocks
  //BufItemPool<workitem>::delete_buf_item_pool(ctx->_stub, get_cuda_device_id());
  //BufItemPool<uint32_t>::delete_buf_item_pool(ctx->door_bell, get_cuda_device_id());
}

// Ring door_bell to do echo task on the GPU
void do_g_pp(pp_handler_ctx *ctx, struct p2p_hbufs *h_wi) 
{
  if(ctx->launch_type == 0)
    return;
  for(int i = 0; i < h_wi->num_items; i++)
    ctx->_stub->hdr_bufs.burst_items[i] = h_wi->burst_items[i];
  ctx->_stub->hdr_bufs.num_items = h_wi->num_items;
  _mm_mfence();
  if(ctx->launch_type == 2) {
#ifdef PROFILE_MODE
    nvtxRangePush("persistent-swap-hdr");
#endif
    ACCESS_ONCE(*(ctx->door_bell)) = 1;
    _mm_mfence();
    while(ACCESS_ONCE(*(ctx->door_bell)) != 2);
#ifdef PROFILE_MODE
    nvtxRangePop();
#endif
  } else if(ctx->launch_type == 1) {
    dim3 blockSize(G_PP_TB_SZ, 1, 1);
    dim3 gridSize(ctx->_stub->hdr_bufs.num_items / G_PP_TB_SZ + 1, 1);
    //printf("Launching g_pp Work-Size: %d\n", nb_rx);
    //printf("GRID: %d, %d, %d ; Block: %d, %d, %d\n", \
    //    gridSize.x, gridSize.y, gridSize.z, blockSize.x, blockSize.y, blockSize.z);

#ifdef PROFILE_MODE
    nvtxRangePush("kernel-swap-hdr");
#endif
    // FIXME: This needs to be cleanly handled. Assumes there is only one bi
    pp_params *d_stub = BufItemPool<pp_params>::get_dev_ptr(ctx->_stub);

    swap_hdrs<<<gridSize, blockSize, 0, ctx->g_stream>>>(d_stub);
    checkCudaErrors(cudaStreamSynchronize(ctx->g_stream));
#ifdef PROFILE_MODE
    nvtxRangePop();
#endif
  }
}
