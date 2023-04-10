// Copyright (c) Adithya Kumar, The Pennsylvania State University. All rights reserved.
// Licensed under the MIT License.

#include "dpdk_init.h"
/*
 * Initializes a given port using global settings and with the RX buffers
 * coming from the mbuf_pool passed as a parameter.
 * TODO: Unbundle this into setting up ports and setting up queues
 */
static int
port_init(uint16_t port, int queue_id, struct dpdk_memseg_info *memseg_info, int nsegs)
{	
  int retval;

  struct rte_eth_conf port_conf;
  memset(&port_conf, 0, sizeof(port_conf));
  port_conf.rxmode.max_rx_pkt_len = RTE_ETHER_MAX_LEN;

  struct rte_eth_dev_info dev_info;
  if (!rte_eth_dev_is_valid_port(port))
    return -1;
  retval = rte_eth_dev_info_get(port, &dev_info);
  if (retval != 0) {
    printf("Error during getting device (port %u) info: %s\n",
        port, strerror(-retval));
    return retval;
  }

  if(dev_info.rx_offload_capa & DEV_RX_OFFLOAD_JUMBO_FRAME) {
    port_conf.rxmode.offloads |= DEV_RX_OFFLOAD_JUMBO_FRAME;
    port_conf.rxmode.max_rx_pkt_len = MAX_MTU;
  } else {
    printf("> No JUMBO frame support, reset MAX_MTU and restart application\n");
  }

  if (dev_info.tx_offload_capa & DEV_TX_OFFLOAD_MBUF_FAST_FREE)
    port_conf.txmode.offloads |=
      DEV_TX_OFFLOAD_MBUF_FAST_FREE;
  else
    printf("> No MBUF_FAST_FREE support\n");

  if (dev_info.tx_offload_capa & DEV_TX_OFFLOAD_IPV4_CKSUM)
    port_conf.txmode.offloads |= DEV_TX_OFFLOAD_IPV4_CKSUM;
  else
    printf("> No IPV4 checksum offload possible\n");

  if (dev_info.tx_offload_capa & DEV_TX_OFFLOAD_UDP_CKSUM)
    port_conf.txmode.offloads |= DEV_TX_OFFLOAD_UDP_CKSUM;
  else
    printf("> No UDP checksum offload possible\n");

  if (dev_info.tx_offload_capa & DEV_TX_OFFLOAD_OUTER_IPV4_CKSUM)
    port_conf.txmode.offloads |= DEV_TX_OFFLOAD_OUTER_IPV4_CKSUM;
  else
    printf("> No outer IPV4 checksum offload possible\n");

  if (dev_info.tx_offload_capa & DEV_TX_OFFLOAD_OUTER_UDP_CKSUM)
    port_conf.txmode.offloads |= DEV_TX_OFFLOAD_OUTER_UDP_CKSUM;
  else
    printf("> No outer UDP checksum offload possible\n");

  if(! (dev_info.rx_offload_capa & DEV_RX_OFFLOAD_SCATTER)) {
    printf("> No Buffer split offload possible\n");
  }

  if(! (dev_info.rx_offload_capa & RTE_ETH_RX_OFFLOAD_BUFFER_SPLIT)) {
    printf("> No Buffer split offload possible\n");
  }

  if(nsegs < 1 || nsegs > 2) {
    printf("Only 1/2 segments currently supported in RX Buffers\n");
    return -1;
  }

  struct rte_eth_rxconf rxconf_qsplit;                   
  struct rte_eth_rxseg_split *rx_seg;                    
  union rte_eth_rxseg rx_useg[2] = {};
  // FIXME: Check if buffer split works with just a single split and clean
  if(nsegs == 2) {
    // SETTING UP BUFFER SPLITS
    memcpy(&rxconf_qsplit, &dev_info.default_rxconf, sizeof(rxconf_qsplit));          
    rxconf_qsplit.offloads = DEV_RX_OFFLOAD_JUMBO_FRAME | DEV_RX_OFFLOAD_SCATTER | RTE_ETH_RX_OFFLOAD_BUFFER_SPLIT;
    rxconf_qsplit.rx_nseg = 2;
    rxconf_qsplit.rx_seg = rx_useg;                                                   

    rx_seg = &rx_useg[0].split;                                                          
    rx_seg->mp = memseg_info[0]._mbuf_pool;
    rx_seg->length = memseg_info[0].elt_size - RTE_PKTMBUF_HEADROOM;                                                   
    rx_seg->offset = 0; // offset within this segment                                                           

    rx_seg = &rx_useg[1].split;                                                          
    rx_seg->mp = memseg_info[1]._mbuf_pool;                                                          
    rx_seg->length = memseg_info[1].elt_size - RTE_PKTMBUF_HEADROOM;
    //rx_seg->length = 0; // Implies rest of the packet goes here                                                                  
    rx_seg->offset = 0; // offset within this segment                          
    port_conf.rxmode.offloads |= DEV_RX_OFFLOAD_SCATTER | RTE_ETH_RX_OFFLOAD_BUFFER_SPLIT;
    port_conf.txmode.offloads |= DEV_TX_OFFLOAD_MULTI_SEGS;                       
  }

  /* Configure the Ethernet device. */
  retval = rte_eth_dev_configure(port, MAX_RX_QUEUE_PER_PORT, MAX_TX_QUEUE_PER_PORT, &port_conf);
  if (retval != 0)
    return retval;
  
  uint16_t nb_rxd = RX_RING_SIZE;
  uint16_t nb_txd = TX_RING_SIZE;
  retval = rte_eth_dev_adjust_nb_rx_tx_desc(port, &nb_rxd, &nb_txd);
  if (retval != 0)
    return retval;

  /* Configure the RX/TX queues */
  printf("Setting TX queue\n");
  //struct rte_eth_txconf txconf;
  //txconf = dev_info.default_txconf;
  //txconf.offloads = port_conf.txmode.offloads;
  /* Allocate and set up TX queue per Ethernet port. */
  //retval = rte_eth_tx_queue_setup(port, queue_id, nb_txd,
  //    rte_eth_dev_socket_id(port), &txconf);
  retval = rte_eth_tx_queue_setup(port, queue_id, nb_txd,
      rte_eth_dev_socket_id(port), NULL);

  if (retval < 0)
    return retval;

  printf("Setting RX queue\n");
  /* Allocate and set up RX queues for Ethernet port. */
  if(nsegs == 2) {
    // SPLIT RX SEGMENTS
    retval = rte_eth_rx_queue_setup(port, queue_id, nb_rxd, 
        rte_eth_dev_socket_id(port), &rxconf_qsplit, NULL);
  } else {
    // NORMAL RX SEGMENTS
    retval = rte_eth_rx_queue_setup(port, queue_id, nb_rxd,
        rte_eth_dev_socket_id(port), NULL, memseg_info[0]._mbuf_pool);
  }
  if (retval < 0)
    return retval;

  return 1;
}

static int
start_port(uint16_t nic_port) {
  /* Start the Ethernet port. */
  int retval = rte_eth_dev_start(nic_port);
  if (retval < 0)
    return retval;

  /* Display the port MAC address. */
  struct rte_ether_addr addr;
  retval = rte_eth_macaddr_get(nic_port, &addr);
  if (retval != 0)
    return retval;

  printf("Port %u MAC: %02" PRIx8 " %02" PRIx8 " %02" PRIx8
      " %02" PRIx8 " %02" PRIx8 " %02" PRIx8 "\n",
      nic_port,
      addr.addr_bytes[0], addr.addr_bytes[1],
      addr.addr_bytes[2], addr.addr_bytes[3],
      addr.addr_bytes[4], addr.addr_bytes[5]);

  /* Enable RX in promiscuous mode for the Ethernet device. */
  retval = rte_eth_promiscuous_enable(nic_port);
  if (retval != 0)
    return retval;

  return 0;
}

// Sets up an external buffer either on host/device using gpudirect mappings
// buf_loc = -2 => Allocate a mbuf pool on the default huge page allocator
// buf_loc = -1 => Allocate a memzone 
// buf_loc = x => Allocate on GPU-x
static unsigned int
setup_mbufs(uint16_t dpdk_port, struct dpdk_memseg_info& _buf_info) {
  size_t ext_buf_sz = RTE_ALIGN_CEIL(_buf_info.num_mbufs * _buf_info.elt_size, _buf_info.pgsz);
  uint16_t elt_size = _buf_info.elt_size; // Assumed this aligns with cacheline size Else call RTE_ALIGN_CEIL
  
  printf("setup_mbufs: num_mbufs: %ld, size: %ld, pgsz: %ld, elt_size: %d, socket_id: %d\n", 
      _buf_info.num_mbufs, ext_buf_sz, _buf_info.pgsz, elt_size, rte_socket_id());
	
  /***** DPDK MBUF pools allocated from the default huge page allocator (Not CUDA registered) *****/
  if(_buf_info.buf_loc == -2) {
    printf("setup_mbufs: using rte_pktmbuf_pool_create\n");
    _buf_info._mbuf_pool = rte_pktmbuf_pool_create(_buf_info.memseg_name, _buf_info.num_mbufs,
        MBUF_CACHE_SIZE, 0, _buf_info.elt_size, rte_socket_id());
    return 1;
  }

  /**** Allocate external memory - CPU_PINNED_MEM memory is registered with CUDA ****/
	void *d_ext_addr = NULL;
  if(_buf_info.buf_loc == -1) {
    printf("setup_mbufs: using rte_memzone\n");
    _buf_info._alloc_info.mz_info = rte_memzone_reserve_aligned(_buf_info.memseg_name, ext_buf_sz,
        rte_socket_id(),
        RTE_MEMZONE_IOVA_CONTIG |
        RTE_MEMZONE_1GB |
        RTE_MEMZONE_SIZE_HINT_ONLY,
        _buf_info.pgsz);
    if (_buf_info._alloc_info.mz_info == NULL) {
      printf("Could not allocate MEMZONE\n");
      return 0;
    }
    d_ext_addr = _buf_info._alloc_info.mz_info->addr;
#ifndef GPU_DISABLED
    printf("setup_mbufs: using rte_memzone and registering it with cudaHostRegister\n");
    // Register it with cuda
    cudaSetDeviceFlags(cudaDeviceMapHost);
		cudaError_t error_code = 
      cudaHostRegister(d_ext_addr, ext_buf_sz, cudaHostRegisterDefault | cudaHostRegisterMapped);
		if(error_code != cudaSuccess) {
			printf("Cuda Host register failed\n");
      return 0;
		}
#endif
#ifdef TRACE_MODE
#ifndef GPU_DISABLED
    void *cuda_ext_addr;
		cudaHostGetDevicePointer(&cuda_ext_addr, d_ext_addr, 0);
		printf("setup_mbufs: CudaHostAlloc: d_ptr: %p, h_ptr: %p\n", cuda_ext_addr, d_ext_addr);
#endif /* GPU_DISABLED */
#endif /* TRACE_MODE */
  } else if (_buf_info.buf_loc >= 0) {
#ifndef GPU_DISABLED
		printf("setup_mbufs: GDR_MEM on device: %d using gdr_mem_manager\n", _buf_info.buf_loc);
    _buf_info._alloc_info.gdr_memsg = new gdr_memseg_info;
		_buf_info._alloc_info.gdr_memsg->input_size = ext_buf_sz;
		gdr_mem_manager *G = get_gdr_mem_manager(_buf_info.buf_loc);
		if(G->alloc(_buf_info._alloc_info.gdr_memsg) != 0) {
		  printf("%s():%i: Error with GDR_MEM allocation, is GDR_MEM initialized?\n", __func__, __LINE__);
      return 0;
		}
		printf("setup_mbufs: GDR_MEM info: d_ptr: %p, h_ptr: %p\n", \
			(void*)_buf_info._alloc_info.gdr_memsg->pdev_addr, 
      (void*)_buf_info._alloc_info.gdr_memsg->phost_ptr);
		d_ext_addr = reinterpret_cast<void*>(_buf_info._alloc_info.gdr_memsg->pdev_addr);
#else
    printf("setup_mbufs: Allocation on device not possible because GPU is disabled\n");
    return 0;
#endif
	}

	if(d_ext_addr == MAP_FAILED) {
		printf("%s():%i: Failed to allocated external buffers\n", __func__, __LINE__);
    return 0;
	}
  
  if((uintptr_t)d_ext_addr % _buf_info.pgsz) {
    printf("setup_mbuf warning!: Addr %p not aligned with pgsz %ld\n", d_ext_addr, _buf_info.pgsz);
  }

	/* populate IOVA addresses */
	int n_pages = ext_buf_sz / _buf_info.pgsz;
	rte_iova_t *iovas = NULL;
	iovas = static_cast<rte_iova_t*>(malloc(sizeof(*iovas) * n_pages));
	for (int cur_page = 0; cur_page < n_pages; cur_page++) {
		rte_iova_t iova;
		size_t offset;
		void *cur;
		offset = _buf_info.pgsz * cur_page;
		cur = RTE_PTR_ADD(d_ext_addr, offset);
		/* touch the page before getting its IOVA */
		//*(volatile char *)cur = 0;
		iova = rte_mem_virt2iova(cur);
		iovas[cur_page] = iova;
    //printf("cur: %p, iova: %p\n", cur, iova);
	}

	/* Register the external memory with dpdk */
	if (rte_extmem_register(d_ext_addr, ext_buf_sz, iovas, n_pages, _buf_info.pgsz) != 0) {
		printf("%s():%i: Failed to register memory\n", __func__, __LINE__);
    return 0;
	}

  /* Setup DMA Mappings on the NIC */
	printf("Setting up DMA mappings: %p, len: %ld\n", d_ext_addr, ext_buf_sz);
	struct rte_eth_dev_info dev_info;
	rte_eth_dev_info_get(dpdk_port, &dev_info);
	if(rte_dev_dma_map(dev_info.device, d_ext_addr, iovas[0], ext_buf_sz)) {
		printf("%s():%i: Failed to register dma mapping\n", __func__, __LINE__);
    return 0;
	}
	
  struct rte_pktmbuf_extmem *xmem = \
            static_cast<rte_pktmbuf_extmem*>(malloc(sizeof(struct rte_pktmbuf_extmem)));
	xmem->buf_ptr = d_ext_addr;
	xmem->buf_len = ext_buf_sz;
	xmem->elt_size = elt_size;		
	xmem->buf_iova = *iovas;
	_buf_info._ext_mem = xmem;

  /* Allocate a mempool in the external memory to hold the mbufs. */
  // FIXME: Revisit the cache sizing for these external buffers
  _buf_info._mbuf_pool = rte_pktmbuf_pool_create_extbuf(
      _buf_info.memseg_name, _buf_info.num_mbufs, 
      MBUF_CACHE_SIZE, 0, _buf_info.elt_size, rte_socket_id(), _buf_info._ext_mem, 1);

  if (_buf_info._mbuf_pool == NULL) {
    printf("%s():%i: Failed to setup external mempool\n", __func__, __LINE__);
    return 0;
  }

  printf("setup_mbuf: External MBUF INFO: nb_mbufs: %d, buf_len: %d, buf_ptr: %p\n", 
      _buf_info._mbuf_pool->populated_size, _buf_info._mbuf_pool->elt_size, _buf_info._ext_mem->buf_ptr);

	return 1;
}

int
init_dpdk_ctx(struct dpdk_ctx *ctx)
{
  printf("Setting up DPDK port: %d, queue_id: %d, mem_alloc_type: %d, device_id: %d\n", 
      ctx->nic_port, ctx->queue_id, ctx->mem_alloc_type, ctx->device_id);

  /********************************************************************************
   * HOST_MEM_ONLY allocates the entire buffer in a host-pinned memory
   * DEV_MEM_ONLY allocates the entire buffer in a device memory
   * SPLIT_BUFFER allocates header => host, and payload => device memory
   ******************************************************************************/
  if (ctx->mem_alloc_type == HOST_MEM_ONLY) {
    ctx->nsegs = 1;
    ctx->memseg_info = new dpdk_memseg_info[ctx->nsegs];
    ctx->memseg_info[0].buf_loc = -2;
    ctx->memseg_info[0].elt_size = MAX_MTU + RTE_PKTMBUF_HEADROOM; 
    ctx->memseg_info[0].pgsz = RTE_PGSIZE_4K;
    ctx->memseg_info[0].num_mbufs = NUM_MBUFS; 
    strlcpy(ctx->memseg_info[0].memseg_name, "HM1", RTE_MEMZONE_NAMESIZE);
  } else if (ctx->mem_alloc_type == DEV_MEM_ONLY) {
    ctx->nsegs = 1;
    ctx->memseg_info = new dpdk_memseg_info[ctx->nsegs];
    ctx->memseg_info[0].buf_loc = get_cuda_device_id();
    ctx->memseg_info[0].elt_size = MAX_MTU + RTE_PKTMBUF_HEADROOM; 
    ctx->memseg_info[0].num_mbufs = NUM_MBUFS; 
    ctx->memseg_info[0].pgsz = DEVICE_PG_SZ;
    strlcpy(ctx->memseg_info[0].memseg_name, "DM1", RTE_MEMZONE_NAMESIZE);
  } else if (ctx->mem_alloc_type == BUFFER_SPLIT) {
    ctx->nsegs = 2;
    ctx->memseg_info = new dpdk_memseg_info[ctx->nsegs];
    ctx->memseg_info[0].buf_loc = -1;
    ctx->memseg_info[0].elt_size = RPC_HEADER_LEN + RTE_PKTMBUF_HEADROOM;
    ctx->memseg_info[0].num_mbufs = NUM_MBUFS;
    ctx->memseg_info[0].pgsz = RTE_PGSIZE_4K;
    strlcpy(ctx->memseg_info[0].memseg_name, "HM1", RTE_MEMZONE_NAMESIZE);
    ctx->memseg_info[1].buf_loc = ctx->device_id; 
    ctx->memseg_info[1].pgsz = DEVICE_PG_SZ;
    ctx->memseg_info[1].elt_size = RPC_MTU + RTE_PKTMBUF_HEADROOM;
    ctx->memseg_info[1].num_mbufs = NUM_MBUFS;
    strlcpy(ctx->memseg_info[1].memseg_name, "DM1", RTE_MEMZONE_NAMESIZE);
  }

  for(int i = 0 ; i < ctx->nsegs ; i++) {
    if(setup_mbufs(ctx->nic_port, ctx->memseg_info[i]) == 0) {
      printf("%s():%i: Failed to setup mbufs\n", __func__, __LINE__);
      return 0;
    }
  }

  /* Initialize ports. */
	if (port_init(ctx->nic_port, ctx->queue_id, ctx->memseg_info, ctx->nsegs) == 0) {
    printf("%s():%i: Failed to init the port %" PRIu16 "\n", __func__, __LINE__, ctx->nic_port);
    return 0;
  }

  /* Start the port */
  if(start_port(ctx->nic_port) != 0) {
    printf("%s():%i: Failed to start port %" PRIu16 "\n", __func__, __LINE__, ctx->nic_port);
    return 0;
  }

	/* Create the rings for */ 
//	ctx->work_ring = rte_ring_create(
//								"WORK_RING", 
//								MAX_WI_SIZE, 
//								rte_socket_id(), 
//								RING_F_SP_ENQ | RING_F_SC_DEQ);
//	ctx->tx_ring = rte_ring_create(
//								"TX_RING",
//								MAX_WI_SIZE,
//								rte_socket_id(),
//								RING_F_SP_ENQ | RING_F_SC_DEQ);
//	ctx->init_ring = rte_ring_create(
//								"INIT_RING",
//								MAX_WI_SIZE,
//								rte_socket_id(),
//								RING_F_SP_ENQ | RING_F_SC_DEQ);
//	ctx->echo_ring = rte_ring_create(
//								"ECHO_RING",
//								MAX_WI_SIZE,
//								rte_socket_id(),
//								RING_F_SP_ENQ | RING_F_SC_DEQ);

  ctx->num_ready_rrs = 0;
  ctx->dpdk_mbufs = new p2p_hbufs;
  ctx->hdr_bufs = new p2p_hbufs;
  ctx->payload_bufs = new p2p_bufs;

  return 1;
}

void
stop_dpdk(struct dpdk_ctx *ctx)
{
	printf("Closing port %d...", ctx->nic_port);
	rte_eth_dev_stop(ctx->nic_port);
	rte_eth_dev_close(ctx->nic_port);
}

