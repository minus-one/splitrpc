// Copyright (c) Adithya Kumar, The Pennsylvania State University. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <emmintrin.h>

#define CUDA_DRIVER_API
#include <helper_functions.h>
#include <helper_cuda.h>

#include "gdr_mem_manager.h"
#include "debug_utils.h"

#define PAGE_ROUND_UP(x, n)     (((x) + ((n) - 1)) & ~((n) - 1))

static size_t find_lcm(size_t x, size_t y)
{
  size_t gcd = 1;
  for(size_t i = 1 ; i <= x && i <= y; ++i) {
    if(x % i == 0 && y % i == 0)
      gcd = i;
  }

  return static_cast<size_t>((x * y) / gcd);
}

class P2pRpcAppRrSimpleMemPool {
private:
    int pool_size;
    int device_id;
    size_t req_pool_size, resp_pool_size, doorbells_pool_size;
    size_t padded_req_pool_size, padded_resp_pool_size, padded_doorbells_pool_size;
    void *req_pool_addr_range, *resp_pool_addr_range, *doorbells_pool_addr_range;

    gdr_memseg_info doorbells_gdr_mm;
   
    void *mem_region_va;
    size_t tot_mem_region_sz_va;

  public:
    size_t req_size, resp_size;
    
    inline volatile uint32_t* get_doorbells_host()
    {
      //return (volatile uint32_t*)doorbells_gdr_mm.phost_ptr;
      return (volatile uint32_t*)doorbells_pool_addr_range;
    }

    inline volatile uint32_t* get_doorbells_device()
    {
      //return (volatile uint32_t*)doorbells_gdr_mm.pdev_addr;
      return (volatile uint32_t*)doorbells_pool_addr_range;
    }

    inline void* get_req_addr_range()
    {
      return req_pool_addr_range;
    }

    inline size_t get_req_addr_pool_size()
    {
      return padded_req_pool_size;
    }

    inline size_t get_resp_addr_pool_size()
    {
      return padded_resp_pool_size;
    }

    inline void* get_resp_addr_range()
    {
      return resp_pool_addr_range;
    }

    inline void* get_state_addr_range()
    {
      return doorbells_pool_addr_range; 
    }

    inline int get_pool_size()
    {
      return pool_size;
    }

    P2pRpcAppRrSimpleMemPool(int _pool_size, int _device_id, size_t _req_size, size_t _resp_size)
    {
      CUdevice dev;
      checkCudaErrors(cudaFree(0));
      checkCudaErrors(cuCtxGetDevice(&dev));

      pool_size = _pool_size;
      device_id = _device_id;
      req_size = _req_size;
      resp_size = _resp_size;
      int StreamMemOpsSupport=1;
      checkCudaErrors(cuDeviceGetAttribute(&StreamMemOpsSupport, CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_MEM_OPS, dev));
      if(StreamMemOpsSupport) {
        printf("Stream Mem OPs support present on device_id: %d\n", device_id);
      } else {
        printf("Stream Mem OPs support NOT! present on device_id: %d\n", device_id);
      }

      printf("P2pRpcAppRrSimpleMemPool: NOT USING CUDAVMM Allocating device memory on device_id: %d\n", device_id);
      req_pool_size = req_size * pool_size;
      resp_pool_size = resp_size * pool_size;
      doorbells_pool_size = sizeof(uint32_t) * pool_size;

      padded_req_pool_size = req_pool_size;
      padded_resp_pool_size = resp_pool_size;

      padded_doorbells_pool_size = doorbells_pool_size;

      TRACE_PRINTF("req_pool_size: %ld, resp_pool_size: %ld, doorbells_pool_size: %ld\n",
          req_pool_size, resp_pool_size, doorbells_pool_size);

      checkCudaErrors(cuMemAlloc((CUdeviceptr*)&req_pool_addr_range, padded_req_pool_size));
      checkCudaErrors(cuMemAlloc((CUdeviceptr*)&resp_pool_addr_range, padded_resp_pool_size));
      checkCudaErrors(cudaHostAlloc(&doorbells_pool_addr_range, padded_doorbells_pool_size, cudaHostAllocMapped));

      TRACE_PRINTF("req_pool_addr_range: Start= %p, End= %p\n", 
          req_pool_addr_range, (void*)((uint8_t*)req_pool_addr_range + padded_req_pool_size));
      TRACE_PRINTF("resp_pool_addr_range: Start= %p, End= %p\n", 
          resp_pool_addr_range, (void*)((uint8_t*)resp_pool_addr_range + padded_resp_pool_size));
      TRACE_PRINTF("doorbells_pool_addr_range: Start= %p, End= %p\n", 
          doorbells_pool_addr_range, (void*)((uint8_t*)doorbells_pool_addr_range + padded_doorbells_pool_size));

      //// Now map the doorbells to GDR
      //doorbells_gdr_mm.input_size = padded_doorbells_pool_size;
      //gdr_mem_manager *G = get_gdr_mem_manager(device_id);
      //if(G->pin_and_map_memory(&doorbells_gdr_mm, (CUdeviceptr)doorbells_pool_addr_range,  padded_doorbells_pool_size) != 0) {
      //  printf("Failed to pin doorbells to GDR\n");
      //  pool_size = 0;
      //  req_pool_addr_range = NULL;
      //  resp_pool_addr_range = NULL;
      //  doorbells_pool_addr_range = NULL;
      //  return;
      //}
    }
 
    ~P2pRpcAppRrSimpleMemPool() {
      checkCudaErrors(cuMemFree((CUdeviceptr)req_pool_addr_range));
      checkCudaErrors(cuMemFree((CUdeviceptr)resp_pool_addr_range));
      //checkCudaErrors(cuMemFree((CUdeviceptr)doorbells_pool_addr_range));
    }

};

class P2pRpcAppRrMemPool {
 private:
    int pool_size;
    int device_id;
    size_t req_pool_size, resp_pool_size, doorbells_pool_size;
    size_t padded_req_pool_size, padded_resp_pool_size, padded_doorbells_pool_size;
    void *req_pool_addr_range, *resp_pool_addr_range, *doorbells_pool_addr_range;

    // CUDA VMM API
    size_t granularity = 0;
    CUmemAllocationProp allocProp = {};
    CUmemGenericAllocationHandle reqAllocHandle, respAllocHandle, doorbellsAllocHandle;
    CUmemAccessDesc accessDesc = {};
    gdr_memseg_info doorbells_gdr_mm;
   
    void *mem_region_va;
    size_t tot_mem_region_sz_va;

  public:
    size_t req_size, resp_size;

    inline volatile uint32_t* get_doorbells_host()
    {
      //return (volatile uint32_t*)doorbells_gdr_mm.phost_ptr;
      return (volatile uint32_t*)doorbells_pool_addr_range;
    }

    inline volatile uint32_t* get_doorbells_device()
    {
      //return (volatile uint32_t*)doorbells_gdr_mm.pdev_addr;
      return (volatile uint32_t*)doorbells_pool_addr_range;
    }

    inline void* get_req_addr_range()
    {
      return req_pool_addr_range;
    }

    inline size_t get_req_addr_pool_size()
    {
      return padded_req_pool_size;
    }

    inline size_t get_resp_addr_pool_size()
    {
      return padded_resp_pool_size;
    }

    inline void* get_resp_addr_range()
    {
      return resp_pool_addr_range;
    }

    inline void* get_state_addr_range()
    {
      return doorbells_pool_addr_range; 
    }

    inline int get_pool_size()
    {
      return pool_size;
    }

    P2pRpcAppRrMemPool(int _pool_size, int _device_id, size_t _req_size, size_t _resp_size)
    {
      pool_size = _pool_size;
      device_id = _device_id;
      req_size = _req_size;
      resp_size = _resp_size;

      CUdevice dev;
      int supportsVMM = 0;
      checkCudaErrors(cudaFree(0));
      checkCudaErrors(cuCtxGetDevice(&dev));
      checkCudaErrors(cuDeviceGetAttribute(&supportsVMM, CU_DEVICE_ATTRIBUTE_VIRTUAL_ADDRESS_MANAGEMENT_SUPPORTED, dev));
      if(!supportsVMM) {
        printf("VMM support not present on this device_id: %d, rr-pool failed to create\n", device_id);
        pool_size = 0;
        req_pool_addr_range = NULL;
        resp_pool_addr_range = NULL;
        doorbells_pool_addr_range = NULL;
        return;
      }
      int RDMASupported = 0;
      checkCudaErrors(cuDeviceGetAttribute(&RDMASupported, CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WITH_CUDA_VMM_SUPPORTED, dev));
      if (!RDMASupported) {
        printf("GDR + VMM support not present on this device_id: %d, rr-pool failed to create\n", device_id);
        pool_size = 0;
        req_pool_addr_range = NULL;
        resp_pool_addr_range = NULL;
        doorbells_pool_addr_range = NULL;
        return;
      }
      int StreamMemOpsSupport=1;
      checkCudaErrors(cuDeviceGetAttribute(&StreamMemOpsSupport, CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_MEM_OPS, dev));
      if(StreamMemOpsSupport) {
        printf("Stream Mem OPs support present on device_id: %d\n", device_id);
      } else {
        printf("Stream Mem OPs support NOT! present on device_id: %d\n", device_id);
      }
      printf("P2pRpcAppRrMemPool: VMM Support available, creating physical memory and VMM mappings on device_id: %d\n", device_id);

      req_pool_size = req_size * pool_size;
      resp_pool_size = resp_size * pool_size;
      doorbells_pool_size = sizeof(uint32_t) * pool_size;

      TRACE_PRINTF("req_pool_size: %ld, resp_pool_size: %ld, doorbells_pool_size: %ld\n",
          req_pool_size, resp_pool_size, doorbells_pool_size);

      // CUDA physical memory allocation
      granularity = 0; 
      size_t gran = 0;
      memset(&allocProp, 0, sizeof(CUmemAllocationProp));
      allocProp.type = CU_MEM_ALLOCATION_TYPE_PINNED;
      allocProp.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
      allocProp.location.id = device_id;
      allocProp.allocFlags.gpuDirectRDMACapable = 1;
      checkCudaErrors(cuMemGetAllocationGranularity(&gran, &allocProp, CU_MEM_ALLOC_GRANULARITY_MINIMUM));

      // In case gran is smaller than GPU_PAGE_SIZE
      granularity = PAGE_ROUND_UP(gran, GPU_PAGE_SIZE);
      TRACE_PRINTF("Recommended gran size: %ld, PageSize: %ld, Setting granularity: %ld\n", gran, GPU_PAGE_SIZE, granularity);

      //padded_req_pool_size = PAGE_ROUND_UP(req_pool_size, granularity);
      //padded_resp_pool_size = PAGE_ROUND_UP(resp_pool_size, granularity);
      padded_req_pool_size = find_lcm(req_pool_size, granularity);
      padded_resp_pool_size = find_lcm(resp_pool_size, granularity);

      padded_doorbells_pool_size = PAGE_ROUND_UP(doorbells_pool_size, granularity);

      TRACE_PRINTF("padded_req_pool_size: %ld, padded_resp_pool_size: %ld, padded_doorbells_pool_size: %ld\n",
          padded_req_pool_size, padded_resp_pool_size, padded_doorbells_pool_size);

      checkCudaErrors(cuMemCreate(&reqAllocHandle, padded_req_pool_size, &allocProp, 0));
      checkCudaErrors(cuMemCreate(&respAllocHandle, padded_resp_pool_size, &allocProp, 0));
      //checkCudaErrors(cuMemCreate(&doorbellsAllocHandle, padded_doorbells_pool_size, &allocProp, 0));

      // Allocate large VA and map individual stuff inside them
      tot_mem_region_sz_va = (padded_req_pool_size * 2) + (padded_resp_pool_size * 2) + padded_doorbells_pool_size;
      checkCudaErrors(cuMemAddressReserve((CUdeviceptr*)&mem_region_va, tot_mem_region_sz_va, granularity, 0ULL, 0ULL));
      // Map various sub-regions within the VA
      // Note we put the doorbells to a higher VA range so that when writing to them,
      // we know the req/resp has been written prior to doorbells being available
      req_pool_addr_range = mem_region_va;
      resp_pool_addr_range = (uint8_t*)mem_region_va + (padded_req_pool_size * 2);
      //doorbells_pool_addr_range = (uint8_t*)mem_region_va + (padded_req_pool_size * 2) + (padded_resp_pool_size * 2);

      // Allocate VA
      //checkCudaErrors(cuMemAddressReserve((CUdeviceptr*)&req_pool_addr_range, padded_req_pool_size * 2, granularity, 0ULL, 0ULL));
      //checkCudaErrors(cuMemAddressReserve((CUdeviceptr*)&resp_pool_addr_range, padded_resp_pool_size * 2, granularity, 0ULL, 0ULL));
      //checkCudaErrors(cuMemAddressReserve((CUdeviceptr*)&doorbells_pool_addr_range, padded_doorbells_pool_size, granularity, 0ULL, 0ULL));

      // Map the phy mem to VA (twice so that this becomes a ring buffer) 
      checkCudaErrors(cuMemMap((CUdeviceptr)req_pool_addr_range, padded_req_pool_size, 0, reqAllocHandle, 0));
      checkCudaErrors(cuMemMap((CUdeviceptr)((uint8_t*)req_pool_addr_range + padded_req_pool_size), padded_req_pool_size, 0, reqAllocHandle, 0));
      checkCudaErrors(cuMemMap((CUdeviceptr)resp_pool_addr_range, padded_resp_pool_size, 0, respAllocHandle, 0));
      checkCudaErrors(cuMemMap((CUdeviceptr)((uint8_t*)resp_pool_addr_range + padded_resp_pool_size), padded_resp_pool_size, 0, respAllocHandle, 0));

      // Map the phy mem of doorbells to the VA of doorbells
      //checkCudaErrors(cuMemMap((CUdeviceptr)doorbells_pool_addr_range, padded_doorbells_pool_size, 0, doorbellsAllocHandle, 0));

      // Provide access permissions for the VA ranges
      accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
      accessDesc.location.id = device_id;
      accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
      checkCudaErrors(cuMemSetAccess((CUdeviceptr)req_pool_addr_range, padded_req_pool_size * 2, &accessDesc, 1));
      checkCudaErrors(cuMemSetAccess((CUdeviceptr)resp_pool_addr_range, padded_resp_pool_size * 2, &accessDesc, 1));
      //checkCudaErrors(cuMemSetAccess((CUdeviceptr)doorbells_pool_addr_range, padded_doorbells_pool_size, &accessDesc, 1));
      checkCudaErrors(cudaHostAlloc(&doorbells_pool_addr_range, padded_doorbells_pool_size, cudaHostAllocMapped));

      TRACE_PRINTF("req_pool_addr_range: Start= %p, End= %p\n", 
          req_pool_addr_range, (void*)((uint8_t*)req_pool_addr_range + padded_req_pool_size));
      TRACE_PRINTF("resp_pool_addr_range: Start= %p, End= %p\n", 
          resp_pool_addr_range, (void*)((uint8_t*)resp_pool_addr_range + padded_resp_pool_size));
      TRACE_PRINTF("doorbells_pool_addr_range: Start= %p, End= %p\n", 
          doorbells_pool_addr_range, (void*)((uint8_t*)doorbells_pool_addr_range + padded_doorbells_pool_size));
          
      //// Now map the doorbells to GDR
      //doorbells_gdr_mm.input_size = padded_doorbells_pool_size;
      //gdr_mem_manager *G = get_gdr_mem_manager(device_id);
      //if(G->pin_and_map_memory(&doorbells_gdr_mm, (CUdeviceptr)doorbells_pool_addr_range,  padded_doorbells_pool_size) != 0) {
      //  printf("Failed to pin doorbells to GDR\n");
      //  pool_size = 0;
      //  req_pool_addr_range = NULL;
      //  resp_pool_addr_range = NULL;
      //  doorbells_pool_addr_range = NULL;
      //  return;
      //}
///////////////////////////////////// END OF ALL MEMORY ALLOCATIONS ////////////////////////////////////////////
    }

    ~P2pRpcAppRrMemPool() {
      checkCudaErrors(cuMemUnmap((CUdeviceptr)req_pool_addr_range, padded_req_pool_size));
      checkCudaErrors(cuMemUnmap((CUdeviceptr)((uint8_t*)req_pool_addr_range + padded_req_pool_size), padded_req_pool_size));
      checkCudaErrors(cuMemUnmap((CUdeviceptr)resp_pool_addr_range, padded_resp_pool_size));
      checkCudaErrors(cuMemUnmap((CUdeviceptr)((uint8_t*)resp_pool_addr_range + padded_resp_pool_size), padded_resp_pool_size));
      //checkCudaErrors(cuMemUnmap((CUdeviceptr)doorbells_pool_addr_range, padded_doorbells_pool_size));
      checkCudaErrors(cuMemAddressFree((CUdeviceptr)mem_region_va, tot_mem_region_sz_va));
      checkCudaErrors(cuMemRelease(reqAllocHandle));
      checkCudaErrors(cuMemRelease(respAllocHandle));
      //checkCudaErrors(cuMemRelease(doorbellsAllocHandle));
    }
};


