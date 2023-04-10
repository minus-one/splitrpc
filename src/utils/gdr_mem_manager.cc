#include "gdr_mem_manager.h"
#include <unordered_map>

// Set the input_size in gdr_memseg_info before calling this function
int gdr_mem_manager::alloc(gdr_memseg_info* g_m)
{
  cudaSetDevice(device_id);
  cudaFree(0);
  gdr_mh_t mh;
  gdr_info_t info;
  CUdeviceptr dev_addr = 0;
  void *host_ptr  = NULL;
  const unsigned int FLAG = 1;
  size_t pin_size, alloc_size, rounded_size;

  if((NULL == gdr_descr) || (0 == g_m->input_size)) {
    fprintf(stderr, "alloc_pin_gdrcopy: erroneous input parameters, \
        gdr_descr=%p, input_size=%zd\n", gdr_descr, g_m->input_size);
    return 1;
  }

  /*----------------------------------------------------------------*
   * Setting sizes                                                   */
  if(g_m->input_size < NV_MIN_PIN_SIZE)
    g_m->input_size = NV_MIN_PIN_SIZE;

  rounded_size = (g_m->input_size + GPU_PAGE_SIZE - 1) & GPU_PAGE_MASK;
  pin_size = rounded_size;
  alloc_size = rounded_size + g_m->input_size;

  /*----------------------------------------------------------------*
   * Allocate device memory.                                        */
  CUresult e = cuMemAlloc(&dev_addr, alloc_size);
  if(CUDA_SUCCESS != e) {
    fprintf(stderr, "cuMemAlloc failed\n");
    return 1;
  }

  TRACE_PRINTF("GDR_MM: Device: %d, cuMemAlloc Ptr: %p, Size: %ld\n", 
      device_id, (void*)dev_addr, alloc_size);

  g_m->free_address = (uintptr_t)dev_addr;
  //GDRDRV needs a 64kB aligned address. 
  //No more guaranteed with recent cuMemAlloc/cudaMalloc
  if(dev_addr % GPU_PAGE_SIZE) {
    pin_size = g_m->input_size;
    dev_addr += (GPU_PAGE_SIZE - (dev_addr % GPU_PAGE_SIZE));
  }

  /*----------------------------------------------------------------*
   * Set attributes for the allocated device memory.                */
  if(CUDA_SUCCESS != cuPointerSetAttribute(&FLAG, CU_POINTER_ATTRIBUTE_SYNC_MEMOPS, dev_addr)) {
    fprintf(stderr, "cuPointerSetAttribute\n");
    cuMemFree(dev_addr);
    gdr_close(gdr_descr);
    return 1;
  }
  /*----------------------------------------------------------------*
   * Pin the device buffer                                          */
  if(0 != gdr_pin_buffer(gdr_descr, dev_addr, pin_size, 0, 0, &mh))
  {
    fprintf(stderr, "gdr_pin_buffer\n");
    cuMemFree(dev_addr);
    gdr_close(gdr_descr);
    return 1;
  }
  /*----------------------------------------------------------------*
   * Map the buffer to user space                                   */
  if(0!= gdr_map(gdr_descr, mh, &host_ptr, pin_size)) {
    fprintf(stderr, "gdr_map\n");
    gdr_unpin_buffer(gdr_descr, mh);
    cuMemFree(dev_addr);
    gdr_close(gdr_descr);
    return 1;
  }
  /*  Retrieve info about the mapping                                */
  if(0 != gdr_get_info(gdr_descr, mh, &info)) {
    fprintf(stderr, "gdr_get_info\n");
    gdr_unmap(gdr_descr, mh, host_ptr, pin_size);
    gdr_unpin_buffer(gdr_descr, mh);
    cuMemFree(dev_addr);
    gdr_close(gdr_descr);
    return 1;        
  }

  /* Success - set up return values                                 */
  //g_m->pgdr       = g;
  g_m->pdev_addr    = (uintptr_t)dev_addr;
  g_m->pgdr_handle  = mh;
  g_m->phost_ptr    = (uintptr_t)host_ptr;
  //*pmmap_offset = dev_addr - info.va;
  g_m->palloc_size  = pin_size;
  printf("GDR_MM alloc, device: %d, phost_ptr: %p\n", device_id, (void*)g_m->phost_ptr);
  return 0;
}

// Assumes the len is rounded and is aligned correctly
int gdr_mem_manager::pin_and_map_memory(gdr_memseg_info *g_m, CUdeviceptr dev_addr, size_t len)
{
  cudaSetDevice(device_id);
  cudaFree(0);

  gdr_mh_t mh;
  gdr_info_t info;

  void *host_ptr  = NULL;
  TRACE_PRINTF("GDR_MM: Device: %d, CUdeviceptr: %p, Size: %ld\n", 
      device_id, (void*)dev_addr, len);
  
  //GDRDRV needs a 64kB aligned address. 
  //No more guaranteed with recent cuMemAlloc/cudaMalloc
  if(dev_addr % GPU_PAGE_SIZE) {
    fprintf(stderr, "Cannot pin and map memory not aligned with GPU page size\n");
    return 1;
    //dev_addr += (GPU_PAGE_SIZE - (dev_addr % GPU_PAGE_SIZE));
  }

  size_t pin_size = len;
  /*----------------------------------------------------------------*
   * Pin the device buffer                                          */
  if(0 != gdr_pin_buffer(gdr_descr, dev_addr, pin_size, 0, 0, &mh))
  {
    fprintf(stderr, "gdr_pin_buffer error\n");
    gdr_close(gdr_descr);
    return 1;
  }
  /*----------------------------------------------------------------*
   * Map the buffer to user space                                   */
  if(0!= gdr_map(gdr_descr, mh, &host_ptr, pin_size)) {
    fprintf(stderr, "gdr_map error\n");
    gdr_unpin_buffer(gdr_descr, mh);
    gdr_close(gdr_descr);
    return 1;
  }
  /*  Retrieve info about the mapping                                */
  if(0 != gdr_get_info(gdr_descr, mh, &info)) {
    fprintf(stderr, "gdr_get_info error\n");
    gdr_unmap(gdr_descr, mh, host_ptr, pin_size);
    gdr_unpin_buffer(gdr_descr, mh);
    gdr_close(gdr_descr);
    return 1;        
  }

  /* Success - set up return values                                 */
  g_m->pdev_addr    = (uintptr_t)dev_addr;
  g_m->pgdr_handle  = mh;
  g_m->phost_ptr    = (uintptr_t)host_ptr;
  g_m->palloc_size  = pin_size;
  printf("GDR_MM pin_and_map, device: %d, phost_ptr: %p, dev_ptr: %p\n", 
      device_id, (void*)g_m->phost_ptr, (void*)g_m->pdev_addr);
  return 0;
}

void gdr_mem_manager::cleanup(gdr_memseg_info* g_m)
{
  TRACE_PRINTF("GDR_MM cleanup for: phost_ptr: %p\n", (void*)g_m->phost_ptr);
  if(NULL != (void*)g_m->phost_ptr) {
    gdr_unmap(gdr_mem_manager::gdr_descr, g_m->pgdr_handle, (void*)g_m->phost_ptr, g_m->palloc_size);
  }
  gdr_unpin_buffer(gdr_mem_manager::gdr_descr, g_m->pgdr_handle);
  if(g_m->free_address) {
    cuMemFree((CUdeviceptr)g_m->free_address);
  }
}

// Maintains a per-device gdr-mem-manager
static std::unordered_map<int, gdr_mem_manager*> G;

gdr_mem_manager* get_gdr_mem_manager(int device_id) {
  if(G.find(device_id) == G.end()) {
    G[device_id] = new gdr_mem_manager(device_id);
  }
  return G[device_id];
} 
