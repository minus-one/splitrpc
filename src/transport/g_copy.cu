#include "g_copy.cuh"
#include "p2p_rpc.h"
#include "config_utils.h"
#ifdef PROFILE_MODE
#include <nvToolsExt.h>
#endif

#define MAX_QUEUE_SIZE 32

#define SCATTER_GATHER_TB_SZ 1024

__launch_bounds__(SCATTER_GATHER_TB_SZ) __global__ void CopyKernelSingleTB(        
    g_copy_params *_stub,
    volatile uint32_t *door_bell)
{
  const uintptr_t* __restrict input_buf_ptrs = _stub->input_buf_ptrs; 
  uintptr_t* __restrict output_buf_ptrs = _stub->output_buf_ptrs;
  const size_t* __restrict byte_buf_size = _stub->byte_buf_size;

  int lane_id = threadIdx.x;
  uint8_t* __restrict input_buffer;
  size_t byte_size;
  uint8_t* __restrict output_buffer;

  for(int buf_idx = 0; buf_idx < _stub->num_items ; buf_idx++) {
      input_buffer = (uint8_t*)input_buf_ptrs[buf_idx];
      byte_size = byte_buf_size[buf_idx];
      output_buffer = (uint8_t*)output_buf_ptrs[buf_idx];

    if (((byte_size % 4) == 0) && (((uint64_t)input_buffer % 4) == 0) &&
        (((uint64_t)output_buffer % 4) == 0)) {
      int32_t* input_4 = (int32_t*)input_buffer;
      int32_t* output_4 = (int32_t*)output_buffer;
      int element_count = byte_size / 4;
      for (int elem_id = lane_id; elem_id < element_count;
          elem_id += SCATTER_GATHER_TB_SZ) {
        output_4[elem_id] = input_4[elem_id];
      }
    } else {
      for (int elem_id = lane_id; elem_id < byte_size;                           
          elem_id += SCATTER_GATHER_TB_SZ) {                                        
        output_buffer[elem_id] =                                     
          __ldg(input_buffer + elem_id);                               
      }
    }
  }

/*
  for(int buf_idx = 0 ; buf_idx < _stub->num_items ; buf_idx++) {
    const uint8_t* input_buffer = (uint8_t*)input_buf_ptrs[buf_idx];
    int byte_size = byte_buf_size[buf_idx];
    uint8_t* output_buffer = (uint8_t*)output_buf_ptrs[buf_idx];

     if (((byte_size % 4) == 0) && (((uint64_t)input_buffer % 4) == 0) &&
        (((uint64_t)output_buffer % 4) == 0)) {
      int32_t* input_4 = (int32_t*)input_buffer;
      int32_t* output_4 = (int32_t*)output_buffer;
      int element_count = byte_size / 4;
      for (int elem_id = lane_id; elem_id < element_count;
          elem_id += SCATTER_GATHER_TB_SZ) {
        output_4[elem_id] = input_4[elem_id];
      }
    } else {
      for (int elem_id = lane_id; elem_id < byte_size;                           
          elem_id += SCATTER_GATHER_TB_SZ) {                                        
        output_buffer[elem_id] =                                     
          __ldg(input_buffer + elem_id);                               
      }
    }
  }
  */
}

__launch_bounds__(SCATTER_GATHER_TB_SZ) __global__ void CopyKernel(        
    g_copy_params *_stub,
    volatile uint32_t *door_bell)
{
  const uintptr_t* __restrict input_buf_ptrs = _stub->input_buf_ptrs; 
  const size_t* __restrict byte_buf_size = _stub->byte_buf_size;
  uintptr_t* __restrict output_buf_ptrs = _stub->output_buf_ptrs;

  int buf_idx = blockIdx.x;
  const uint8_t* input_buffer = (uint8_t*)input_buf_ptrs[buf_idx];
  size_t byte_size = byte_buf_size[buf_idx];
  uint8_t* output_buffer = (uint8_t*)output_buf_ptrs[buf_idx];
   
  int lane_id = threadIdx.x;
  if (((byte_size % 4) == 0) && (((uint64_t)input_buffer % 4) == 0) &&
      (((uint64_t)output_buffer % 4) == 0)) {
    int32_t* input_4 = (int32_t*)input_buffer;
    int32_t* output_4 = (int32_t*)output_buffer;
    int element_count = byte_size / 4;
    for (int elem_id = lane_id; elem_id < element_count;
         elem_id += SCATTER_GATHER_TB_SZ) {
      output_4[elem_id] = input_4[elem_id];
    }
  } else {
    for (int elem_id = lane_id; elem_id < byte_size;                           
         elem_id += SCATTER_GATHER_TB_SZ) {                                        
      output_buffer[elem_id] =                                     
          __ldg(input_buffer + elem_id);                               
    }
  }
}

// Launches a kernel from the device side to do cudamemcpyasync
__launch_bounds__(SCATTER_GATHER_TB_SZ) __global__ void CopyKernelAsync(        
    g_copy_params *_stub)
{
  const uintptr_t* __restrict input_buf_ptrs = _stub->input_buf_ptrs; 
  const size_t* __restrict byte_buf_size = _stub->byte_buf_size;
  uintptr_t* __restrict output_buf_ptrs = _stub->output_buf_ptrs;

  int buf_idx = threadIdx.x;
  //int buf_idx = blockIdx.x;
  const uint8_t* input_buffer = (uint8_t*)input_buf_ptrs[buf_idx];
  size_t byte_size = byte_buf_size[buf_idx];
  uint8_t* output_buffer = (uint8_t*)output_buf_ptrs[buf_idx];

  cudaStream_t s;
  cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
  cudaMemcpyAsync((void*)output_buffer, (void*)input_buffer, byte_size, cudaMemcpyDeviceToDevice, s);
}

#ifdef __cplusplus                                                             
extern "C" {                                                                   
#endif                                                                         

CopyCtx* 
init_copy_ctx_on_stream(cudaStream_t work_stream)
{
  CopyCtx *new_copy_ctx = new CopyCtx;
  new_copy_ctx->launch_type = get_gpu_copy_type();
  new_copy_ctx->work_stream = work_stream;
  checkCudaErrors(cudaEventCreateWithFlags(&new_copy_ctx->work_complete, cudaEventDisableTiming));

  new_copy_ctx->h_stub = BufItemPool<g_copy_params>::create_buf_item_pool(2, get_cuda_device_id());
  new_copy_ctx->d_stub = BufItemPool<g_copy_params>::get_dev_ptr(new_copy_ctx->h_stub);
  new_copy_ctx->door_bell = BufItemPool<uint32_t>::create_buf_item_pool(2, get_cuda_device_id());
  new_copy_ctx->d_door_bell = BufItemPool<uint32_t>::get_dev_ptr(new_copy_ctx->door_bell);

  ACCESS_ONCE(*(new_copy_ctx->door_bell)) = 0;
  ACCESS_ONCE(*(new_copy_ctx->door_bell + 1)) = 0;
  _mm_mfence();

  if(new_copy_ctx->launch_type == 0) {
    printf("G_COPY: Configured to use GPU DMA engine on stream %p\n", (void*)work_stream);
  } else if(new_copy_ctx->launch_type == 1) {
    printf("G_COPY: Configured to launch work kernels on stream %p\n", (void*)work_stream);
  } else if(new_copy_ctx->launch_type == 2) {
    //dim3 blockSize(SCATTER_GATHER_TB_SZ, 1, 1);
    //dim3 gridSize(1, 1);
    //g_copy_params *d_stub = BufItemPool<g_copy_params>::get_dev_ptr(new_copy_ctx->_stub);
    //uint32_t *d_door_bell = BufItemPool<uint32_t>::get_dev_ptr(new_copy_ctx->door_bell);
    printf("G_COPY: persistent kernels unimplemented\n");
  } 
  return new_copy_ctx;
}

CopyCtx* init_copy_ctx() {
  cudaStream_t stream;
  checkCudaErrors(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  return init_copy_ctx_on_stream(stream);
}

CopyCtx *default_copy_ctx = NULL;
CopyCtx* 
get_default_copy_ctx()
{
  if(default_copy_ctx != NULL)
    return default_copy_ctx;
  default_copy_ctx = init_copy_ctx();
  return default_copy_ctx;
}

int
sg_on_gpu(CopyCtx *ctx,
    p2p_sk_buf *skb,
    int instance)
{
  TRACE_PRINTF("SG: CopyCtx: %p, SKB: %p\n", (void*)ctx, (void*)skb);
  if(ctx == NULL)
    ctx = get_default_copy_ctx();

  if(ctx->launch_type == 0) {
    for(int i = 0; i < skb->num_items; i++) {
      checkCudaErrors(
          cudaMemcpyAsync(
            (void*)skb->o_buf[i], 
            (void*)skb->i_buf[i], 
            skb->len[i], 
            cudaMemcpyDeviceToDevice, ctx->work_stream));
    }
  } else if(ctx->launch_type == 1) {
    std::memcpy((void*)(ctx->h_stub + instance), skb, sizeof(g_copy_params));
    _mm_mfence();
    CopyKernelSingleTB<<< 1, SCATTER_GATHER_TB_SZ, 0, ctx->work_stream>>>(
        (g_copy_params*)(ctx->d_stub + instance), ctx->d_door_bell + instance);
  } else if(ctx->launch_type == 2) {
    std::memcpy(ctx->h_stub + instance, skb, sizeof(g_copy_params));
    _mm_mfence();
    CopyKernel<<<skb->num_items, SCATTER_GATHER_TB_SZ, 0, ctx->work_stream>>>(
        ctx->d_stub + instance, ctx->d_door_bell + instance);
  } else if(ctx->launch_type == 3) {
    std::memcpy(ctx->h_stub + instance, skb, sizeof(g_copy_params));
    _mm_mfence();
    CopyKernelAsync<<<1, skb->num_items, 0, ctx->work_stream>>>(ctx->d_stub + instance);
  }
  cudaError_t err = cudaGetLastError();
  if ( err != cudaSuccess ) {
    printf("CUDA Error: %s\n", cudaGetErrorString(err));
    return 0;
  }
  return 1;
}

// Caller must synchronize
int
gather_on_gpu(CopyCtx *ctx,
    p2p_bufs *buf_ptrs,
    uint8_t *io_buf)
{
#ifdef PROFIL_MODE
nvtxRangePush("gather-on-gpu");
#endif
  if(ctx == NULL)
    ctx = get_default_copy_ctx();

  TRACE_PRINTF("GATHER: %d bufs into %p\n", buf_ptrs->num_items, (void*)io_buf);

  size_t offset = 0;
  if(ctx->launch_type == 0) {
    // Use GPU's DMA engines
    for(int i = 0; i < buf_ptrs->num_items; i++) {
      checkCudaErrors(cudaMemcpyAsync((void*)(io_buf + offset), (void*)buf_ptrs->burst_items[i], buf_ptrs->item_size[i], 
          cudaMemcpyDefault, ctx->work_stream));
      offset += buf_ptrs->item_size[i]; 
    }
    // Mark copy to be complete
    //checkCudaErrors(cudaEventRecord(ctx->copy_complete, ctx->g_copy_stream));
  } else if(ctx->launch_type == 1) {
    // Use Copy Kernel
    g_copy_params tmp;
    tmp.num_items = 0;
    uintptr_t w_addr = (uintptr_t)(io_buf);
    tmp.input_buf_ptrs[tmp.num_items] = buf_ptrs->burst_items[0];
    tmp.byte_buf_size[tmp.num_items] = buf_ptrs->item_size[0];
    tmp.output_buf_ptrs[tmp.num_items] = w_addr; 
    w_addr += buf_ptrs->item_size[0]; 
    uintptr_t last_end_ptr = buf_ptrs->burst_items[0] + buf_ptrs->item_size[0];

    for(int i = 1 ; i < buf_ptrs->num_items ; i++) {
      if(buf_ptrs->burst_items[i] == last_end_ptr) {
        // extend
        tmp.byte_buf_size[tmp.num_items] += buf_ptrs->item_size[i];
        last_end_ptr += buf_ptrs->item_size[i];
        w_addr += buf_ptrs->item_size[i];
      } else {
        // stop and start new
        tmp.num_items++;
        tmp.input_buf_ptrs[tmp.num_items] = buf_ptrs->burst_items[i];
        tmp.byte_buf_size[tmp.num_items] = buf_ptrs->item_size[i];
        tmp.output_buf_ptrs[tmp.num_items] = w_addr; 
        w_addr += buf_ptrs->item_size[i];
        last_end_ptr = buf_ptrs->burst_items[i] + buf_ptrs->item_size[i];
      }
    }
    tmp.num_items++;
    TRACE_PRINTF("GATHER: %d bufs into %p\n", tmp.num_items, (void*)io_buf);
    //for(int i = 0 ; i < tmp.num_items; i++) {
    //  checkCudaErrors(cudaMemcpyAsync((void*)tmp.output_buf_ptrs[i], (void*)(tmp.input_buf_ptrs[i]), 
    //        tmp.byte_buf_size[i], cudaMemcpyDefault, ctx->g_copy_stream));
    //}
    std::memcpy(ctx->h_stub, &tmp, sizeof(g_copy_params));
    _mm_mfence();

    //CopyKernelAsync<<<1, tmp.num_items, 0, ctx->g_copy_stream>>>(ctx->d_stub);
    //CopyKernel<<<tmp.num_items, SCATTER_GATHER_TB_SZ, 0, ctx->g_copy_stream>>>(
    CopyKernelSingleTB<<< 1, SCATTER_GATHER_TB_SZ, 0, ctx->work_stream>>>(
        ctx->d_stub,
        ctx->d_door_bell);
    
    //cudaError_t err = cudaGetLastError();
    //if ( err != cudaSuccess ) {
    //  printf("CUDA Error: %s\n", cudaGetErrorString(err));
    //  exit(1);
    //}
  }

#ifdef PROFILE_MODE
  nvtxRangePop();
#endif
  return offset;
}

// FIXME: This is probably not the right way to do this
// Caller must synchronize
int 
scatter_on_gpu(CopyCtx *ctx,
    p2p_bufs *buf_ptrs,
    uint8_t *io_buf,
    int io_buf_size)
{
#ifdef PROFILE_MODE
  nvtxRangePush("scatter_on_gpu");
#endif
  if(ctx == NULL)
    ctx = get_default_copy_ctx();

  TRACE_PRINTF("SCATTER: %p into %d bufs\n", (void*)io_buf, buf_ptrs->num_items);

  size_t offset = 0;
  int curr_pkt_size = 0;

  if(ctx->launch_type == 0) {
    for(int i = 0 ; i < buf_ptrs->num_items; i++) {
      curr_pkt_size = (io_buf_size <= RPC_MTU) ? io_buf_size : RPC_MTU;
      io_buf_size -= curr_pkt_size;
      checkCudaErrors(cudaMemcpyAsync((void*)buf_ptrs->burst_items[i], (void*)(io_buf + offset), curr_pkt_size, 
          cudaMemcpyDefault, ctx->work_stream));
      buf_ptrs->item_size[i] = curr_pkt_size;
      offset += curr_pkt_size;
    }
    // Mark copy to be complete
    //checkCudaErrors(cudaEventRecord(ctx->work_complete, ctx->work_stream));
  } else if (ctx->launch_type == 1) {
    g_copy_params tmp;
    tmp.num_items = 0;
    uintptr_t r_addr = (uintptr_t)(io_buf);
    curr_pkt_size = (io_buf_size <= RPC_MTU) ? io_buf_size : RPC_MTU;
    tmp.input_buf_ptrs[tmp.num_items] = r_addr; 
    tmp.byte_buf_size[tmp.num_items] = curr_pkt_size;
    tmp.output_buf_ptrs[tmp.num_items] = buf_ptrs->burst_items[0];
    r_addr += curr_pkt_size;
    io_buf_size -= curr_pkt_size;
    uintptr_t last_end_ptr = buf_ptrs->burst_items[0] + curr_pkt_size;

    for(int i = 1 ; i < buf_ptrs->num_items ; i++) {
      curr_pkt_size = (io_buf_size <= RPC_MTU) ? io_buf_size : RPC_MTU;
      if(buf_ptrs->burst_items[i] == last_end_ptr) {
        tmp.byte_buf_size[tmp.num_items] += curr_pkt_size;
        last_end_ptr += curr_pkt_size;
      } else {
        tmp.num_items++;
        tmp.input_buf_ptrs[tmp.num_items] = r_addr; 
        tmp.byte_buf_size[tmp.num_items] = curr_pkt_size;
        tmp.output_buf_ptrs[tmp.num_items] = buf_ptrs->burst_items[i];
        last_end_ptr = buf_ptrs->burst_items[i] + curr_pkt_size;
      }
      r_addr += curr_pkt_size;
      io_buf_size -= curr_pkt_size;
    }
    tmp.num_items++;
    TRACE_PRINTF("SCATTER: %p into %d bufs\n", (void*)io_buf, tmp.num_items);
    //for(int i = 0 ; i < tmp.num_items; i++) {
    //  checkCudaErrors(cudaMemcpyAsync((void*)tmp.output_buf_ptrs[i], (void*)(tmp.input_buf_ptrs[i]), 
    //        tmp.byte_buf_size[i], cudaMemcpyDefault, ctx->work_stream));
    //}
    std::memcpy(ctx->h_stub+1, &tmp, sizeof(g_copy_params));
    _mm_mfence();
    //CopyKernelAsync<<<1, tmp.num_items, 0, ctx->work_stream>>>(d_stub);
    //CopyKernel<<<tmp.num_items, SCATTER_GATHER_TB_SZ, 0, ctx->work_stream>>>(
    CopyKernelSingleTB<<< 1, SCATTER_GATHER_TB_SZ, 0, ctx->work_stream>>>(
        ctx->d_stub+1,
        ctx->d_door_bell+1);

    //cudaError_t err = cudaGetLastError();
    //if ( err != cudaSuccess ) {
    //  printf("CUDA Error: %s\n", cudaGetErrorString(err));
    //  exit(1);
    //}
  } else {
    printf("Warning! Unimplemented copy stream. doing no copy\n");
  }

#ifdef PROFILE_MODE
nvtxRangePop();
#endif
  return offset;
}

__launch_bounds__(SCATTER_GATHER_TB_SZ) __global__ void SetDummyDataKernel(void *start_addr, int len, uint8_t dummy_value) {
  int lane_id = threadIdx.x;
  uint8_t* buf_start = (uint8_t*)start_addr;
  for(int elem_id = lane_id ; elem_id < len; elem_id += SCATTER_GATHER_TB_SZ) {
    buf_start[elem_id] = dummy_value;
  }
}

void SetDummyData(void *start_addr, int len, uint8_t dummy_value) {
  printf("Setting dummy data for %p, len: %d, value: %d\n", start_addr, len, dummy_value);
  SetDummyDataKernel<<<1, SCATTER_GATHER_TB_SZ, 0, 0>>>(start_addr, len, dummy_value);
  checkCudaErrors(cudaStreamSynchronize(0));
}

/*************************************** OBSOLETE APIS ******************************************/

//__launch_bounds__(SCATTER_GATHER_TB_SZ) __global__ void GatherKernel(        
//    const uintptr_t* __restrict input_buf_ptrs,                            
//    const size_t* __restrict byte_buf_size,
//    const size_t* __restrict byte_offsets,
//    uint8_t* __restrict output_buffer)
//{
//  int buf_idx = blockIdx.x;
//  int lane_id = threadIdx.x;
//  const uint8_t* request_input_buffer = (uint8_t*)input_buf_ptrs[buf_idx];
//  int byte_size = byte_buf_size[buf_idx];
//  int byte_size_offset = byte_offsets[buf_idx];
//  uint8_t* output_buffer_with_offset = output_buffer + byte_size_offset;
//
//  if (((byte_size % 4) == 0) && (((uint64_t)request_input_buffer % 4) == 0) &&
//      (((uint64_t)output_buffer_with_offset % 4) == 0)) {
//    int32_t* input_4 = (int32_t*)request_input_buffer;
//    int32_t* output_4 = (int32_t*)output_buffer_with_offset;
//    int element_count = byte_size / 4;
//    for (int elem_id = lane_id; elem_id < element_count;
//         elem_id += SCATTER_GATHER_TB_SZ) {
//      output_4[elem_id] = input_4[elem_id];
//    }
//  } else {
//    for (int elem_id = lane_id; elem_id < byte_size;                           
//         elem_id += SCATTER_GATHER_TB_SZ) {                                        
//      output_buffer_with_offset[elem_id] =                                     
//          __ldg(request_input_buffer + elem_id);                               
//    }
//  }
//}
//                                                                               
//__launch_bounds__(SCATTER_GATHER_TB_SZ) __global__ void ScatterKernel(        
//    const uint8_t* __restrict input_buffer,
//    size_t* __restrict byte_offsets,
//    uintptr_t* __restrict output_ptr_buffer,                            
//    size_t* __restrict byte_buf_size)
//{
//  int request_idx = blockIdx.x;
//  int lane_id = threadIdx.x;
//  uint8_t* request_output_buffer = (uint8_t*)output_ptr_buffer[request_idx];          
//  int byte_size = byte_buf_size[request_idx];                               
//  int byte_offset = byte_offsets[request_idx];                 
//                                                                               
//  const uint8_t* input_buffer_with_offset = input_buffer + byte_offset;        
//  if (((byte_size % 4) == 0) && (((uint64_t)request_output_buffer % 4) == 0) && 
//      (((uint64_t)input_buffer_with_offset % 4) == 0)) {                      
//    int32_t* output_4 = (int32_t*)request_output_buffer;                         
//    int32_t* input_4 = (int32_t*)input_buffer_with_offset;                   
//    int element_count = byte_size / 4;                                         
//    for (int elem_id = lane_id; elem_id < element_count;                       
//         elem_id += SCATTER_GATHER_TB_SZ) {                                        
//      output_4[elem_id] = input_4[elem_id];                                    
//    }                                                                          
//  } else {                                                                     
//    for (int elem_id = lane_id; elem_id < byte_size;                           
//         elem_id += SCATTER_GATHER_TB_SZ) {                                        
//             request_output_buffer[elem_id] = input_buffer_with_offset[elem_id]; 
//    }                                                                          
//  }                                                                            
//}
//
///*
//* input_bufs: Input p2p buffers containing burst-items and their sizes
//* output_buffer: The output buffer into which to write the data
//* byte_offsets: Represents the offsets for each of the input_bufs into the output_buffer
//*/
//cudaError_t
//RunGatherKernel(
//    p2p_bufs *d_input_bufs,
//    size_t *byte_offsets,
//    size_t request_count,
//    uint8_t *output_buffer,
//    cudaStream_t stream)
//{
//    const uintptr_t *input_buf_ptrs = d_input_bufs->burst_items;
//    const size_t *byte_buf_size = d_input_bufs->item_size;
//  GatherKernel<<<request_count, SCATTER_GATHER_TB_SZ, 0, stream>>>(
//      input_buf_ptrs, byte_buf_size,  byte_offsets, output_buffer);
//  return cudaGetLastError();
//}
//
//cudaError_t                                                                    
//RunScatterKernel(                                                               
//    uint8_t *input_buffer,
//    size_t *byte_offsets, 
//    p2p_bufs *d_output_bufs,
//    size_t request_count,
//    cudaStream_t stream)
//{
//    uintptr_t *output_ptr_buffer = d_output_bufs->burst_items;
//    size_t *byte_buf_size = d_output_bufs->item_size;
//  ScatterKernel<<<request_count, SCATTER_GATHER_TB_SZ, 0, stream>>>(          
//      input_buffer, byte_offsets, output_ptr_buffer, byte_buf_size);                                                          
//  return cudaGetLastError();                                                   
//}
                                                                               
// Gathers a bunch of memory into a single place [blocking call]
//int gather_on_gpu_sync(CopyCtx *ctx,
//    p2p_bufs *buf_ptrs,
//    uint8_t *io_buf)
//{
//  size_t offset = 0;
//  if(ctx->launch_type >= 0 && ctx->launch_type < 3) {
//    for(int i = 0 ; i < buf_ptrs->num_items; i++) {
//      if(ctx->launch_type == 0) {
//        // Call is synchronous
//        cuMemcpyDtoD((CUdeviceptr)(io_buf + offset), (CUdeviceptr)buf_ptrs->burst_items[i], buf_ptrs->item_size[i]);
//        //cuMemcpyHtoD((CUdeviceptr)(io_buf + offset), (const void*)buf_ptrs->burst_items[i], buf_ptrs->item_size[i]);
//      } else if(ctx->launch_type == 1) {
//        cuMemcpyDtoDAsync((CUdeviceptr)(io_buf + offset), (CUdeviceptr)buf_ptrs->burst_items[i], buf_ptrs->item_size[i], ctx->work_stream);
//        //cuMemcpyHtoDAsync((CUdeviceptr)(io_buf + offset), (const void*)buf_ptrs->burst_items[i], buf_ptrs->item_size[i], ctx->work_stream);
//      } else if(ctx->launch_type == 2) {
//        cudaMemcpyAsync((void*)(io_buf + offset), (void*)buf_ptrs->burst_items[i], buf_ptrs->item_size[i], 
//            cudaMemcpyDefault, ctx->work_stream);
//      } 
//      offset += buf_ptrs->item_size[i]; 
//    }
//  } else if(ctx->launch_type == 3) {
//    // Use Copy Kernel
//    int last_idx = 0;
//    ctx->h_stub->input_buf_ptrs[last_idx] = buf_ptrs->burst_items[0];
//    ctx->h_stub->byte_buf_size[last_idx] = buf_ptrs->item_size[0];
//    ctx->h_stub->output_buf_ptrs[last_idx] = (uintptr_t)(io_buf);
//    offset += buf_ptrs->item_size[0]; 
//    uintptr_t last_end_ptr = buf_ptrs->burst_items[0] + buf_ptrs->item_size[0];
//
//    for(int i = 1 ; i < buf_ptrs->num_items ; i++) {
//      if(buf_ptrs->burst_items[i] == last_end_ptr) {
//        // extend
//        ctx->h_stub->byte_buf_size[last_idx] += buf_ptrs->item_size[i];
//        last_end_ptr += buf_ptrs->item_size[i];
//        offset += buf_ptrs->item_size[i];
//      } else {
//        // stop and start new
//        last_idx++;
//        ctx->h_stub->input_buf_ptrs[last_idx] = buf_ptrs->burst_items[i];
//        ctx->h_stub->byte_buf_size[last_idx] = buf_ptrs->item_size[i];
//        ctx->h_stub->output_buf_ptrs[last_idx] = (uintptr_t)(io_buf + offset);
//        offset += buf_ptrs->item_size[i];
//        last_end_ptr = buf_ptrs->burst_items[i] + buf_ptrs->item_size[i];
//      }
//      //ctx->h_stub->input_buf_ptrs[i] = buf_ptrs->burst_items[i];
//      //ctx->h_stub->byte_buf_size[i] = buf_ptrs->item_size[i];
//      //ctx->h_stub->output_buf_ptrs[i] = (uintptr_t)(io_buf + offset);
//      //offset += buf_ptrs->item_size[i]; 
//    }
//
//
//    CopyKernel<<<buf_ptrs->num_items, SCATTER_GATHER_TB_SZ, 0, ctx->work_stream>>>(
//        ctx->d_stub,
//        ctx->d_door_bell);
//  } else if(ctx->launch_type == 4) {
//    for(int i = 0 ; i < buf_ptrs->num_items ; i++) {
//      ctx->h_stub->input_buf_ptrs[i] = buf_ptrs->burst_items[i];
//      ctx->h_stub->byte_buf_size[i] = buf_ptrs->item_size[i];
//      ctx->h_stub->output_buf_ptrs[i] = (uintptr_t)(io_buf + offset);
//      offset += buf_ptrs->item_size[i]; 
//    }
//    dim3 gridSize(1, 1);
//    CopyKernelAsync<<<gridSize, buf_ptrs->num_items, 0, ctx->work_stream>>>(
//        ctx->d_stub);
//  }
//
//  if(ctx->launch_type > 0)
//    cudaStreamSynchronize(ctx->work_stream);
//  return offset;
//}

#ifdef __cplusplus                                                             
}                                                                              
#endif                                                                         
