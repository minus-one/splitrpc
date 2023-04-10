#include <cuda.h>          
#include <cuda_runtime.h>  

#define CUDA_DRIVER_API      
#include <helper_functions.h>
#include <helper_cuda.h>     

int main()
{
  int device_id = 0;
  CUdevice dev;                           

  checkCudaErrors(cudaSetDevice(device_id));
  checkCudaErrors(cudaFree(0));           
  checkCudaErrors(cuCtxGetDevice(&dev));  

  int StreamMemOpsSupport=1;                                                                                       
  checkCudaErrors(cuDeviceGetAttribute(&StreamMemOpsSupport, CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_MEM_OPS, dev));    
  if(StreamMemOpsSupport) {                                                                                        
    printf("Stream Mem OPs support present on device_id: %d\n", device_id);                                        
  } else {                                                                                                         
    printf("Stream Mem OPs support NOT! present on device_id: %d\n", device_id);                                   
  }                                                                                                                
  
  int supportsVMM=0;
  checkCudaErrors(cuDeviceGetAttribute(&supportsVMM, CU_DEVICE_ATTRIBUTE_VIRTUAL_ADDRESS_MANAGEMENT_SUPPORTED, dev));
  if(!supportsVMM) {
    printf("VMM support not present on this device_id: %d\n", device_id);
  } else {
    printf("CUDA VMM Support available on device_id: %d\n", device_id);
  }

  int RDMASupported = 0;
  checkCudaErrors(cuDeviceGetAttribute(&RDMASupported, CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WITH_CUDA_VMM_SUPPORTED, dev));
  if (!RDMASupported) {
    printf("GDR + VMM support not present on this device_id: %d\n", device_id);
  } else {
    printf("GDR + VMM support available on device_id: %d\n", device_id);
  }
  return 0;
}
