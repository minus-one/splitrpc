#! /bin/zsh

export NSYS_PATH=/opt/nvidia/nsight-systems/2021.4.1/bin

source common_server_config.rc

#export NSYS_FILE_NAME=${P2P_RPC_DPDK_MEM_ALLOC_TYPE}_1024
#export NSYS_FILE_NAME=dpdk_gpu_only
export NSYS_FILE_NAME=nsys_profile_rpc_bench_`basename ${BIN_NAME}`

numactl -N ${NUMA_NODE} -l \
  ${NSYS_PATH}/nsys profile -o \
  ${NSYS_FILE_NAME}.qdrep \
  --force-overwrite=true \
  ${BIN_NAME}
  
# Command to collect results (run separately)
#/opt/nvidia/nsight-systems/2021.4.1/bin/nsys stats\
#  ${NSYS_FILE_NAME}.qdrep\
#  -r gpukernsum,nvtxsum -f csv\
#  -o ${NSYS_FILE_NAME}_stats 
