#! /bin/zsh

if [ $EUID != 0 ]; then
  sudo -E "$0" "$@"
  exit $?
fi

if [ -z ${P2P_RPC_BASE_PATH+1} ];
then
  echo "env var P2P_RPC_BASE_PATH is not set. please set and re-run the script"
  exit(1)
else
  echo "P2P_RPC_BASE_PATH is set to: ${P2P_RPC_BASE_PATH}"
fi

# Load all the LD paths
source ${P2P_RPC_BASE_PATH}/conf_scripts/env_config.rc
# Load device specific vars
source ${P2P_RPC_BASE_PATH}/conf_scripts/dev_config.rc
# Load app-configurations
source ${P2P_RPC_BASE_PATH}/conf_scripts/app_config.sh
# Specify the path to the binaries for this bench
export P2P_RPC_APP_PATH=${P2P_RPC_BASE_PATH}/rpc_bench/build

# P2P-RPC specific params
# 0 = Host, 1 = Device, 2 = Split
export P2P_RPC_DPDK_MEM_ALLOC_TYPE=2
# 1 = Kernel, 2 = CDP, 3 = P.T., 4 = CUDAGraphs
export P2P_RPC_WORK_LAUNCH_TYPE=1
# 0 = DMA, 1 = Kernel (Default)
export P2P_RPC_GPU_COPY_TYPE=1

# 0 = Copy performed (Default) 1 = ZC, no copy
export P2P_RPC_ZEROCOPY_MODE=0

# 0 = Sync server, 1 = Async server, 2 = Dynamic batching server
export P2P_RPC_SERVER_MODE=0

# Batch size to ONNXRuntime
export P2P_RPC_ORT_BATCH_SIZE=1

APP_CONFIG_NAME=$1
echo "APP_CONFIG: ${APP_CONFIG_NAME}"
SETAPPCONFIGS ${APP_CONFIG_NAME}
echo "P2P_RPC_MTU: ${P2P_RPC_MTU}, P2P_RPC_REQ_SIZE: ${P2P_RPC_REQ_SIZE}, P2P_RPC_RESP_SIZE: ${P2P_RPC_RESP_SIZE}"

#### Set the BIN_NAME
case ${APP_CONFIG_NAME} in
  echo_app)
    BIN_NAME=${P2P_RPC_APP_PATH}/echo_rpc_bench
    ;;
  resnet18)
    BIN_NAME=${P2P_RPC_APP_PATH}/ort_rpc_bench
    ;;
  resnet50)
    BIN_NAME=${P2P_RPC_APP_PATH}/ort_rpc_bench
    ;;
  bert)
    #export P2P_RPC_ZEROCOPY_MODE=1
    #export P2P_RPC_SERVER_MODE=0
    BIN_NAME=${P2P_RPC_APP_PATH}/ort_rpc_bench
    ;;
  vecadd1k)
    export P2P_RPC_WORK_LAUNCH_TYPE=3
    export P2P_RPC_SERVER_MODE=0
    export P2P_RPC_ORT_BATCH_SIZE=1
    export P2P_RPC_ZEROCOPY_MODE=1
    BIN_NAME=${P2P_RPC_APP_PATH}/vec_add_rpc_bench
    ;;
  vecadd1M)
    export P2P_RPC_WORK_LAUNCH_TYPE=4
    export P2P_RPC_SERVER_MODE=0
    BIN_NAME=${P2P_RPC_APP_PATH}/vec_add_rpc_bench
    ;;
  lenet)
    export P2P_RPC_ZEROCOPY_MODE=1
    export P2P_RPC_SERVER_MODE=0
    BIN_NAME=${P2P_RPC_APP_PATH}/lenet_rpc_bench
    ;;
  lstm)
    export P2P_RPC_ZEROCOPY_MODE=1
    export P2P_RPC_SERVER_MODE=0
    BIN_NAME=${P2P_RPC_APP_PATH}/lstm_rpc_bench
    ;;
  matmul16)
    export P2P_RPC_ZEROCOPY_MODE=1
    export P2P_RPC_SERVER_MODE=0
    BIN_NAME=${P2P_RPC_APP_PATH}/mm_rpc_bench
    ;;
  matmul32)
    export P2P_RPC_ZEROCOPY_MODE=1
    export P2P_RPC_SERVER_MODE=0
    export P2P_RPC_WORK_LAUNCH_TYPE=3
    BIN_NAME=${P2P_RPC_APP_PATH}/mm_rpc_bench
    ;;
  matmul)
    BIN_NAME=${P2P_RPC_APP_PATH}/mm_rpc_bench
    ;;
  *)
    BIN_NAME=${P2P_RPC_APP_PATH}/echo_rpc_bench
    ;;
esac

echo "Starting ${BIN_NAME} on NUMA: ${NUMA_NODE}"

### This can be used a temporary override to start on a device
#export P2P_RPC_DPDK_PORT=0
#numactl --physcpubind ${NUMA_CORE_MASK} -m ${NUMA_NODE} ${BIN_NAME} -l ${DPDK_LCORES} -a db:00.0,rxq_pkt_pad_en=1

# Actual command
numactl -a --physcpubind ${NUMA_CORE_MASK} -m ${NUMA_NODE} ${BIN_NAME} -l ${DPDK_LCORES}

# Debug command
#export DEBUG_APP=/usr/local/cuda-11.4/bin/cuda-gdb
#export DEBUG_APP=gdb
#numactl --physcpubind ${NUMA_CORE_MASK} -m ${NUMA_NODE} ${DEBUG_APP} --args ${BIN_NAME} -l ${DPDK_LCORES}

# Profile command
#export NSYS_PATH=/opt/nvidia/nsight-systems/2021.4.1/bin
#export NSYS_FILE_NAME=nsys_${APP_CONFIG_NAME}_sync
###numactl -a --physcpubind ${NUMA_CORE_MASK} -m ${NUMA_NODE} 
#${NSYS_PATH}/nsys profile -t cuda,nvtx -o ${NSYS_FILE_NAME}.qdrep --force-overwrite=true ${BIN_NAME}
 
# Command to collect results (run separately)
#/opt/nvidia/nsight-systems/2021.4.1/bin/nsys stats\
#  ${NSYS_FILE_NAME}.qdrep\
#  -r gpukernsum,nvtxsum -f csv\
#  -o ${NSYS_FILE_NAME}_stats 
