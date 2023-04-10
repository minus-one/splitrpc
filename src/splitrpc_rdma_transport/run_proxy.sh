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

#export DEBUG_APP=/usr/local/cuda-11.4/bin/cuda-gdb
export DEBUG_APP=gdb

# Load all the LD paths
source ${P2P_RPC_BASE_PATH}/conf_scripts/env_config.rc
# Load device specific vars
source ${P2P_RPC_BASE_PATH}/conf_scripts/dev_config.rc
# Load app-configurations
source ${P2P_RPC_BASE_PATH}/conf_scripts/app_config.sh
# Specify the path to the binaries for this bench
export P2P_RPC_APP_PATH=${P2P_RPC_BASE_PATH}/rdma_transport/build

# P2P-RPC specific params
# 0 = Host, 1 = Device, 2 = Split
export P2P_RPC_DPDK_MEM_ALLOC_TYPE=2
# 0 = Copy performed (Default) 1 = ZC, no copy
export P2P_RPC_ZEROCOPY_MODE=0
export P2P_RPC_CUDA_DEVICE_ID=-1

APP_CONFIG_NAME=$1
echo "APP_CONFIG: ${APP_CONFIG_NAME}"
SETAPPCONFIGS ${APP_CONFIG_NAME}
echo "P2P_RPC_MTU: ${P2P_RPC_MTU}, P2P_RPC_REQ_SIZE: ${P2P_RPC_REQ_SIZE}, P2P_RPC_RESP_SIZE: ${P2P_RPC_RESP_SIZE}"

#### Set the BIN_NAME
#BIN_NAME=${P2P_RPC_APP_PATH}/gpu_dpdk_rdma_proxy_handler
BIN_NAME=${P2P_RPC_APP_PATH}/gpu_rdma_proxy_handler

echo "Enabling LIBVMA"
export LD_PRELOAD=libvma.so

echo "Starting ${BIN_NAME} on NUMA: ${NUMA_NODE}"

#numactl --physcpubind ${NUMA_CORE_MASK} -m ${NUMA_NODE} ${BIN_NAME} -a 192.168.25.1 -p 20886
${BIN_NAME} -a 192.168.25.1 -p 20886
#${BIN_NAME} -l ${DPDK_LCORES} 
