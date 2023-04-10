#! /bin/zsh

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
export P2P_RPC_APP_PATH=${P2P_RPC_BASE_PATH}/local_bench/build

# 1 = Kernel, 2 = CDP, 3 = P.T., 4 = CUDAGraphs
export P2P_RPC_WORK_LAUNCH_TYPE=1

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
    BIN_NAME=${P2P_RPC_APP_PATH}/ort_rpc_bench
    ;;
  vecadd1k)
    BIN_NAME=${P2P_RPC_APP_PATH}/vec_add_rpc_bench
    ;;
  vecadd1M)
    BIN_NAME=${P2P_RPC_APP_PATH}/vec_add_rpc_bench
    ;;
  lenet)
    BIN_NAME=${P2P_RPC_APP_PATH}/lenet_rpc_bench
    ;;
  lstm)
    BIN_NAME=${P2P_RPC_APP_PATH}/lstm_rpc_bench
    ;;
  matmul16)
    BIN_NAME=${P2P_RPC_APP_PATH}/mm_rpc_bench
    ;;
  matmul32)
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

# Actual command
numactl -a --physcpubind ${NUMA_CORE_MASK} -m ${NUMA_NODE} ${BIN_NAME}

# Debug command
#export DEBUG_APP=/usr/local/cuda-11.4/bin/cuda-gdb
#export DEBUG_APP=gdb
#numactl --physcpubind ${NUMA_CORE_MASK} -m ${NUMA_NODE} ${DEBUG_APP} --args ${BIN_NAME} -l ${DPDK_LCORES}

# Profile command
#export NSYS_PATH=/opt/nvidia/nsight-systems/2021.4.1/bin
#export NSYS_FILE_NAME=nsys_${APP_CONFIG_NAME}
#####numactl -a --physcpubind ${NUMA_CORE_MASK} -m ${NUMA_NODE} 
#${NSYS_PATH}/nsys profile -o ${NSYS_FILE_NAME}.qdrep --force-overwrite=true ${BIN_NAME}
 
# Command to collect results (run separately)
#/opt/nvidia/nsight-systems/2021.4.1/bin/nsys stats\
#  ${NSYS_FILE_NAME}.qdrep\
#  -r gpukernsum,nvtxsum -f csv\
#  -o ${NSYS_FILE_NAME}_stats 
