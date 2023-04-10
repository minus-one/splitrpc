#! /bin/zsh

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
export P2P_RPC_APP_PATH=${P2P_RPC_BASE_PATH}/grpc_bench/build

# 1 = Kernel, 2 = CDP, 3 = P.T., 4 = CUDAGraphs
export P2P_RPC_WORK_LAUNCH_TYPE=1

# Batch size to ONNXRuntime
export P2P_RPC_ORT_BATCH_SIZE=1

APP_CONFIG_NAME=$1
echo "APP_CONFIG: ${APP_CONFIG_NAME}"
SETAPPCONFIGS ${APP_CONFIG_NAME}
echo "P2P_RPC_MTU: ${P2P_RPC_MTU}, P2P_RPC_REQ_SIZE: ${P2P_RPC_REQ_SIZE}, P2P_RPC_RESP_SIZE: ${P2P_RPC_RESP_SIZE}"

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
    export P2P_RPC_WORK_LAUNCH_TYPE=4
    BIN_NAME=${P2P_RPC_APP_PATH}/vec_add_rpc_bench
    ;;
  vecadd1M)
    export P2P_RPC_WORK_LAUNCH_TYPE=4
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

LIBVMA_SET=$2
case ${LIBVMA_SET} in
  libvma)
    echo "Enabling LIBVMA"
    export LD_PRELOAD=libvma.so
    ;;
  *)
    echo "LIBVMA not enabled"
    ;;
esac

echo "Starting ${BIN_NAME} on NUMA: ${NUMA_NODE}"
echo numactl --physcpubind ${NUMA_CORE_MASK} --preferred=${NUMA_NODE} ${BIN_NAME} 
numactl --physcpubind ${NUMA_CORE_MASK} --preferred=${NUMA_NODE} ${BIN_NAME} 
#numactl --physcpubind ${NUMA_CORE_MASK} --preferred=${NUMA_NODE} gdb ${BIN_NAME} 

#PROCESS_PID=$!
#echo ${PROCESS_PID} > server.pid
#echo "Setting real-time-priority for pid: ${PROCESS_PID}"
#chrt -a -f -p 99 ${PROCESS_PID}
