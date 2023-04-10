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
export P2P_RPC_APP_PATH=${P2P_RPC_BASE_PATH}/lib_loadgen/build

APP_CONFIG_NAME=$1
EXPT_SUFFIX=$2
echo "APP_CONFIG: ${APP_CONFIG_NAME} EXPT_SUFFIX: ${EXPT_CONFIG}"
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
  vecadd1k)
    BIN_NAME=${P2P_RPC_APP_PATH}/vec_add_rpc_bench
    ;;
  vecadd1M)
    BIN_NAME=${P2P_RPC_APP_PATH}/vec_add_rpc_bench
    ;;
  lenet)
    BIN_NAME=${P2P_RPC_APP_PATH}/ort_rpc_bench
    ;;
  lenet_sockperf)
    BIN_NAME=${P2P_RPC_APP_PATH}/ort_rpc_bench
    ;;
  lstm)
    BIN_NAME=${P2P_RPC_APP_PATH}/ort_rpc_bench
    ;;
  *)
    BIN_NAME=${P2P_RPC_APP_PATH}/ort_rpc_bench
    ;;
esac

#### Format <source_mac_str>,<source_ip_str>,<source_port_str>,<dest_mac_str>,<dest_ip_str>,<dest_port_str>
export P2P_RPC_URI=0c:42:a1:10:41:18,192.168.25.2,50052,b8:ce:f6:cc:6a:52,192.168.25.1,50051
#export P2P_RPC_URI=b8:ce:f6:cc:6a:52,192.168.25.1,50052,0c:42:a1:10:41:18,192.168.25.2,50051

######## LoadGen specific
## 0 = UDP, 1 = GRPC, 2 = P2P-RPC
export P2P_RPC_CLIENT_TYPE=2
## 0 = closed loop, 1 = open loop
export P2P_RPC_REQ_GEN_TYPE=0
## Number of threads
export P2P_RPC_N_GEN_THR=1
export P2P_RPC_N_LISTENER_THR=1
## 1 = Dump the raw stats
export P2P_RPC_DUMP_RAW_STATS=0

####### LoadGen Run specific
export P2P_RPC_BENCH_ARR_RATE=1
export P2P_RPC_BENCH_TOTAL_REQUESTS=100
export P2P_RPC_NUM_WARMUP=10
## 0 = Skip verification (Default), 1 = Verify each reply
export P2P_RPC_CLIENT_VERIFY_RUN=1

export P2P_RPC_BENCH_EXP_NAME=t_${APP_CONFIG_NAME}_${P2P_RPC_REQ_GEN_TYPE}_${P2P_RPC_BENCH_ARR_RATE}_${EXPT_SUFFIX}

#numactl -N 1 -l gdb --args ${BIN_NAME} 
#valgrind --leak-check=full --show-leak-kinds=all ${BIN_NAME}

# Actual command
numactl -a --physcpubind ${NUMA_CORE_MASK} -m ${NUMA_NODE} ${BIN_NAME} -l ${DPDK_LCORES}

#export NSYS_PATH=/opt/nvidia/nsight-systems/2021.4.1/bin
#numactl -N 1 -l\
#  ${NSYS_PATH}/nsys profile -o\
#  ${P2P_RPC_BENCH_EXP_NAME}.qdrep\
#  --force-overwrite=true\
#  ${BIN_NAME}
#${NSYS_PATH}/nsys stats ${P2P_RPC_BENCH_EXP_NAME}.qdrep\
#  --force-overwrite=true\
#  -r nvtxsum -f csv\
#  -o nvtx_${P2P_RPC_BENCH_EXP_NAME}

cat agg_${P2P_RPC_BENCH_EXP_NAME}_*
