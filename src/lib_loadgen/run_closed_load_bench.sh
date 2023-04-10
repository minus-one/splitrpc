#! /bin/zsh

APP_NAME=$1
BIN_NAME=../../build/${APP_NAME}_rpc_bench

source env_config.rc
source dev_config.rc

export CUDA_PATH=/usr/local/cuda-11.4
export GDR_PATH=/export/home/azk68/code-base/gdrcopy
export DPDK_PATH=/export/home/azk68/code-base/dpdk_install

export PKG_CONFIG_PATH=${DPDK_PATH}/lib/x86_64-linux-gnu/pkgconfig
export LD_LIBRARY_PATH=${CUDA_PATH}/lib64/:${DPDK_PATH}/lib/x86_64-linux-gnu/:${GDR_PATH}/lib

export CUDA_ARCH_CODE_GEN=70
#export MLX5_SHUT_UP_BF=0
export NSYS_PATH=/opt/nvidia/nsight-systems/2021.4.1/bin

#export P2P_RPC_URI=192.168.25.1:50051
export P2P_RPC_URI=0c:42:a1:10:41:18,192.168.25.2,50052,b8:ce:f6:cc:6a:52,192.168.25.1,50051
export P2P_RPC_NUM_WARMUP=500
export P2P_RPC_REQ_GEN_TYPE=0
export P2P_RPC_N_GEN_THR=1
export P2P_RPC_N_LISTENER_THR=1
export P2P_RPC_DUMP_RAW_STATS=1
export P2P_RPC_CLIENT_TYPE=1
export P2P_RPC_DPDK_PORT=0

EXPT_PREFIX=s_1_${APP_NAME}_${P2P_RPC_REQ_GEN_TYPE}
#LIST_SIZES=(64 128 256 512 1024 2048 4096 8192)
#LIST_SIZES=(64 128 256 512 1024)
LIST_SIZES=(1024)

echo "PKT_SIZE, TP, Mean, P50, P90, P95, P99" 
for PKT_SIZE in "${LIST_SIZES[@]}"
do
	((PAYLOAD_SIZE=PKT_SIZE-8))
	echo "Bechmarking PKT_SIZE: "${PKT_SIZE}
	export P2P_RPC_FIXED_REQ_SIZE=${PAYLOAD_SIZE}
  export P2P_RPC_BENCH_TOTAL_REQUESTS=1000
  export P2P_RPC_BENCH_ARR_RATE=1
  export P2P_RPC_BENCH_EXP_NAME=${EXPT_PREFIX}_${PKT_SIZE}
	#numactl -N 1 -l gdb --args ${BIN_NAME} 
  #valgrind --leak-check=full --show-leak-kinds=all ${BIN_NAME}
  numactl -N 1 -l ${BIN_NAME}
  #numactl -N 1 -l\
  #  ${NSYS_PATH}/nsys profile -o\
  #  ${P2P_RPC_BENCH_EXP_NAME}.qdrep\
  #  --force-overwrite=true\
  #  ${BIN_NAME}
  tp=`grep "achieved_throughput" agg_${EXPT_PREFIX}_${PKT_SIZE}*.json | cut -d':' -f 2`
	mean_soj=`grep "mean_soj" agg_${EXPT_PREFIX}_${PKT_SIZE}*.json | cut -d':' -f 2`
	p50_soj=`grep "p50_soj" agg_${EXPT_PREFIX}_${PKT_SIZE}*.json | cut -d':' -f 2`
	p90_soj=`grep "p90_soj" agg_${EXPT_PREFIX}_${PKT_SIZE}*.json | cut -d':' -f 2`
	p95_soj=`grep "p95_soj" agg_${EXPT_PREFIX}_${PKT_SIZE}*.json | cut -d':' -f 2`
	p99_soj=`grep "p99_soj" agg_${EXPT_PREFIX}_${PKT_SIZE}*.json | cut -d':' -f 2`
	echo ${PKT_SIZE}, ${tp} ${mean_soj} ${p50_soj} ${p90_soj} ${p95_soj} ${p99_soj}
  #${NSYS_PATH}/nsys stats ${P2P_RPC_BENCH_EXP_NAME}.qdrep\
  #  --force-overwrite=true\
  #  -r nvtxsum -f csv\
  #  -o nvtx_${P2P_RPC_BENCH_EXP_NAME}
done
