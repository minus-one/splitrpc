#! /bin/zsh

APP_NAME=$1
BIN_NAME=../build/${APP_NAME}_rpc_bench

export P2P_RPC_URI=192.168.25.1:50051
export P2P_RPC_NUM_WARMUP=1000
export P2P_RPC_N_GEN_THR=1
export P2P_RPC_N_LISTENER_THR=1

# LoadGen Type
export P2P_RPC_REQ_GEN_TYPE=1
export P2P_RPC_BENCH_TOTAL_REQUESTS=50000
export P2P_RPC_BENCH_ARR_RATE=1000
 
EXPT_PREFIX=${APP_NAME}_${P2P_RPC_REQ_GEN_TYPE}
#LIST_SIZES=(64 128 256 512 1024 2048 4096 8192)
LIST_SIZES=(64)
#LIST_SIZES=(1024 2048 4096)

echo "PKT_SIZE, TP, Mean, P50, P90, P95, P99" 
for PKT_SIZE in "${LIST_SIZES[@]}"
do
  ((PAYLOAD_SIZE=PKT_SIZE-8))
  echo "Bechmarking PKT_SIZE: "${PKT_SIZE}
  export P2P_RPC_FIXED_REQ_SIZE=${PAYLOAD_SIZE}
  export P2P_RPC_BENCH_EXP_NAME=${EXPT_PREFIX}_${PKT_SIZE}
  #gdb --args ${BIN_NAME} 
  #valgrind --leak-check=full --show-leak-kinds=all ${BIN_NAME}
  ${BIN_NAME}
  tp=`grep "achieved_throughput" agg_${EXPT_PREFIX}_${PKT_SIZE}*.json | cut -d':' -f 2`
  mean_soj=`grep "mean_soj" agg_${EXPT_PREFIX}_${PKT_SIZE}*.json | cut -d':' -f 2`
  p50_soj=`grep "p50_soj" agg_${EXPT_PREFIX}_${PKT_SIZE}*.json | cut -d':' -f 2`
  p90_soj=`grep "p90_soj" agg_${EXPT_PREFIX}_${PKT_SIZE}*.json | cut -d':' -f 2`
  p95_soj=`grep "p95_soj" agg_${EXPT_PREFIX}_${PKT_SIZE}*.json | cut -d':' -f 2`
  p99_soj=`grep "p99_soj" agg_${EXPT_PREFIX}_${PKT_SIZE}*.json | cut -d':' -f 2`
  echo ${PKT_SIZE}, ${tp} ${mean_soj} ${p50_soj} ${p90_soj} ${p95_soj} ${p99_soj}
done
