#! /bin/zsh

function SETAPPCONFIGS {
  APP_CONFIG_NAME=$1
    case ${APP_CONFIG_NAME} in
    echo_app)
    export P2P_RPC_MTU=1024
    export P2P_RPC_REQ_SIZE=1024
    export P2P_RPC_RESP_SIZE=1024
    ;;
  resnet18)
    export P2P_RPC_MTU=8192
    export P2P_RPC_REQ_SIZE=602112
    export P2P_RPC_RESP_SIZE=4000
    export P2P_RPC_ORT_MODEL_NAME=resnet18v2batchedtrt.onnx
    ;;
  resnet50)
    export P2P_RPC_MTU=8192
    export P2P_RPC_REQ_SIZE=602112
    export P2P_RPC_RESP_SIZE=4000
    export P2P_RPC_ORT_MODEL_NAME=resnet50v2batchedtrt.onnx
    ;;
  bert)
    export P2P_RPC_MTU=8192
    export P2P_RPC_REQ_SIZE=6152
    export P2P_RPC_RESP_SIZE=2056
    export P2P_RPC_ORT_MODEL_NAME=bertsquad10-symbolic.onnx
    ;;
  vecadd1k)
    export P2P_RPC_MTU=1024
    export P2P_RPC_REQ_SIZE=1024
    export P2P_RPC_RESP_SIZE=1024
    ;;
  vecadd1M)
    export P2P_RPC_MTU=8192
    export P2P_RPC_REQ_SIZE=$((1024*1024))
    export P2P_RPC_RESP_SIZE=$((1024*1024))
    ;;
  lenet)
    export P2P_RPC_MTU=784
    export P2P_RPC_REQ_SIZE=784
    export P2P_RPC_RESP_SIZE=40
    ;;
  lstm)
    export P2P_RPC_MTU=8192
    export P2P_RPC_REQ_SIZE=8192
    export P2P_RPC_RESP_SIZE=1024
    ;;
  matmul16)
    export P2P_RPC_MTU=2048
    export P2P_RPC_REQ_SIZE=2048
    export P2P_RPC_RESP_SIZE=1024
    ;;
  matmul32)
    export P2P_RPC_MTU=8192
    export P2P_RPC_REQ_SIZE=8192
    export P2P_RPC_RESP_SIZE=4096
    ;;
  matmul)
    export P2P_RPC_MTU=8192
    export P2P_RPC_REQ_SIZE=32768
    export P2P_RPC_RESP_SIZE=16384
    ;;
  *)
    export P2P_RPC_MTU=8192
    export P2P_RPC_REQ_SIZE=8192
    export P2P_RPC_RESP_SIZE=8192
    ;;
  esac
}
