# Copyright (c) Adithya Kumar, The Pennsylvania State University. All rights reserved.
# Licensed under the MIT License.

PROJECT('rdma_bench' LANGUAGES CXX C CUDA)
CMAKE_MINIMUM_REQUIRED(VERSION 3.10)
SET(CMAKE_CXX_STANDARD 11)
SET(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES $ENV{CUDA_ARCH_CODE_GEN})
endif()
SET(CMAKE_POSITION_INDEPENDENT_CODE ON)
SET(CUDA_SEPARABLE_COMPILATION ON)

ADD_COMPILE_DEFINITIONS("ALLOW_EXPERIMENTAL_API")

SET(GDR_INCLUDE_DIRS "$ENV{GDR_PATH}/include/")

ADD_SUBDIRECTORY(../external/spdlog ${CMAKE_BINARY_DIR}/spdlog)
ADD_SUBDIRECTORY(../utils/ ${CMAKE_BINARY_DIR}/gdr_mm)
INCLUDE_DIRECTORIES(../Common ../utils/ ../ ../Common ${CMAKE_SOURCE_DIR} ${GDR_INCLUDE_DIRS})

if(NOT TARGET spdlog)
  # Stand-alone build
  find_package(spdlog REQUIRED)
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Ofast -ffast-math -funroll-loops -march=native -std=c++11 ${CMAKE_COMMON_FLAGS} -Wall -Wextra -Wno-unused-function")

SET(THREADS_PREFER_PTHREAD_FLAG ON)
FIND_PACKAGE(Threads REQUIRED)
link_libraries(Threads::Threads spdlog::spdlog)

####################### DPDK #################################

FIND_PACKAGE(PkgConfig REQUIRED)
PKG_CHECK_MODULES(DPDK REQUIRED libdpdk)
string(REPLACE ";" " " DPDK_CFLAGS_OTHER "${DPDK_CFLAGS_OTHER}")
SET(DPDK_LDFLAGS "${DPDK_LDFLAGS} -Wl,--no-whole-archive -lmlx5 -libverbs -pthread -lnuma -ldl")
message(STATUS "DPDK_INCLUDE_DIRS ${DPDK_INCLUDE_DIRS}")
message(STATUS "DPDK_CFLAGS ${DPDK_CFLAGS_OTHER}")
message(STATUS "DPDK_LDFLAGS ${DPDK_LDFLAGS}")
#INCLUDE_DIRECTORIES(PUBLIC ${DPDK_INCLUDE_DIRS})
#LINK_LIBRARIES(${DPDK_LDFLAGS})
#SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${DPDK_CFLAGS_OTHER}")

########################## CUDA #################################

FIND_PACKAGE(CUDA 10 REQUIRED)
FIND_LIBRARY(CUDA_DRIVER_LIBRARY
             NAMES cuda_driver cuda
             HINTS ${CUDA_TOOLKIT_ROOT_DIR}
                   ENV CUDA_PATH
             PATH_SUFFIXES nvidia/current lib64 lib/x64 lib lib64/stubs lib/x64/stubs lib/stubs stubs compat)
message(STATUS "CUDA_INCLUDE_DIRS ${CUDA_INCLUDE_DIRS}")

############################ RDMA ################################

FIND_LIBRARY(IBVERBS_LIBRARY ibverbs)
FIND_LIBRARY(RDMACM_LIBRARY rdmacm)

###################################################################

LIST(APPEND RPC_APPS
  echo_rpc_bench
  vec_add_rpc_bench
  ort_rpc_bench
  lenet_rpc_bench
  lstm_rpc_bench
  mm_rpc_bench
  )

foreach(RPC_APP IN LISTS RPC_APPS)
  message(STATUS "Building APP: ${RPC_APP}")
  ADD_EXECUTABLE(${RPC_APP} ../apps/${RPC_APP}.cc gpu_rdma_common.cc gpu_rdma_batched_server.cc ../utils/g_utils.cu)
  TARGET_LINK_LIBRARIES(${RPC_APP} PRIVATE gdr_mm)
  SET_TARGET_PROPERTIES(${RPC_APP} PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
  SET_TARGET_PROPERTIES(${RPC_APP} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
  TARGET_INCLUDE_DIRECTORIES(${RPC_APP} PUBLIC ${CUDA_INCLUDE_DIRS})
  TARGET_LINK_LIBRARIES(${RPC_APP} PUBLIC ${CUDA_LIBRARIES} ${CUDA_DRIVER_LIBRARY} ${IBVERBS_LIBRARY} ${RDMACM_LIBRARY})
endforeach()

TARGET_SOURCES(vec_add_rpc_bench PUBLIC ../apps/vec_add/vector_add.cu)
TARGET_INCLUDE_DIRECTORIES(vec_add_rpc_bench PUBLIC ../apps/vec_add)

TARGET_SOURCES(lenet_rpc_bench PUBLIC ../apps/lenet/lenet_vanilla.cu)
TARGET_INCLUDE_DIRECTORIES(lenet_rpc_bench PUBLIC ../apps/lenet/)

TARGET_SOURCES(mm_rpc_bench PUBLIC ../apps/mm.cu)
TARGET_INCLUDE_DIRECTORIES(mm_rpc_bench PUBLIC ../apps/)

############################ ONNXRUNTIME ######################################

SET(ORT_INCLUDE_DIRS $ENV{ORT_PATH}/include/onnxruntime $ENV{ORT_PATH}/include/onnxruntime/core/session)
#add_definitions(-DUSE_CUDA)
#SET(ORT_LIBRARIES -L$ENV{ORT_PATH}/build/cuda/RelWithDebInfo -lonnxruntime)
add_definitions(-DUSE_TRT)
SET(ORT_LIBRARIES -L$ENV{ORT_PATH}/build/trt/RelWithDebInfo -lonnxruntime)

message(STATUS "ORT_INCLUDE_DIRS" ${ORT_INCLUDE_DIRS})
message(STATUS "ORT_LIBS" ${ORT_LIBRARIES})

TARGET_INCLUDE_DIRECTORIES(ort_rpc_bench PUBLIC ../apps/ ${ORT_INCLUDE_DIRS})
TARGET_LINK_LIBRARIES(ort_rpc_bench PUBLIC ${ORT_LIBRARIES})

############################ NNFUSION ######################################

SET(NNFUSION_SRC_DIRS "../apps/nnfusion_lstm")
message(STATUS "NNFUSION_SRC_DIRS: ${NNFUSION_SRC_DIRS}")
TARGET_INCLUDE_DIRECTORIES(lstm_rpc_bench PUBLIC ${NNFUSION_SRC_DIRS})
TARGET_SOURCES(lstm_rpc_bench PUBLIC ${NNFUSION_SRC_DIRS}/nnfusion_rt.cu)
TARGET_LINK_LIBRARIES(lstm_rpc_bench PUBLIC ${CUDA_CUBLAS_LIBRARIES} ${CUDNN_LIBRARY})

if(EXISTS "${CMAKE_BINARY_DIR}/Constant")
  message(STATUS "LSTM constants files exists")
else()
add_custom_command(
    TARGET lstm_rpc_bench 
    POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_SOURCE_DIR}/${NNFUSION_SRC_DIRS}/Constant ${CMAKE_BINARY_DIR}/Constant
)
endif()

################################## PROXY HANDLER ###############################

ADD_EXECUTABLE(gpu_rdma_proxy_handler gpu_rdma_common.cc gpu_rdma_proxy_handler.cc)
TARGET_COMPILE_DEFINITIONS(gpu_rdma_proxy_handler PUBLIC -DGPU_DISABLED=1)
TARGET_LINK_LIBRARIES(gpu_rdma_proxy_handler ${IBVERBS_LIBRARY} ${RDMACM_LIBRARY})

ADD_EXECUTABLE(gpu_dpdk_rdma_proxy_handler gpu_rdma_common.cc gpu_dpdk_rdma_proxy_handler.cc ../transport/dpdk_init.cc)
TARGET_COMPILE_DEFINITIONS(gpu_dpdk_rdma_proxy_handler PUBLIC -DGPU_DISABLED=1)
TARGET_INCLUDE_DIRECTORIES(gpu_dpdk_rdma_proxy_handler PUBLIC ${DPDK_INCLUDE_DIRS})
TARGET_LINK_LIBRARIES(gpu_dpdk_rdma_proxy_handler PUBLIC ${IBVERBS_LIBRARY} ${RDMACM_LIBRARY} ${DPDK_LDFLAGS})
SET_TARGET_PROPERTIES(gpu_dpdk_rdma_proxy_handler PROPERTIES COMPILE_FLAGS "${CMAKE_CXX_FLAGS} ${DPDK_CFLAGS_OTHER}")

###################################################################

option(TRACE_MODE "Enabling tracing of outputs" OFF)
if(TRACE_MODE)
  message(STATUS "Enabling trace mode")
  add_definitions(-DTRACE_MODE=1)
endif(TRACE_MODE)

option(PROFILE_MODE "Enables profiling of individual components" OFF)
if(PROFILE_MODE)
  message(STATUS "Enabling profile mode")
  add_definitions(-DPROFILE_MODE=1)
foreach(RPC_APP IN LISTS RPC_APPS)
  TARGET_LINK_LIBRARIES(${RPC_APP} PUBLIC ${CUDA_NVTX_LIBRARY})
endforeach()
endif(PROFILE_MODE)

################################################################################
