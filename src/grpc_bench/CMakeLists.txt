# Copyright (c) Adithya Kumar, The Pennsylvania State University. All rights reserved.
# Licensed under the MIT License.

PROJECT('grpc_bench' LANGUAGES CXX C CUDA)
CMAKE_MINIMUM_REQUIRED(VERSION 3.10)
SET(CMAKE_CXX_STANDARD 11)
SET(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES $ENV{CUDA_ARCH_CODE_GEN})
endif()
SET(CMAKE_POSITION_INDEPENDENT_CODE ON)
SET(CUDA_SEPARABLE_COMPILATION ON)

ADD_SUBDIRECTORY(../external/spdlog ./external/spdlog)

if(NOT TARGET spdlog)
  # Stand-alone build
  find_package(spdlog REQUIRED)
endif()

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -std=c++11 ${CMAKE_COMMON_FLAGS} -Wall -Wextra -Wno-unused-function")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
INCLUDE_DIRECTORIES(../Common ../utils/ ../ ../Common ./)

SET(THREADS_PREFER_PTHREAD_FLAG ON)
FIND_PACKAGE(Threads REQUIRED)
link_libraries(Threads::Threads spdlog::spdlog)

FIND_PACKAGE(CUDA 10 REQUIRED)
FIND_LIBRARY(CUDA_DRIVER_LIBRARY
             NAMES cuda_driver cuda
             HINTS ${CUDA_TOOLKIT_ROOT_DIR}
                   ENV CUDA_PATH
             PATH_SUFFIXES nvidia/current lib64 lib/x64 lib)
IF (NOT CUDA_DRIVER_LIBRARY)
    FIND_LIBRARY(CUDA_DRIVER_LIBRARY
                 NAMES cuda_driver cuda
                 HINTS ${CUDA_TOOLKIT_ROOT_DIR}
                       ENV CUDA_PATH
                 PATH_SUFFIXES lib64/stubs lib/x64/stubs lib/stubs stubs compat)
ENDIF ()

message(STATUS "CMAKE_CUDA_FLAGS ${CMAKE_CUDA_FLAGS}")
message(STATUS "CUDA_INCLUDE_DIRS ${CUDA_INCLUDE_DIRS}")
message(STATUS "CUDA_LIBRARIES ${CUDA_LIBRARIES}")
message(STATUS "CUDA_DRIVER_LIBRARY ${CUDA_DRIVER_LIBRARY}")


########################## GRPC + PROTOBUF ###################################

find_package(Protobuf REQUIRED)
find_package(gRPC CONFIG REQUIRED)
add_library(infer_proto infer.proto)
target_link_libraries(infer_proto PUBLIC protobuf::libprotobuf gRPC::grpc gRPC::grpc++)
target_include_directories(infer_proto PUBLIC ${CMAKE_CURRENT_BINARY_DIR} ${CUDA_INCLUDE_DIRS})
get_target_property(grpc_cpp_plugin_location gRPC::grpc_cpp_plugin LOCATION)
protobuf_generate(TARGET infer_proto LANGUAGE cpp)
protobuf_generate(TARGET infer_proto LANGUAGE grpc GENERATE_EXTENSIONS .grpc.pb.h .grpc.pb.cc PLUGIN "protoc-gen-grpc=${grpc_cpp_plugin_location}")

####################TARGET DEFINITIONS#########################################

ADD_EXECUTABLE(ort_rpc_bench ort_rpc_bench.cc)
TARGET_LINK_LIBRARIES(ort_rpc_bench PRIVATE infer_proto)

#ADD_EXECUTABLE(grpc_client grpc_client.cc)
#TARGET_LINK_LIBRARIES(grpc_client PRIVATE infer_proto)

ADD_EXECUTABLE(lenet_rpc_bench lenet_rpc_bench.cc ../apps/lenet/lenet_vanilla.cu ../utils/gdr_mem_manager.cc)
TARGET_LINK_LIBRARIES(lenet_rpc_bench PRIVATE infer_proto)
TARGET_INCLUDE_DIRECTORIES(lenet_rpc_bench PUBLIC ../apps/lenet/)
#TARGET_LINK_LIBRARIES(lenet_rpc_bench ${CUDA_NVTX_LIBRARY})
SET_TARGET_PROPERTIES(lenet_rpc_bench PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
SET_TARGET_PROPERTIES(lenet_rpc_bench PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

ADD_EXECUTABLE(vec_add_rpc_bench vec_add_rpc_bench.cc ../apps/vec_add/vector_add.cu ../utils/gdr_mem_manager.cc)
TARGET_LINK_LIBRARIES(vec_add_rpc_bench PRIVATE infer_proto)
TARGET_INCLUDE_DIRECTORIES(vec_add_rpc_bench PUBLIC ../apps/vec_add/)
SET_TARGET_PROPERTIES(vec_add_rpc_bench PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
SET_TARGET_PROPERTIES(vec_add_rpc_bench PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

ADD_EXECUTABLE(mm_rpc_bench mm_rpc_bench.cc ../apps/mm.cu ../utils/gdr_mem_manager.cc)
TARGET_LINK_LIBRARIES(mm_rpc_bench PRIVATE infer_proto)
TARGET_INCLUDE_DIRECTORIES(mm_rpc_bench PUBLIC ../apps/)
SET_TARGET_PROPERTIES(mm_rpc_bench PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
SET_TARGET_PROPERTIES(mm_rpc_bench PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

ADD_EXECUTABLE(lstm_rpc_bench lstm_rpc_bench.cc ../apps/nnfusion_lstm/nnfusion_rt.cu ../utils/gdr_mem_manager.cc)
TARGET_LINK_LIBRARIES(lstm_rpc_bench PRIVATE infer_proto)
TARGET_INCLUDE_DIRECTORIES(lstm_rpc_bench PUBLIC ../apps/nnfusion_lstm/)
SET_TARGET_PROPERTIES(lstm_rpc_bench PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
SET_TARGET_PROPERTIES(lstm_rpc_bench PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
TARGET_LINK_LIBRARIES(lstm_rpc_bench PUBLIC ${CUDA_CUBLAS_LIBRARIES} ${CUDNN_LIBRARY})

SET(NNFUSION_SRC_DIRS "../apps/nnfusion_lstm")
if(EXISTS "${CMAKE_BINARY_DIR}/Constant")
  message(STATUS "LSTM constants files exists")
else()
add_custom_command(
    TARGET lstm_rpc_bench 
    POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_SOURCE_DIR}/${NNFUSION_SRC_DIRS}/Constant ${CMAKE_BINARY_DIR}/Constant
)
endif()

############################ ONNXRUNTIME ######################################

SET(ORT_INCLUDE_DIRS $ENV{ORT_PATH}/include/onnxruntime $ENV{ORT_PATH}/include/onnxruntime/core/session)
#add_definitions(-DUSE_CUDA)
#SET(ORT_LIBRARIES -L$ENV{ORT_PATH}/build/cuda/RelWithDebInfo -lonnxruntime)
add_definitions(-DUSE_TRT)
SET(ORT_LIBRARIES -L$ENV{ORT_PATH}/build/trt/RelWithDebInfo -lonnxruntime)

message(STATUS "ORT_INCLUDE_DIRS" ${ORT_INCLUDE_DIRS})
message(STATUS "ORT_LIBS" ${ORT_LIBRARIES})

TARGET_INCLUDE_DIRECTORIES(ort_rpc_bench PUBLIC ${ORT_INCLUDE_DIRS} "../apps/")
TARGET_LINK_LIBRARIES(ort_rpc_bench PUBLIC ${ORT_LIBRARIES})

############################################ CUDA ################################


SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CMAKE_CUDA_FLAGS}")
TARGET_INCLUDE_DIRECTORIES(ort_rpc_bench PUBLIC ${CUDA_INCLUDE_DIRS})
TARGET_LINK_LIBRARIES(ort_rpc_bench PUBLIC ${CUDA_LIBRARIES} ${CUDA_DRIVER_LIBRARY})

TARGET_INCLUDE_DIRECTORIES(lenet_rpc_bench PUBLIC ${CUDA_INCLUDE_DIRS})
TARGET_LINK_LIBRARIES(lenet_rpc_bench PUBLIC ${CUDA_LIBRARIES} ${CUDA_DRIVER_LIBRARY})

TARGET_INCLUDE_DIRECTORIES(vec_add_rpc_bench PUBLIC ${CUDA_INCLUDE_DIRS})
TARGET_LINK_LIBRARIES(vec_add_rpc_bench PUBLIC ${CUDA_LIBRARIES} ${CUDA_DRIVER_LIBRARY})

TARGET_INCLUDE_DIRECTORIES(mm_rpc_bench PUBLIC ${CUDA_INCLUDE_DIRS})
TARGET_LINK_LIBRARIES(mm_rpc_bench PUBLIC ${CUDA_LIBRARIES} ${CUDA_DRIVER_LIBRARY})

TARGET_INCLUDE_DIRECTORIES(lstm_rpc_bench PUBLIC ${CUDA_INCLUDE_DIRS})
TARGET_LINK_LIBRARIES(lstm_rpc_bench PUBLIC ${CUDA_LIBRARIES} ${CUDA_DRIVER_LIBRARY})

###################################################################################
####################### GDRCOPY ##############################
SET(GDR_INCLUDE_DIRS "$ENV{GDR_PATH}/include/")
SET(GDR_LIBRARIES "-L$ENV{GDR_PATH}/lib -lgdrapi")

message(STATUS "GDR_INCLUDE_DIRS ${GDR_INCLUDE_DIRS}")
TARGET_INCLUDE_DIRECTORIES(lenet_rpc_bench PUBLIC ${GDR_INCLUDE_DIRS})
TARGET_LINK_LIBRARIES(lenet_rpc_bench PUBLIC ${GDR_LIBRARIES})
TARGET_INCLUDE_DIRECTORIES(vec_add_rpc_bench PUBLIC ${GDR_INCLUDE_DIRS})
TARGET_LINK_LIBRARIES(vec_add_rpc_bench PUBLIC ${GDR_LIBRARIES})
TARGET_INCLUDE_DIRECTORIES(mm_rpc_bench PUBLIC ${GDR_INCLUDE_DIRS})
TARGET_LINK_LIBRARIES(mm_rpc_bench PUBLIC ${GDR_LIBRARIES})
TARGET_INCLUDE_DIRECTORIES(lstm_rpc_bench PUBLIC ${GDR_INCLUDE_DIRS})
TARGET_LINK_LIBRARIES(lstm_rpc_bench PUBLIC ${GDR_LIBRARIES})

################################################################

option(TRACE_MODE "Enabling tracing of outputs" OFF)
if(TRACE_MODE)
  message(STATUS "Enabling trace mode")
  add_definitions(-DTRACE_MODE=1)
endif(TRACE_MODE)

option(PROFILE_MODE "Enables profiling of individual components" OFF)
if(PROFILE_MODE)
  message(STATUS "Enabling profile mode")
  add_definitions(-DPROFILE_MODE=1)
  target_link_libraries(ort_rpc_bench PUBLIC ${CUDA_NVTX_LIBRARY})
  target_link_libraries(lenet_rpc_bench PUBLIC ${CUDA_NVTX_LIBRARY})
  target_link_libraries(lstm_rpc_bench PUBLIC ${CUDA_NVTX_LIBRARY})
  target_link_libraries(vec_add_rpc_bench PUBLIC ${CUDA_NVTX_LIBRARY})
  target_link_libraries(mm_rpc_bench PUBLIC ${CUDA_NVTX_LIBRARY})
endif(PROFILE_MODE)

################################################################################
