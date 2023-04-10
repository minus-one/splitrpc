# Copyright (c) Adithya Kumar, The Pennsylvania State University. All rights reserved.
# Licensed under the MIT License.

PROJECT('lib_loadgen' LANGUAGES CXX C)
CMAKE_MINIMUM_REQUIRED(VERSION 3.10)
SET(CMAKE_CXX_STANDARD 11)
SET(CMAKE_CXX_STANDARD_REQUIRED ON)

ADD_COMPILE_DEFINITIONS("ALLOW_EXPERIMENTAL_API")

ADD_SUBDIRECTORY(../external/spdlog ./external/spdlog)

if(NOT TARGET spdlog)
  # Stand-alone build
  find_package(spdlog REQUIRED)
endif()

LIST(APPEND RPC_APPS
  echo_rpc_bench
  vec_add_rpc_bench
  ort_rpc_bench)

SET(ALL_SRCS
    ../transport/dpdk_init.cc
    dist_rpc_bench.cc)

# The client does not need GPU support. So disable it
add_definitions(-DGPU_DISABLED=1)

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -std=c++17 ${CMAKE_COMMON_FLAGS}")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")

INCLUDE_DIRECTORIES(../Common ../utils/ ../ ../transport)
SET(THREADS_PREFER_PTHREAD_FLAG ON)
FIND_PACKAGE(Threads REQUIRED)
link_libraries(Threads::Threads spdlog::spdlog)

########################## GRPC + PROTOBUF ###################################

find_package(Protobuf REQUIRED)
find_package(gRPC CONFIG REQUIRED)
add_library(infer_proto infer.proto)
target_link_libraries(infer_proto PUBLIC protobuf::libprotobuf gRPC::grpc gRPC::grpc++)
target_include_directories(infer_proto PUBLIC ${CMAKE_CURRENT_BINARY_DIR})
get_target_property(grpc_cpp_plugin_location gRPC::grpc_cpp_plugin LOCATION)
protobuf_generate(TARGET infer_proto LANGUAGE cpp)
protobuf_generate(TARGET infer_proto LANGUAGE grpc GENERATE_EXTENSIONS .grpc.pb.h .grpc.pb.cc PLUGIN "protoc-gen-grpc=${grpc_cpp_plugin_location}")

###############################################################################

option(TRACE_MODE "Enabling tracing of outputs" OFF)
option(PROFILE_MODE "Enables NVTX profiling of individual components" OFF)

if(TRACE_MODE)
  message(STATUS "Enabling trace mode")
  add_definitions(-DTRACE_MODE=1)
endif(TRACE_MODE)

if(PROFILE_MODE)
  message(STATUS "Enabling profile mode")
  add_definitions(-DPROFILE_MODE=1)
  #FIND_PACKAGE(CUDA 10 REQUIRED)
  #INCLUDE_DIRECTORIES(${CUDA_INCLUDE_DIRS})
  #find_library(CUDA_NVTX_LIBRARY
  #  NAMES nvToolsExt nvTools nvtoolsext nvtools nvtx NVTX
  #  PATHS "${CUDA_CUDART_LIBRARY_DIR}" "${CUDA_TOOLKIT_ROOT_DIR}" ENV LD_LIBRARY_PATH
  #  PATH_SUFFIXES "lib64" "common/lib64" "common/lib" "lib"
  #  DOC "Location of the CUDA Toolkit Extension (NVTX) library"
  #  NO_DEFAULT_PATH
  #  )
  #link_libraries(${CUDA_NVTX_LIBRARY})
endif(PROFILE_MODE)
###############################################################################

####################### DPDK ##############################
FIND_PACKAGE(PkgConfig REQUIRED)
PKG_CHECK_MODULES(DPDK REQUIRED libdpdk)
string(REPLACE ";" " " DPDK_CFLAGS_OTHER "${DPDK_CFLAGS_OTHER}")
SET(DPDK_LDFLAGS "${DPDK_LDFLAGS} -Wl,--no-whole-archive -lmlx5 -libverbs -pthread -lnuma -ldl")

message(STATUS "DPDK_INCLUDE_DIRS ${DPDK_INCLUDE_DIRS}")
message(STATUS "DPDK_CFLAGS ${DPDK_CFLAGS_OTHER}")
message(STATUS "DPDK_LDFLAGS ${DPDK_LDFLAGS}")

INCLUDE_DIRECTORIES(PUBLIC ${DPDK_INCLUDE_DIRS})
LINK_LIBRARIES(${DPDK_LDFLAGS})
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${DPDK_CFLAGS_OTHER}")
###########################################################

############################################ CUDA ################################
#FIND_PACKAGE(CUDA 10 REQUIRED)
#FIND_LIBRARY(CUDA_DRIVER_LIBRARY
#             NAMES cuda_driver cuda
#             HINTS ${CUDA_TOOLKIT_ROOT_DIR}
#                   ENV CUDA_PATH
#             PATH_SUFFIXES nvidia/current lib64 lib/x64 lib)
#IF (NOT CUDA_DRIVER_LIBRARY)
#    FIND_LIBRARY(CUDA_DRIVER_LIBRARY
#                 NAMES cuda_driver cuda
#                 HINTS ${CUDA_TOOLKIT_ROOT_DIR}
#                       ENV CUDA_PATH
#                 PATH_SUFFIXES lib64/stubs lib/x64/stubs lib/stubs stubs compat)
#ENDIF ()
#
##message(STATUS "CMAKE_CUDA_FLAGS ${CMAKE_CUDA_FLAGS}")
##message(STATUS "CUDA_INCLUDE_DIRS ${CUDA_INCLUDE_DIRS}")
##message(STATUS "CUDA_LIBRARIES ${CUDA_LIBRARIES}")
##message(STATUS "CUDA_DRIVER_LIBRARY ${CUDA_DRIVER_LIBRARY}")
#
#INCLUDE_DIRECTORIES( PUBLIC ${CUDA_INCLUDE_DIRS})
#LINK_LIBRARIES(${CUDA_LIBRARIES} ${CUDA_DRIVER_LIBRARY})
#SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CMAKE_CUDA_FLAGS}")
####################################################################################

foreach(RPC_APP IN LISTS RPC_APPS)
  message(STATUS "Building APP: ${RPC_APP}")
  ADD_EXECUTABLE(${RPC_APP} ${ALL_SRCS})
  TARGET_LINK_LIBRARIES(${RPC_APP} PRIVATE infer_proto)
endforeach()

TARGET_SOURCES(echo_rpc_bench PUBLIC echo_client.cc)
TARGET_SOURCES(vec_add_rpc_bench PUBLIC vec_add_client.cc)
TARGET_SOURCES(ort_rpc_bench PUBLIC ort_client.cc)
