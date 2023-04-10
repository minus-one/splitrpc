CMAKE_MINIMUM_REQUIRED(VERSION 3.10)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 70)
endif()

PROJECT('gdr_mm' LANGUAGES CXX C CUDA)

SET(CMAKE_CXX_STANDARD 11)
SET(CMAKE_CXX_STANDARD_REQUIRED ON)
SET(CMAKE_POSITION_INDEPENDENT_CODE ON)
SET(CUDA_SEPARABLE_COMPILATION ON)
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -std=c++11 ${CMAKE_COMMON_FLAGS}")
add_library(gdr_mm gdr_mem_manager.cc)

########################################### CUDA ################################
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

TARGET_INCLUDE_DIRECTORIES(gdr_mm PUBLIC ${CUDA_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR}/nv_helpers)
TARGET_LINK_LIBRARIES(gdr_mm PUBLIC ${CUDA_LIBRARIES} ${CUDA_DRIVER_LIBRARY})
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CMAKE_CUDA_FLAGS}")
SET_TARGET_PROPERTIES(gdr_mm PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
SET_TARGET_PROPERTIES(gdr_mm PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
###################################################################################

####################### GDRCOPY ##############################
SET(GDR_INCLUDE_DIRS "$ENV{GDR_PATH}/include/")
SET(GDR_LIBRARIES "-L$ENV{GDR_PATH}/lib -lgdrapi")

message(STATUS "GDR_INCLUDE_DIRS ${GDR_INCLUDE_DIRS}")
TARGET_INCLUDE_DIRECTORIES(gdr_mm PUBLIC ${GDR_INCLUDE_DIRS})
TARGET_LINK_LIBRARIES(gdr_mm PUBLIC ${GDR_LIBRARIES})
################################################################

option(TRACE_MODE "Enabling tracing of outputs" OFF)
option(PROFILE_MODE "Enables NVTX profiling of individual components" OFF)

if(TRACE_MODE)
  message(STATUS "Enabling trace mode")
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DTRACE_MODE=1")
  SET(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -DTRACE_MODE=1")
endif(TRACE_MODE)

if(PROFILE_MODE)
  message(STATUS "Enabling profile mode")
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DPROFILE_MODE=1")
  SET(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -DPROFILE_MODE=1")
  find_library(CUDA_NVTX_LIBRARY
    NAMES nvToolsExt nvTools nvtoolsext nvtools nvtx NVTX
    PATHS "${CUDA_CUDART_LIBRARY_DIR}" "${CUDA_TOOLKIT_ROOT_DIR}" ENV LD_LIBRARY_PATH
    PATH_SUFFIXES "lib64" "common/lib64" "common/lib" "lib"
    DOC "Location of the CUDA Toolkit Extension (NVTX) library"
    NO_DEFAULT_PATH
    )
  TARGET_LINK_LIBRARIES(gdr_mm PUBLIC ${CUDA_NVTX_LIBRARY})
endif(PROFILE_MODE)

################################################################
