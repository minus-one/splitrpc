// Copyright (c) Adithya Kumar, The Pennsylvania State University. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <sys/time.h>
#include <limits>
#include "spdlog/spdlog.h"
#include <iostream>
#include <sstream>
#include <iomanip>
#include <ctime>

#include "stats_utils.h"

#ifdef PROFILE_MODE

#ifndef GPU_DISABLED
#include <nvToolsExt.h>

#define NVTX_R(x) nvtxRangePush(x);
#define NVTX_P nvtxRangePop();
#define NVTX_M(x) nvtxMark(x);

#endif /* GPU_DISABLED */

static void PROF_PRINT(std::string key, std::vector<uint64_t> vec)
{
  print_stat(key, vec);
}

#else

#define NVTX_R(x)
#define NVTX_P
#define NVTX_M(x) 

#define PROF_PRINT(key, vec)

#endif /* PROFILE_MODE */

static uint64_t getAbsCurNs() {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  uint64_t t = ((tv.tv_sec * 1000 * 1000) + tv.tv_usec) * 1000;
  return t;
}

inline static __attribute__((always_inline)) 
  uint64_t getCurNs() {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  uint64_t t = ts.tv_sec*1000*1000*1000 + ts.tv_nsec;
  return t;
}

inline static __attribute__((always_inline)) 
  void sleepUntil(uint64_t targetNs) {
  uint64_t curNs = getCurNs();
  //if(curNs > targetNs) {
  //  spdlog::critical("SleepUntil is in future: {} ns", (curNs - targetNs));
  //}
  while (curNs < targetNs) {
    uint64_t diffNs = targetNs - curNs;
    struct timespec ts = {(time_t)(diffNs/(1000*1000*1000)), 
      (time_t)(diffNs % (1000*1000*1000))};
    nanosleep(&ts, NULL); //not guaranteed, hence the loop
    curNs = getCurNs();
  }
}

static std::string getCurTsAsString() {
  auto t = std::time(nullptr);
  auto tm = *std::localtime(&t);
  std::ostringstream oss;
  oss << std::put_time(&tm, "%d-%m-%Y %H-%M-%s");
  return oss.str();
}
