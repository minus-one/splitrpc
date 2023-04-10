// Copyright (c) Adithya Kumar, The Pennsylvania State University. All rights reserved.
// Licensed under the MIT License.

#ifndef COMMON_DEFS_H
#define COMMON_DEFS_H

#include <errno.h>
#include <string.h>
#include <iostream>
#include <sstream>
#include <memory>
#include <string>
#include <unistd.h>
#include <math.h>
#include <atomic>
#include <cstdlib>
#include <random>
#include <string>
#include <thread>
#include <random>
#include <fstream>
#include <algorithm>
#include <queue>
#include <sys/time.h>
#include <random>
#include <climits>

#include "spdlog/spdlog.h"
#include "utils/config_utils.h"
#include "utils/random_utils.h"
#include "utils/time_utils.h"

static const uint64_t MAX_PAYLOAD_SIZE = 128 * 1024 * 1024; // 128 MiB

static const uint16_t DEF_ARR_RATE_COV = 0; // Fixed arrival rate

#endif /* COMMON_DEFS_H */
