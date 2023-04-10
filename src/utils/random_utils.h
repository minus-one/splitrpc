// Copyright (c) Adithya Kumar, The Pennsylvania State University. All rights reserved.
// Licensed under the MIT License.

#ifndef RANDOM_UTILS
#define RANDOM_UTILS

#include <random>
#include <limits>

static std::mt19937_64 globalGenerator;
// Initialize random generator with seed
static void seedGenerator(unsigned int seed)
{
  globalGenerator.seed(seed);
}
// Initialize random generator with random seed
static void seedGeneratorRand()
{
  std::random_device rd;
  globalGenerator.seed(rd());
}
// Generate a uniform random number in the range [0, 1)
static double uniform01()
{
  return std::generate_canonical<double, std::numeric_limits<double>::digits>(globalGenerator);
}

static int uniformint(int min, int max)
{
  std::uniform_int_distribution<int> dist(min, max);
  return dist(globalGenerator);
}


#endif
