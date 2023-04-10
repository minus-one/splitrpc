// Copyright (c) Adithya Kumar, The Pennsylvania State University. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include <cmath>
#include <vector>
#include <algorithm>
#include <iostream>

static size_t getTailPercentileIndex(double p, uint64_t numEntries) {
  return static_cast<size_t>(round(p * static_cast<double>(numEntries - 1)));
}

static double getPercentile(std::vector<uint64_t>& metricValues, double p) {
  uint64_t tailPercentileIndex = getTailPercentileIndex(p, metricValues.size());
  std::nth_element(metricValues.begin(), \
      metricValues.begin() + tailPercentileIndex, metricValues.end());
  if(tailPercentileIndex >= metricValues.size())
    return 0.0;
  return metricValues[tailPercentileIndex];
}

static double getMean(std::vector<uint64_t>& metricValues) {
  uint64_t count = 0;
  double mean = 0.0;
  for(auto x: metricValues) {
    double delta = x - mean;
    mean += delta/++count;
  }
  return mean;
}

template<typename T>
class Measure {
  std::string name;
  std::vector<T> measurements;

  public:
  Measure(std::string measurement_name) : name(measurement_name) {}

  ~Measure() {
    measurements.clear();
  }

  inline void add(T m_value) {
    measurements.push_back(m_value);
  }

  inline std::vector<T>& getRawMeasurements() {
    return measurements;
  }

  inline void clear() {
    measurements.clear();
  }

  inline size_t numMeasurements() {
    return measurements.size();
  }

  void dumpStatsToFile(std::string _exp_name) {
    std::ofstream out("raw_"+ name + "_" + _exp_name + ".stat", std::ios::out);
    for(const auto& iter: measurements) {
      out<<iter<<"\n";
    }
  }
};

static void print_stat(std::string key, std::vector<uint64_t> vec)
{
  uint64_t min = 0, max = 0;
  if(vec.size() > 0) {
    min = (*std::min_element(vec.begin(), vec.end()));
    max = (*std::max_element(vec.begin(), vec.end()));
  }
  printf("%s stats(ns) [N, Mean, p90, p95, p99, min, max]: %ld, %0.2f, %0.2f, %0.2f, %0.2f, %ld, %ld\n", \
      key.c_str(), vec.size(), getMean(vec), getPercentile(vec, 0.90),\
      getPercentile(vec , 0.95), getPercentile(vec , 0.99), min, max);
}
