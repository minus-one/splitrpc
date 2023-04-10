// Copyright (c) Adithya Kumar, The Pennsylvania State University. All rights reserved.
// Licensed under the MIT License.

/*
 * Header-only Statistics collection engine
 * Collects stats, and centrally stores them. 
 */

#pragma once

#include "common_defs.h"
#include <unordered_map>
#include <iomanip>

#include "utils/json.hpp"
#include "utils/stats_utils.h"

using json = nlohmann::json;

const std::string LAT_UNIT = "us";
const uint32_t LAT_UNIT_MULTIPLIER = 1e3;

//const std::string LAT_UNIT = "ms";
//const uint32_t LAT_UNIT_MULTIPLIER = 1e6;

//const std::string LAT_UNIT = "ns";
//const uint32_t LAT_UNIT_MULTIPLIER = 1;

class StatsManager {
  std::string _exp_name;
  
  // TODO: Change this to include metrics of different types
  // Use a tuple <metric-name, type> to map it to the metric
  std::unordered_map<std::string, Measure<uint64_t>* > metrics;

  // Aggregate statistics that need to be dumped into the json file
  json aggr_stats;

  void dumpRawStats();

  public:
  uint64_t expStartTimeNs, expEndTimeNs;
  
  StatsManager(std::string exp_name) {
    _exp_name = exp_name;
    expStartTimeNs = getCurNs();
  }

  ~StatsManager() {
    for (const auto &iter : metrics) {
      Measure<uint64_t>* x = iter.second;
      delete x;
    }
    metrics.clear();
  }

  void startExp(uint64_t setStart = 0) {
    if(setStart == 0)
      expStartTimeNs = getCurNs();
    else
      expStartTimeNs = setStart;
    resetStats();
  }

  void endExp() {
    expEndTimeNs = getCurNs();
    trackStatInfo("exec_time_ns", expEndTimeNs - expStartTimeNs); 
  }

  void addMeasurementType(std::string measure_name) {
    if(metrics.find(measure_name) != metrics.end()) {
      spdlog::critical("Stats Error! Duplicate measurement name, already exists");
      return ;
    }
    Measure<uint64_t>* newMeasure = new Measure<uint64_t>(measure_name);
    metrics.insert(std::make_pair(measure_name, newMeasure));
  }

  std::vector<uint64_t>& getRawMeasure(std::string measure_name) {
    if(metrics.find(measure_name) != metrics.end()) {
      return metrics[measure_name]->getRawMeasurements();
    }
    spdlog::critical("Stats Error!, missing measurement name!");
    return metrics.begin()->second->getRawMeasurements();
  }

  void recordEvent(std::string measure_name, uint64_t metric_value) {
    if (metrics.find(measure_name) == metrics.end()) {
      spdlog::critical("Stats Warning!, could not find measure {}, adding...", measure_name);
      Measure<uint64_t>* newMeasure = new Measure<uint64_t>(measure_name);
      metrics.insert(std::make_pair(measure_name, newMeasure));
    }
    metrics[measure_name]->add(metric_value);
  }

  void trackStatInfo(std::string key_name, uint64_t value) {
    aggr_stats[key_name] = value;
  }

  void trackStatInfo(std::string key_name, std::string value) {
    aggr_stats[key_name] = value;
  }

  void trackStatInfo(std::string key_name, double value) {
    aggr_stats[key_name] = value;
  }

  void saveStats(std::vector<std::string> agg_measures = {"sojournTime"}, bool dump_raw = false);

  void resetStats() {
    for (const auto &iter : metrics) {
      Measure<uint64_t>* x = iter.second;
      x->clear();
    }
  }
};

void StatsManager::dumpRawStats() {
  for (const auto &iter : metrics) {
    spdlog::info("Dumping raw stats of exp: {}", _exp_name);
    iter.second->dumpStatsToFile(_exp_name);
  }
}

void StatsManager::saveStats(std::vector<std::string> agg_measures, bool dump_raw) {
  if(dump_raw)
    dumpRawStats();

  std::string stats_filename = "agg_" + _exp_name + "_stat.json";
  std::ofstream out(stats_filename, std::ios::out);
  
  for(std::string measure_name : agg_measures) {
    Measure<uint64_t>* key_measure = metrics[measure_name];
    std::vector<uint64_t>& key_measurements = key_measure->getRawMeasurements();
    aggr_stats["n_"     + measure_name]           = key_measure->numMeasurements();
    uint64_t min = 0, max = 0;
    if(key_measurements.size() > 0) {
      min = (*std::min_element(key_measurements.begin(), key_measurements.end()));
      max = (*std::max_element(key_measurements.begin(), key_measurements.end()));
    }
    aggr_stats["min_"   + measure_name + "_"+ LAT_UNIT]   = (min) / LAT_UNIT_MULTIPLIER;
    aggr_stats["max_"   + measure_name + "_"+ LAT_UNIT]   = (max) / LAT_UNIT_MULTIPLIER;
    aggr_stats["mean_"  + measure_name + "_"+ LAT_UNIT]   = getMean(key_measurements) / LAT_UNIT_MULTIPLIER;
    aggr_stats["p50_"   + measure_name + "_"+ LAT_UNIT]   = getPercentile(key_measurements, 0.50) / LAT_UNIT_MULTIPLIER;
    aggr_stats["p90_"   + measure_name + "_"+ LAT_UNIT]   = getPercentile(key_measurements, 0.90) / LAT_UNIT_MULTIPLIER;
    aggr_stats["p95_"   + measure_name + "_"+ LAT_UNIT]   = getPercentile(key_measurements, 0.950) / LAT_UNIT_MULTIPLIER;
    aggr_stats["p99_"   + measure_name + "_"+ LAT_UNIT]   = getPercentile(key_measurements, 0.990) / LAT_UNIT_MULTIPLIER;
    aggr_stats["p999_"  + measure_name + "_"+ LAT_UNIT]   = getPercentile(key_measurements, 0.9990) / LAT_UNIT_MULTIPLIER;
  }
  out << std::setw(4) << aggr_stats << std::endl;
  out.close();
  spdlog::info("Saved stats to file: {}", stats_filename);
}
