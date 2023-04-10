// Copyright (c) Adithya Kumar, The Pennsylvania State University. All rights reserved.
// Licensed under the MIT License.

#ifndef LOAD_GENERATOR_H
#define LOAD_GENERATOR_H

#include "distribution.h"
#include "common_defs.h"

// Simple load-generator, function yields after waiting as per arrival rate
class LoadGenerator
{
  // Arrival Process
  Distribution* arrDist;

  public:
    // Initialize the load generator with arrival rate (lambda)
    // Set ARR_RATE_COV to 0.0 for deterministic arrival
    // Else, COV determines the Coefficient of variation of arrival distribution
    // By default ARR_RATE_COV is 1.0 for exponential distribution, that is poisson arrivals
    LoadGenerator(double lambda) {
      // Setup Distributions for ArrivalRate
      uint16_t arrRateCoV = readEnvInfo<uint16_t>("P2P_RPC_ARR_RATE_COV", DEF_ARR_RATE_COV);
      if(arrRateCoV == 0) {
        printf("LoadGen using uniform deterministic arrival pattern with lambda: %f\n", lambda);
        arrDist = new DistributionDet(lambda);
      } else if(arrRateCoV == 1) {
        printf("LoadGen using Poisson arrival pattern with lambda: %f\n", 1.0/lambda);
        arrDist = new DistributionExp(1.0/lambda);
      } else {
        printf("LoadGen using HyperExponential arrival pattern\n");
        arrDist = new DistributionHyp(lambda, static_cast<double>(arrRateCoV));
      }
    }

    ~LoadGenerator() {
      delete arrDist;
    }

    // Waits until the next request arrival, and return the time-stamp of arrival
    // Pass the timestamp back again to the next call of nextReqArr to generate load
    // Set blocking=false, in case you don't want the function to be blocking.
    uint64_t nextReqArr(uint64_t prevReqTimeNs, bool blocking=true) {
      uint64_t nextArrNs = prevReqTimeNs + (arrDist->nextRand() * 1E9);
      // Wait for next arrival
      if(blocking)
        sleepUntil(nextArrNs);
      return nextArrNs;
    }
};

#endif /* REQUEST_GENERATOR_H */
