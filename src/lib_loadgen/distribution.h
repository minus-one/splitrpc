// Copyright (c) Adithya Kumar, The Pennsylvania State University. All rights reserved.
// Licensed under the MIT License.

/*
 * Header-only mechanism to create different distributions
 * Currently supports:-
 * (i) Deterministic - Fixed rate
 * (ii) Exponential - parameterized by mean rate
 * (iii) HyperExponential - parameterized by mean rate and coefficient of variation (CV^2) values.
 */
#ifndef DISTRIBUTION_H
#define DISTRIBUTION_H

#include "common_defs.h"
#include <cmath>

class Distribution
{
  public:
    Distribution()
    {
    }
    virtual ~Distribution()
    {
    }
    // Returns next random number
    virtual double nextRand() = 0;

    // Set the mean value of the distribution (may not make sense for all types)
    virtual void setMeanValue(double mean) = 0;
};

// Inv-Transform method to generate exponentials
// Pull random num(0,1), use inverse, calc exp
// Inv(F(X)) = - (log(1-X)/Mu) [X~unif(0,1)]
class DistributionExp : public Distribution
{                                          
  private:                                   
    double rate; // Mean rate              

  public:                                    
    DistributionExp(double mean)           
    {                                      
      rate = mean;                         
    }                                      
    virtual ~DistributionExp()             
    {}                                     
    
    virtual double nextRand()
    {
      return -(std::log(uniform01())/rate);
    }

    // Set the mean of the exponential,    
    // To generate at the set rate         
    virtual void setMeanValue(double mean) 
    {                                      
      rate = mean;                         
    }                                      
};                                         

/*
 * Generates HyperExponentials as per the target
 * mean and CV^2 value.
 * Uses Morse's method taken from Simulating Computer
 * Systems by H.M. MacDougall
 * Source taken from:
 * http://www.csee.usf.edu/~kchriste/tools/genhyp2.c
 * Rate is the expected mean
 * CoV2 is the desired Coefficient-of-Variance^2
 */
class DistributionHyp : public Distribution
{
  private:
    double rate, CoV2;

  public:
    DistributionHyp(double mean, double cov)
    {
      rate = mean;
      CoV2 = cov;
    }
    virtual ~DistributionHyp()
    {}

    virtual double nextRand()
    {
      double p;                     // Probability value for Morse's method
      double z1, z2;                // Uniform random numbers from 0 to 1
      double hyp_value;             // Computed exponential value to be returned
      double temp = 0.0;            // Temporary double value

      // Pull a uniform random number (0 < z1 < 1)
      z1 = uniform01();

      // Pull another uniform random number (0 < z2 < 1)
      z2 = uniform01();

      // Compute hyperexponential random variable using Morse's method
      p = 0.5 * (1.0 - sqrt((CoV2 - 1.0) / (CoV2 + 1.0)));
      if (z1 > p)
        temp = rate / (1.0 - p);
      else
        temp = rate / p;
      hyp_value = -0.5 * temp * std::log(z2);
      return(hyp_value);
    }

    virtual void setMeanValue(double mean)
    {
      rate = mean;
    }

    virtual void setCoVValue(double cov)
    {
      CoV2 = cov;
    }

};

// Will always return the same value, xkcd style!
class DistributionDet : public Distribution
{
  private:
    double val;

  public:
    DistributionDet(double mean)
    {
      val = mean;
    }
    virtual ~DistributionDet()
    {
    }
    
    virtual double nextRand()
    {
      return val;
    }

    virtual void setMeanValue(double mean)
    {
      val = mean;
    }
};

#endif /* DISTRIBUTION_H */
