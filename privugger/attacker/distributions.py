"""
Module describing possible distributions
"""
import numpy as np

POSSIBLE_INTS = [0,1,2,3,4]
POSSIBLE_FLOATS = [i for i in range(6,13)]

#Ints Value
BINOMIAL, BERNOULLI, GEOMETRIC, BETA_BINOMIAL, POISSON, DISCRETE_UNIFORM = range(6)

#Floats Value
NORMAL, UNIFORM, TRUNCATED_NORMAL = range(6,9)
BETA, EXPONENTIAL, LAPLACE, STUDENT_T = range(9, 13)
CAUCHY, GAMMA = range(13, 15)

#MINIMUM COVERAGE
MINIMUM_PERCANTAGE_COVERAGE = 5.0
MINIMUM_COVERAGE = lambda low, high: (MINIMUM_PERCANTAGE_COVERAGE/(high-low))

#Distributions_Support
class Support:
    BINOMIAL = (0, np.inf)
    BERNOULLI = (0,1)
    GEOMETRIC = (1, np.inf)
    BETA_BINOMIAL = (0,np.inf)
    POISSON = (0,np.inf)
    DISCRETE_UNIFORM = (-np.inf, np.inf)

    NORMAL = (-np.inf, np.inf)
    UNIFORM = (-np.inf, np.inf)
    TRUNCATED_NORMAL = (-np.inf, np.inf)
    BETA = (0,1)
    EXPONENTIAL = (0, np.inf) # Figure if this makes sense. Since if the range is (10, 30) than exponential has to either be shifted or it is useless
    LAPLACE = (-np.inf, np.inf)
    STUDENT_T = (-np.inf, np.inf)
    CAUCHY = (-np.inf, np.inf)
    GAMMA = (0, np.inf)