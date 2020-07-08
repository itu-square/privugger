import pymc3 as pm
from hypothesis import given
from hypothesis.strategies import binary,floats,integers,data,builds
import arviz as az
from matplotlib import pyplot as plt
import random

seed = 0

"""
TODO: 
- The name of the distribution has to be made generic
- Make the generator for float distribution
"""
class ProbabilityGenerators:
    def IntDist(self):
        """
        A method for generating a distribution to mimic int distribution

        Distribution Generators
        ----------
        n : int 
            The number of cases for a binomial distribution
        p : float (between 0.0 and 1.0)
            The probability of success for the given test
        
        TODO
        ----
        Problem using random with in here, since it makes the generator only choose the same value
        """
        global seed
        seed += 1
        dist = random.choice([0,1,2])
        if dist == 0:
            return builds(pm.distributions.Binomial, f"Binomial Distribution, seed: {seed}", integers(), floats(min_value = 0.001, max_value=1))
        elif dist == 1:
            return builds(pm.distributions.Bernoulli, f"Bernoulli Distribution, seed: {seed}", floats(min_value = 0.001, max_value=1))
        else:
            return builds(pm.distributions.Geometric, f"Geometric Distribution, seed: {seed}", floats(min_value = 0.001, max_value=1))

    def FloatDist(self, shape=1):
        """
        A method for generating a distributions to mimics floats of any size

        Parameters
        ----------
        shape : int
            The shape of the input values
        TODO
        ----------
        Make it so that shape is fetched from the data
        """
        global seed
        seed += 1
        return builds(pm.distributions.Normal, f"Normal Distribution, seed: {seed}", mu=floats(), sigma=floats(min_value=0.001), shape=shape)

    def SingleDigitFloatDist(self):
        """
        A distribution for values that spreads between 0 and 1

        Parameters
        ----------
        Alpha: float
            The shape of the data (has to be larger than 0)
        Beta: float
            The rate of the data (Has to be larger than 0)
        """
        global seed
        seed += 1
        return builds(pm.distributions.Gamma, f"Normal Distribution alpha, seed: {seed}", floats(min_value=0.001), floats(min_value=0.001))

    def PositiveFloatDist(self):
        """
        A distribution to mimic all values that are larger than 0

        Parameters
        ----------
        Lambda : float
            The incremental rate of the distribution (has to be larger than 0)
        """
        global seed
        seed += 1
        return builds(pm.distributions.Exponential, f"Normal Distribution, seed: {seed}", floats(min_value=0.001))

    def GeneralDist(self, values):
        """
        Ideally the function that will return a generator for the distribution needed
        """
        if isinstance(values, int):
            return self.IntDist()
        elif isinstance(values, float):
            # TODO: Figure out how values actually is supposed to be feeded to the method
            if 1 > values > 0:
                return self.SingleDigitFloatDist()
            elif values > 0:
                return self.PositiveFloatDist()
            else:
                return self.FloatDist() 
        else:
            return None