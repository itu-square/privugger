import pymc3 as pm
from hypothesis import given
from hypothesis.strategies import binary,floats,integers,data,builds
import hypothesis.strategies as st
import arviz as az
from matplotlib import pyplot as plt
import random

def UniformHelper(name="Uniform", size=(0,10)):
    """
    A helper method for uniform
    """
    return pm.distributions.Uniform(name, lower=size[0], upper=size[1])

class ProbabilityGenerators:
    @staticmethod
    def IntDist(name="Int Distribution"):
        """
        A method for generating a distribution to mimic int distribution

        Distribution Generators
        ----------
        n : int 
            The number of cases for a binomial distribution
        p : float (between 0.0 and 1.0)
            The probability of success for the given test
        """
        dist = random.choice([0,1,2])
        if dist == 0:
            return builds(pm.distributions.Binomial, s.from_regex(f"^{name}$"), integers(min_value = 1), floats(min_value = 0.001, max_value=1))
        elif dist == 1:
            return builds(pm.distributions.Bernoulli, st.from_regex(f"^{name}$"), floats(min_value = 0.001, max_value=1))
        else:
            return builds(pm.distributions.Geometric, st.from_regex(f"^{name}$"), floats(min_value = 0.001, max_value=1))
    @staticmethod
    def FloatDist(name=""):
        """
        A distribution for randomly choosing a float distribution

        TODO: 
            - Has to include all distributions that can mimic floats
            - Include additional parameters
        """
        numberOfDist = 10
        dist = random.choice([0,1])
        if not dist:
            return ProbabilityGenerators.NormalDist(name)
        else:
            return ProbabilityGenerators.UniformDist(name)
    @staticmethod
    def NormalDist(name="normal", shape=1):
        """
        A method for generating a distributions to mimics floats of any size

        Parameters
        ----------
        name : string
            The name of the distribution
        shape : int
            The shape of the input values
        """
        return builds(pm.distributions.Normal, 
                        st.from_regex(f"^{name}$"), 
                        mu=floats(allow_infinity=False, 
                            allow_nan=False), 
                        sigma=floats(min_value=0.001,  
                            allow_infinity=False, 
                            allow_nan=False), shape=integers(min_value=shape, max_value=shape))
    
    @staticmethod
    def UniformDist(name="Uniform"):
        """
        A method for generating Uniform distribution
        - Generates distribution for floats

        Parameters
        ----------
        name : string
            Name of the distribution 
        """
        return builds(UniformHelper, 
                name=st.from_regex(f"^{name}$"),
                size=(st.tuples(integers(), integers())
                    .map(sorted)
                    .filter(lambda x: x[0] < x[1])) 
                )

    @staticmethod
    def SingleDigitFloatDist(name="SingleDigitFloats"):
        """
        A distribution for values that spreads between 0 and 1

        Parameters
        ----------
        Alpha: float
            The shape of the data (has to be larger than 0)
        Beta: float
            The rate of the data (Has to be larger than 0)
        """
        return builds(pm.distributions.Gamma, st.from_regex(f"^{name}$"), floats(min_value=0.001), floats(min_value=0.001))

    @staticmethod
    def PositiveFloatDist(name="Non-negative Floats"):
        """
        A distribution to mimic all values that are larger than 0

        Parameters
        ----------
        Lambda : float
            The incremental rate of the distribution (has to be larger than 0)
        """
        return builds(pm.distributions.Exponential, st.from_regex(f"^{name}$"), floats(min_value=0.001))

    @staticmethod
    def GeneralDist(values):
        """
        Ideally the function that will return a generator for the distribution needed

        TODO:
            - Support numpy values, for more in depth features
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