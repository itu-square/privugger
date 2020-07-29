import pymc3 as pm
from hypothesis import given
from hypothesis.strategies import binary,floats,integers,data,builds
import hypothesis.strategies as st
import arviz as az
from matplotlib import pyplot as plt
import random

class ProbabilityDistributions:
    """
    A "enum" class to mimic all the possible distributions
    """
    NormalDist, UniformDist, BinomialDist, BernoulliDist, GeometricDist = range(5)

class ProbabilityGenerators:
    @staticmethod
    def IntListDist(name="Int Distribution", length=1, possible_dist = [i for i in range(2,5)]):
        f = (lambda r: ProbabilityGenerators.IntDist(r, name, possible_dist))
        return st.lists(st.builds(f, r=st.randoms(use_true_random=True)), min_size=length, max_size=length)

    @staticmethod
    def ListDist(name="", length=1, possible_dist = []):
        f = (lambda r: ProbabilityGenerators.FloatGenerator(name, r, possible_dist))
        return st.lists(st.builds(f, r=st.randoms(use_true_random=True)), min_size=length, max_size=length)

    @staticmethod
    def IntDist(rand, name="Int Distribution", distri = []):
        """
        A method for generating a distributions to mimic int distribution

        distributions Generators
        ----------
        n : int 
            The number of cases for a binomial distribution
        p : float (between 0.0 and 1.0)
            The probability of success for the given test
        """
        dist = rand.choice(distri)
        if dist == ProbabilityDistributions.BinomialDist:
            return builds(ProbabilityGenerators.BinomialDist, 
                name=st.from_regex(f"^{name}$"), 
                n=integers(min_value = 1), 
                p=floats(min_value = 0.001, max_value=1),
                uuids=st.uuids())
        elif dist == ProbabilityDistributions.BernoulliDist:
            return builds(ProbabilityGenerators.BernoulliDist, 
                name=st.from_regex(f"^{name}$"), 
                p=floats(min_value = 0.001, max_value=1),
                uuids=st.uuids())
        elif dist == ProbabilityDistributions.GeometricDist:
            return builds(ProbabilityGenerators.GeometricDist, 
                name=st.from_regex(f"^{name}$"), 
                p=floats(min_value = 0.001, max_value=1),
                uuids=st.uuids())
    
    @staticmethod
    def BinomialDist(name, n, p, uuids):
        return (pm.distributions.Binomial(name=name+str(uuids), n=n, p=p), ["Binomial", n,p], name+str(uuids))
    
    @staticmethod
    def BernoulliDist(name, p, uuids):
        return (pm.distributions.Bernoulli(name=name+str(uuids), p=p), ["Bernoulli", p], name+str(uuids))
        
    @staticmethod
    def GeometricDist(name, p, uuids):
        return (pm.distributions.Geometric(name=name+str(uuids), p=p), ["Geometric", p], name+str(uuids))

    @staticmethod
    def FloatGenerator(name, rand, possible_dist = [i for i in range(2)]):
        if not len(possible_dist):
            possible_dist = [i for i in range(2)]
        dist = rand.choice(possible_dist)
        if dist == ProbabilityDistributions.NormalDist:
            return builds(ProbabilityGenerators.NormalDist, name=st.from_regex(f"^{name}$"),
                mu = st.floats(allow_infinity=False,
                    allow_nan=False,
                    min_value=-10000,
                    max_value=10000),
                sigma=st.floats(min_value=0.001,
                    allow_infinity=False, 
                    allow_nan=False,
                    max_value=10000),
                uuids=st.uuids()
            )
        elif dist == ProbabilityDistributions.UniformDist:
            return builds(ProbabilityGenerators.UniformDist,
                name=st.from_regex(f"^{name}$"),
                uuids = st.uuids(),
                size=(st.tuples(st.integers(min_value=-10000, max_value=10000), st.integers(min_value=-10000, max_value=10000))
                    .map(sorted)
                    .filter(lambda x: x[0] < x[1]))
            )

    @staticmethod
    def NormalDist(name, uuids, mu, sigma):
        return (pm.distributions.Normal(name=name+str(uuids), mu=mu, sigma=sigma), ["normal", mu,sigma], name+str(uuids))

    @staticmethod
    def UniformDist(name, uuids, size):
        return (pm.distributions.Uniform(name+str(uuids), lower=size[0], upper=size[1]), ["Uniform", size[0], size[1]], name+str(uuids))


    @staticmethod
    def FloatDist(name, rand, mu, sigma, uuids, size, distri = []):
        """
        A distributions for randomly choosing a float distribution
            - Has to include all distributions that can mimic floats
        """
        dist = random.choice(distri)
        if dist == ProbabilityDistributions.NormalDist:
            return (pm.distributions.Normal(name=name+str(uuids), mu=mu, sigma=sigma), ["normal", mu,sigma], name+str(uuids))
        elif dist == ProbabilityDistributions.UniformDist:
            return (pm.distributions.Uniform(name+str(uuids), lower=size[0], upper=size[1]), ["Uniform", size[0], size[1]], name+str(uuids))

    @staticmethod
    def SingleDigitFloatDist(name="SingleDigitFloats"):
        """
        A distributions for values that spreads between 0 and 1

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
        A distributions to mimic all values that are larger than 0

        Parameters
        ----------
        Lambda : float
            The incremental rate of the distributions (has to be larger than 0)
        """
        return builds(pm.distributions.Exponential, st.from_regex(f"^{name}$"), floats(min_value=0.001))