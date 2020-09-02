import pymc3 as pm
import pymc3.distributions as dist
from hypothesis import given
from hypothesis.strategies import binary,floats,integers,data,builds
import hypothesis.strategies as st
import arviz as az
from matplotlib import pyplot as plt
import numpy as np
from typing import List, Tuple
import random
from sklearn.feature_selection import mutual_info_regression
from hypothesis import given, settings, Phase, HealthCheck, strategies as st

class ProbabilityDistributions:
    """
    A "enum" class to mimic all the possible distributions
    """
    PossibleInts = [i for i in range(6)]
    PossibleFloats = [i for i in range(6,19)]

    #Ints Value
    Binomial, Bernoulli, Geometric, BetaBinomial, Poisson, DiscreteUniform = range(6)

    #Floats Value
    Normal, Uniform, TruncatedNormal = range(6,9)
    Beta, Exponential, Laplace, StudentT = range(9, 13)
    Cauchy, Gamma, LogNormal = range(13, 16)
    ChiSquared, Triangular, Logistic = range(16,19)

class ProbabilityGenerators:
    
    @staticmethod
    def IntList(name, data, length=1, possible_dist=ProbabilityDistributions.PossibleInts):
        """
        Generates a list of distributions to mimic all possible int values
        
        Returns:
        ----------
        Hypothesis.strategies.lists(Hypothesis.strategies.builds)

        Parameters:
        ----------
        name: String
            - The general name the list of distributions should take
        length: Int
            - The length of the list
        possible_dist: List[Int]
            - A list of ints to be chosen from ProbabilityDistributions, indicating which distributions to choose from
        """
        return ProbabilityGenerators.IntDist(data=data, name=name, shape=length)

        # f = (lambda r: ProbabilityGenerators.IntDist(r, name, possible_dist))
        # return st.lists(st.f, r=st.randoms(use_true_random=True)), min_size=length, max_size=length)

    @staticmethod
    def FloatList(name, data, length=1, possible_dist=ProbabilityDistributions.PossibleFloats):
        """
        Generates a list of distributions to mimic all possible float values
        
        Returns:
        ----------
        Hypothesis.strategies.lists(Hypothesis.strategies.builds)

        Parameters:
        ----------
        name: String
            - The general name the list of distributions should take
        length: Int
            - The length of the list
        possible_dist: List[Int]
            - A list of ints to be chosen from ProbabilityDistributions, indicating which distributions to choose from
        """
        return ProbabilityGenerators.FloatGenerator(name, data, shape=length)
        # f = (lambda r: ProbabilityGenerators.FloatGenerator(name, r, possible_dist))
        # return st.lists(st.f, r=st.randoms(use_true_random=True)), min_size=length, max_size=length)

    @staticmethod
    def IntDist(data, name="Int Distribution", distri = ProbabilityDistributions.PossibleInts, shape=1):
        """
        A method for generating a distributions to mimic int distribution

        Parameters
        ----------
        rand : Random
            Hypothesis random in order to enssure that the random is actually random
        name : str
            The name of the ditributions. It will have appended a UUID at the end to enssure validity
        Distri : List[ProbabilityDistributions]
            A list of the desired ProbabilityDistributions to be chosen random from
        """
        rand = data.draw(st.randoms(use_true_random=True))
        dist = rand.choice(distri)
        if dist == ProbabilityDistributions.Binomial:
            return ProbabilityGenerators.Binomial(data=data, name=name, shape=shape)
        elif dist == ProbabilityDistributions.Bernoulli:
            return ProbabilityGenerators.Bernoulli(data=data, name=name, shape=shape) 
        elif dist == ProbabilityDistributions.Geometric:
            return ProbabilityGenerators.Geometric(data=data, name=name, shape=shape)
        elif dist == ProbabilityDistributions.BetaBinomial:
            return ProbabilityGenerators.BetaBinomial(name=name, data=data, shape=shape)
        elif dist == ProbabilityDistributions.Poisson:
            return ProbabilityGenerators.Poisson(name=name, data=data, shape=shape)
        else:
            return ProbabilityGenerators.DiscreteUniform(name=name, data=data, shape=shape)
    
    @staticmethod
    def FloatGenerator(name, data, possible_dist = ProbabilityDistributions.PossibleFloats, shape=1):
        """
        Assumptions:
        Floats and Ints: In order to avoid values becoming to large in pymc3 a boundary for floats and ints has been set to [-10.000;10.000]
        Positive Floats: Setting the start value to 0.001 to ensure that checks >0 overholds.
        """
        rand = data.draw(st.randoms(use_true_random=True))
        dist = rand.choice(possible_dist)
        if dist == ProbabilityDistributions.Normal:
            return ProbabilityGenerators.Normal(data=data, name=name, shape=shape)
        elif dist == ProbabilityDistributions.Uniform:
            return ProbabilityGenerators.Uniform(data=data, name=name, shape=shape)
        elif dist == ProbabilityDistributions.TruncatedNormal:
            return ProbabilityGenerators.TruncatedNormal(name=name, data=data, shape=shape)
        elif dist == ProbabilityDistributions.Beta:
            return ProbabilityGenerators.Beta(name=name,data=data, shape=shape)
        elif dist == ProbabilityDistributions.Exponential:
            return ProbabilityGenerators.Exponential(name=name, data=data, shape=shape)
        elif dist == ProbabilityDistributions.Laplace:
            return ProbabilityGenerators.Laplace(name=name, data=data, shape=shape)
        elif dist == ProbabilityDistributions.StudentT:
            return ProbabilityGenerators.StudentT(name=name, data=data, shape=shape)
        elif dist == ProbabilityDistributions.Cauchy:
            return ProbabilityGenerators.Cauchy(name=name, data=data, shape=shape)
        elif dist == ProbabilityDistributions.Gamma:
            return ProbabilityGenerators.Gamma(name=name, data=data, shape=shape)
        elif dist == ProbabilityDistributions.LogNormal:
            return ProbabilityGenerators.LogNormal(name=name, data=data, shape=shape)
        elif dist == ProbabilityDistributions.ChiSquared:
            return ProbabilityGenerators.ChiSquared(name=name, data=data, shape=shape)
        elif dist == ProbabilityDistributions.Triangular:
            return ProbabilityGenerators.Triangular(name=name, data=data, shape=shape)
        else:
            return ProbabilityGenerators.Logistic(name=name, data=data, shape=shape)
    
    # Int Distributions
    
    @staticmethod
    def Binomial(data, name, shape=1):
        ints = st.integers(min_value=1, max_value=10000)
        probability = floats(min_value=0.001, max_value=0.9999,allow_infinity=False, allow_nan=False)
        n = data.draw(ints)
        p = data.draw(probability)
        a = dist.Binomial(name=name, n=n, p=p, shape=shape)
        b = ["Binomial", n,p]
        return (a,b)
    
    @staticmethod
    def Bernoulli(data, name, shape=1):
        probability = floats(min_value=0.001, max_value=0.9999,allow_infinity=False, allow_nan=False)
        p = data.draw(probability)
        a = dist.Bernoulli(name=name, p=p, shape=shape)
        b = ["Bernoulli", p]
        return (a,b)
        
    @staticmethod
    def Geometric(data, name, shape=1):
        probability = floats(min_value=0.001, max_value=0.9999,allow_infinity=False, allow_nan=False)
        p = data.draw(probability)
        a = dist.Geometric(name=name, p=p, shape=shape)
        b = ["Geometric", p]
        return (a,b)

    @staticmethod
    def BetaBinomial(data, name, shape=1):
        ints = st.integers(min_value=1, max_value=10000)
        positive_float = floats(min_value=0.001, max_value=10000,allow_infinity=False, allow_nan=False)
        n = data.draw(ints)
        alpha = data.draw(positive_float)
        beta = data.draw(positive_float)
        a = dist.BetaBinomial(name, alpha=alpha, beta=beta, n=n, shape=shape)
        b = ["BetaBinomial", n, alpha, beta]
        return (a,b)

    @staticmethod
    def Poisson(data, name, shape=1):
        non_negativ_float = floats(min_value=0, max_value=10000,allow_infinity=False, allow_nan=False)
        mu = data.draw(non_negativ_float)
        a = dist.Poisson(name, mu=mu, shape=shape)
        b = ["Poisson", mu]
        return (a,b) 
    
    @staticmethod
    def DiscreteUniform(data, name, shape=1):
        size = (st.tuples(st.integers(min_value=-10000, max_value=10000), st.integers(min_value=-10000, max_value=10000))
                    .map(sorted)
                    .filter(lambda x: x[0] < x[1]))
        lower, upper = data.draw(size)
        a = dist.DiscreteUniform(name, lower, upper, shape=shape)
        b = ["DiscreteUniform", lower, upper]
        return (a,b)

    # Float Distributions

    @staticmethod
    def Normal(data, name, shape=1):
        floats = st.floats(allow_infinity=False, allow_nan=False, min_value=-10000, max_value=10000)
        positive_floats = st.floats(min_value=0.1,allow_infinity=False, allow_nan=False,max_value=10000)
        mu = data.draw(floats)
        sigma = data.draw(positive_floats)
        a = dist.Normal(name=name, mu=mu, sigma=sigma, shape=shape)
        b = ["Normal", mu,sigma]
        return (a,b)

    @staticmethod
    def Uniform(data, name, shape=1):
        size = (st.tuples(st.integers(min_value=-10000, max_value=10000), st.integers(min_value=-10000, max_value=10000))
                    .map(sorted)
                    .filter(lambda x: x[0] < x[1]))
        lower, upper = data.draw(size)
        a = dist.Uniform(name, lower=lower, upper=upper, shape=shape)
        b = ["Uniform", lower, upper]
        return (a,b)

    @staticmethod
    def TruncatedNormal(data, name, shape=1):
        floats = st.floats(allow_infinity=False, allow_nan=False, min_value=-10000, max_value=10000)
        positive_floats = st.floats(min_value=0.1,allow_infinity=False, allow_nan=False,max_value=10000)
        size = (st.tuples(st.integers(min_value=-10000, max_value=10000), st.integers(min_value=-10000, max_value=10000))
                    .map(sorted)
                    .filter(lambda x: x[0] < x[1]))
        mu = data.draw(floats)
        sigma = data.draw(positive_floats)
        lower, upper = data.draw(size)
        a = dist.TruncatedNormal(name, mu=mu, sigma=sigma, lower=lower, upper=upper, shape=shape)
        b = ["TruncatedNormal", mu, sigma, lower, upper]
        return (a,b)
    
    @staticmethod
    def Beta(data, name, shape=1):
        positive_floats = st.floats(min_value=0.1,allow_infinity=False, allow_nan=False,max_value=10000)
        alpha = data.draw(positive_floats)
        beta = data.draw(positive_floats)
        a = dist.Beta(name, alpha=alpha, beta=beta, shape=shape)
        b = ["Beta", alpha, beta]
        return (a,b)

    @staticmethod
    def Exponential(data, name, shape=1):
        positive_floats = st.floats(min_value=0.1,allow_infinity=False, allow_nan=False,max_value=10000)
        lam = data.draw(positive_floats)
        a = dist.Exponential(name, lam, shape=shape)
        b = ["Exponential", lam]
        return (a,b)

    @staticmethod
    def Laplace(data, name, shape=1):
        floats = st.floats(allow_infinity=False, allow_nan=False, min_value=-10000, max_value=10000)
        positive_floats = st.floats(min_value=0.1,allow_infinity=False, allow_nan=False,max_value=10000)
        mu = data.draw(floats)
        bi = data.draw(positive_floats)
        a = dist.Laplace(name, mu=mu, b=bi, shape=shape)
        b = ["Laplace", mu, bi]
        return (a,b)

    @staticmethod
    def StudentT(data, name, shape=1):
        positive_floats = st.floats(min_value=0.1,allow_infinity=False, allow_nan=False,max_value=10000)
        floats = st.floats(allow_infinity=False, allow_nan=False, min_value=-10000, max_value=10000)
        nu = data.draw(positive_floats)
        mu = data.draw(floats)
        sigma = data.draw(positive_floats)
        a = dist.StudentT(name, nu=nu, mu=mu, sigma=sigma, shape=shape)
        b = ["StudentT", nu, mu, sigma]
        return (a,b)

    @staticmethod
    def Cauchy(data, name, shape=1):
        positive_floats = st.floats(min_value=0.1,allow_infinity=False, allow_nan=False,max_value=10000)
        floats = st.floats(allow_infinity=False, allow_nan=False, min_value=-10000, max_value=10000)
        alpha = data.draw(floats)
        beta = data.draw(positive_floats)
        a = dist.Cauchy(name, alpha=alpha, beta=beta, shape=shape)
        b = ["Cauchy", alpha, beta]
        return (a,b)
    
    @staticmethod
    def Gamma(data, name, shape=1):
        positive_floats = st.floats(min_value=0.1,allow_infinity=False, allow_nan=False,max_value=10000)
        alpha = data.draw(positive_floats)
        beta = data.draw(positive_floats)
        a = dist.Gamma(name, alpha, beta, shape=shape)
        b = ["Gamma", alpha, beta]
        return (a,b)

    @staticmethod
    def LogNormal(data, name, shape=1):
        positive_floats = st.floats(min_value=0.1,allow_infinity=False, allow_nan=False,max_value=10000)
        mu = data.draw(positive_floats)
        sigma = data.draw(positive_floats)
        a = dist.Lognormal(name, mu=mu, sigma=sigma, shape=shape)
        b = ["LogNormal", mu, sigma]
        return (a,b)

    @staticmethod
    def ChiSquared(data, name, shape=1):
        positive_int = integers(min_value = 1, max_value=10000)
        nu = data.draw(positive_int)
        a = dist.ChiSquared(name, nu, shape=shape)
        b = ["ChiSquared", nu]
        return (a,b)

    @staticmethod
    def Triangular(data, name, shape=1):
        floats = st.floats(allow_infinity=False, allow_nan=False, min_value=-10000, max_value=10000)
        float_size = (st.tuples(floats, floats,floats)).map(sorted).filter(lambda x: x[0] < x[1] < x[2])
        lower, middle, upper = data.draw(float_size)
        a = dist.Triangular(name, lower=lower, c=middle, upper=upper, shape=shape)
        b = ["Triangular", lower, middle, upper]
        return(a,b)

    @staticmethod
    def Logistic(data, name, shape=1):
        floats = st.floats(allow_infinity=False, allow_nan=False, min_value=-10000, max_value=10000)
        positive_floats = st.floats(min_value=0.1,allow_infinity=False, allow_nan=False,max_value=10000)
        mu = data.draw(floats)
        s = data.draw(positive_floats)
        a = dist.Logistic(name, mu=mu, s=s, shape=shape)
        b = ["Logistic", mu, s]
        return (a,b)
    
    @staticmethod
    def Analyze(*args, **kwargs):
        def inner(func):
            traces = []
            max_examples = 1 if "max_examples" not in kwargs else kwargs["max_examples"]
            n = 2 if "N" not in kwargs else kwargs["N"]
            samples = 100 if "num_samples" not in kwargs else kwargs["num_samples"]
            @settings(max_examples=max_examples, deadline=None, phases=[Phase.generate],suppress_health_check=[HealthCheck.too_slow])
            @given(st.data())
            def helper(data):
                with pm.Model() as model:
                    N = n # Size of the database
                    x = np.empty(N+1, dtype=object)
                    #Test subject
                    age_alice_database = pm.Uniform("alice_age", lower=0, upper=100)
                    name_age_alice = "alice_age"
                    name_alice_database = pm.Constant("name_alice_database", 0)  
                    
                    x[0] = (name_alice_database, age_alice_database)
                    def parse(argument, islist=False, istuple=False):
                        if isinstance(argument, list):
                            if len(argument) != 1:
                                raise Exception("The size of the list has to contain only one element")
                            dist, info = parse(argument[0], islist=True,istuple=istuple)
                            return (dist,info)
                        elif isinstance(argument, tuple):
                            dist, info = zip(*[parse(arg, islist=islist,istuple=True) for arg in argument])
                            return (dist, info)
                        elif isinstance(argument, int):
                            if islist:
                                dist, info = ProbabilityGenerators.IntList(name="IntList", data=data, length=N)
                                return (dist,info)
                            else:
                                dist,info = data.draw(ProbabilityGenerators.IntDist(data=data, name="IntDist"))
                                return (dist,info)
                        elif isinstance(argument, float):
                            if islist:
                                dist, info = ProbabilityGenerators.FloatList(name="FloatList", data=data, length=N)
                                return (dist, info)
                            else:
                                dist, info = ProbabilityGenerators.FloatGenerator(name="FloatDist", data=data, shape=1)
                                return (dist,info)
                        else:
                            raise Exception("Type is currently not supported")
                    dist, info = zip(*[parse(x) for x in args])
                    dist = dist[0]
                    for i in range(0,N):
                        x[i+1] = tuple([d[i] for d in dist])

                    average = pm.Deterministic("average", func(x))
                    num_samples = samples

                    trace = pm.sample(num_samples, cores=1, step=pm.NUTS()) 
                    output = trace["average"]
                    alice_age = trace["alice_age"]
                    mututal_info = mutual_info_regression([[i] for i in alice_age], output, discrete_features=False)
                    traces.append(mututal_info)                  
            helper()
            return (lambda x=traces:x)
        return inner
