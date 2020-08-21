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
    def IntList(name="Int Distribution", length=1, possible_dist=ProbabilityDistributions.PossibleInts):
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
        f = (lambda r: ProbabilityGenerators.IntDist(r, name, possible_dist))
        return st.lists(st.builds(f, r=st.randoms(use_true_random=True)), min_size=length, max_size=length)

    @staticmethod
    def FloatList(name="", length=1, possible_dist =ProbabilityDistributions.PossibleFloats):
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
        f = (lambda r: ProbabilityGenerators.FloatGenerator(name, r, possible_dist))
        return st.lists(st.builds(f, r=st.randoms(use_true_random=True)), min_size=length, max_size=length)

    @staticmethod
    def IntDist(rand=random.random(), name="Int Distribution", distri = ProbabilityDistributions.PossibleInts):
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
        name = st.from_regex(f"^{name}$")
        number_of_test = integers(min_value = 1, max_value=10000)
        probability = floats(min_value=0.001, max_value=0.9999,allow_infinity=False, allow_nan=False)
        positive_float = floats(min_value=0.001, max_value=10000,allow_infinity=False, allow_nan=False)
        non_negative_float = floats(min_value=0, max_value=10000,allow_infinity=False, allow_nan=False)
        uuids = st.uuids()
        dist = rand.choice(distri)
        size = (st.tuples(st.integers(min_value=-10000, max_value=10000), st.integers(min_value=-10000, max_value=10000))
                    .map(sorted)
                    .filter(lambda x: x[0] < x[1]))
        
        if dist == ProbabilityDistributions.Binomial:
            return builds(ProbabilityGenerators.Binomial, 
                name=name, n=number_of_test, p=probability, uuids=uuids)
        elif dist == ProbabilityDistributions.Bernoulli:
            return builds(ProbabilityGenerators.Bernoulli, 
                name=name, p=probability, uuids=uuids)
        elif dist == ProbabilityDistributions.Geometric:
            return builds(ProbabilityGenerators.Geometric, 
                name=name, p=probability, uuids=uuids)
        elif dist == ProbabilityDistributions.BetaBinomial:
            return builds(ProbabilityGenerators.BetaBinomial, name=name, uuids=uuids, 
                n=number_of_test, alpha=positive_float, beta=positive_float)
        elif dist == ProbabilityDistributions.Poisson:
            return builds(ProbabilityGenerators.Poisson, name=name, uuids=uuids,
                mu=non_negative_float)
        else:
            return builds(ProbabilityGenerators.DiscreteUniform, name=name, uuids=uuids,
                size=size)
    
    @staticmethod
    def FloatGenerator(name, rand=random.random(), possible_dist = ProbabilityDistributions.PossibleFloats):
        """

        Assumptions:
        Floats and Ints: In order to avoid values becoming to large in pymc3 a boundary for floats and ints has been set to [-10.000;10.000]
        Positive Floats: Setting the start value to 0.001 to ensure that checks >0 overholds.
        """
        floats = st.floats(allow_infinity=False, allow_nan=False, min_value=-10000, max_value=10000)
        positive_floats = st.floats(min_value=0.001,allow_infinity=False, allow_nan=False,max_value=10000)
        uuids = st.uuids()
        name = st.from_regex(f"^{name}$")
        ints = st.integers(min_value=-10000, max_value=10000)
        size = (st.tuples(ints, ints)).map(sorted).filter(lambda x: x[0] < x[1])
        float_size = (st.tuples(floats, floats,floats)).map(sorted).filter(lambda x: x[0] < x[1] < x[2])

        dist = rand.choice(possible_dist)
        if dist == ProbabilityDistributions.Normal:
            return builds(ProbabilityGenerators.Normal, name=name,
                mu = floats, sigma=positive_floats, uuids=uuids)
        elif dist == ProbabilityDistributions.Uniform:
            return builds(ProbabilityGenerators.Uniform,
                name=name, uuids = uuids,size=size)
        elif dist == ProbabilityDistributions.TruncatedNormal:
            return builds(ProbabilityGenerators.TruncatedNormal, name=name, uuids=uuids,
                mu=floats, sigma=positive_floats, size=size)
        elif dist == ProbabilityDistributions.Beta:
            return builds(ProbabilityGenerators.Beta, name=name, uuids=uuids,
                alpha=positive_floats, beta=positive_floats)
        elif dist == ProbabilityDistributions.Exponential:
            return builds(ProbabilityGenerators.Exponential, name=name, uuids=uuids,
                lam=positive_floats)
        elif dist == ProbabilityDistributions.Laplace:
            return builds(ProbabilityGenerators.Laplace, name=name, uuids=uuids, 
                mu=floats,mean=positive_floats)
        elif dist == ProbabilityDistributions.StudentT:
            return builds(ProbabilityGenerators.StudentT,name=name, uuids=uuids,
                nu=positive_floats, mu=floats, sigma=floats, lam=positive_floats)
        elif dist == ProbabilityDistributions.Cauchy:
            return builds(ProbabilityGenerators.Cauchy, name=name, uuids=uuids, 
                alpha=floats, beta=positive_floats)
        elif dist == ProbabilityDistributions.Gamma:
            return builds(ProbabilityGenerators.Gamma, name=name, uuids=uuids,
                alpha=positive_floats, beta=positive_floats)
        elif dist == ProbabilityDistributions.LogNormal:
            return builds(ProbabilityGenerators.LogNormal, name=name, uuids=uuids,
                mu=floats, sigma=positive_floats, tau=positive_floats)
        elif dist == ProbabilityDistributions.ChiSquared:
            return builds(ProbabilityGenerators.ChiSquared, name=name, uuids=uuids,
                nu=positive_floats)
        elif dist == ProbabilityDistributions.Triangular:
            return builds(ProbabilityGenerators.Triangular, name=name, uuids=uuids,
                size=float_size)
        else:
            return builds(ProbabilityGenerators.Logistic, name=name, uuids=uuids, 
                mu=floats, s=positive_floats)
    
    # Int Distributions
    
    @staticmethod
    def Binomial(name, n, p, uuids):
        a = dist.Binomial(name=name+str(uuids), n=n, p=p)
        b = ["Binomial", n,p]
        c = name+str(uuids)
        return (a,b,c)
    
    @staticmethod
    def Bernoulli(name, p, uuids):
        a = dist.Bernoulli(name=name+str(uuids), p=p)
        b = ["Bernoulli", p]
        c = name+str(uuids)
        return (a,b,c)
        
    @staticmethod
    def Geometric(name, p, uuids):
        a = dist.Geometric(name=name+str(uuids), p=p)
        b = ["Geometric", p]
        c = name+str(uuids)
        return (a,b,c)

    @staticmethod
    def BetaBinomial(name, uuids, n, alpha, beta):
        a = dist.BetaBinomial(name+str(uuids), alpha=alpha, beta=beta, n=n)
        b = ["BetaBinomial", n, alpha, beta]
        c = name+str(uuids)
        return (a,b,c)

    @staticmethod
    def Poisson(name, uuids, mu):
        a = dist.Poisson(name+str(uuids), mu=mu)
        b = ["Poisson", mu]
        c = name+str(uuids)
        return (a,b,c) 
    
    @staticmethod
    def DiscreteUniform(name, uuids, size):
        a = dist.DiscreteUniform(name+str(uuids), size[0], size[1])
        b = ["DiscreteUniform", size[0], size[1]]
        c = name+str(uuids)
        return (a,b,c)

    # Float Distributions

    @staticmethod
    def Normal(name, uuids, mu, sigma):
        a = dist.Normal(name=name+str(uuids), mu=mu, sigma=sigma)
        b = ["Normal", mu,sigma]
        c = name+str(uuids)
        return (a,b,c)

    @staticmethod
    def Uniform(name, uuids, size):
        a = dist.Uniform(name+str(uuids), lower=size[0], upper=size[1])
        b = ["Uniform", size[0], size[1]]
        c = name+str(uuids)
        return (a,b,c)

    @staticmethod
    def TruncatedNormal(name, uuids, mu, sigma, size):
        a = dist.TruncatedNormal(name+str(uuids), mu=mu, sigma=sigma, lower=size[0], upper=size[1])
        b = ["TruncatedNormal", mu, sigma, size[0], size[1]]
        c = name+str(uuids)
        return (a,b,c)
    
    @staticmethod
    def Beta(name, uuids, alpha, beta):
        a = dist.Beta(name+str(uuids), alpha = alpha, beta = beta)
        b = ["Beta", alpha, beta]
        c = name+str(uuids)
        return (a,b,c)

    @staticmethod
    def Exponential(name, uuids, lam):
        a = dist.Exponential(name+str(uuids), lam)
        b = ["Exponential", lam]
        c = name+str(uuids)
        return (a,b,c)

    @staticmethod
    def Laplace(name, uuids, mu, mean):
        a = dist.Laplace(name+str(uuids), mu, mean)
        b = ["Laplace", mu, mean]
        c = name+str(uuids)
        return (a,b,c)

    @staticmethod
    def StudentT(name, uuids, nu, mu, sigma, lam):
        if sigma > 0:
            a = dist.StudentT(name+str(uuids), nu, mu, sigma)
            b = ["StudentT", nu, mu, sigma]
        else:
            a = dist.StudentT(name+str(uuids), nu, mu, lam)
            b = ["StudentT", nu, mu, lam]
        c = name+str(uuids)
        return (a,b,c)

    @staticmethod
    def Cauchy(name, uuids, alpha, beta):
        a = dist.Cauchy(name+str(uuids), alpha, beta)
        b = ["Cauchy", alpha, beta]
        c = name+str(uuids)
        return (a,b,c)
    
    @staticmethod
    def Gamma(name, uuids, alpha, beta):
        a = dist.Gamma(name+str(uuids), alpha, beta)
        b = ["Gamma", alpha, beta]
        c = name+str(uuids)
        return (a,b,c)

    @staticmethod
    def LogNormal(name, uuids, mu, sigma, tau):
        if sigma > 0:
            a = dist.Lognormal(name+str(uuids), mu=mu, sigma=sigma)
        else:
            a = dist.Lognormal(name+str(uuids), mu=mu, tau=tau)
        b = ["LogNormal", mu, sigma, tau]
        c = name+str(uuids)
        return (a,b,c)

    @staticmethod
    def ChiSquared(name, uuids, nu):
        a = dist.ChiSquared(name+str(uuids), nu)
        b = ["ChiSquared", nu]
        c = name+str(uuids)
        return (a,b,c)

    @staticmethod
    def Triangular(name, uuids, size):
        a = dist.Triangular(name+str(uuids), lower=size[0], c=size[1], upper=size[2])
        b = ["Triangular", size[0], size[1], size[2]]
        c = name+str(uuids)
        return(a,b,c)

    @staticmethod
    def Logistic(name, uuids, mu, s):
        a = dist.Logistic(name+str(uuids), mu=mu, s=s)
        b = ["Logistic", mu, s]
        c = name+str(uuids)
        return (a,b,c)
    
    @staticmethod
    def Analyze(*args):
        def inner(func):
            traces = []
            @settings(max_examples=1, deadline=None, phases=[Phase.generate],suppress_health_check=[HealthCheck.too_slow])
            @given(st.data())
            def helper(data):
                with pm.Model() as model:
                    N = 2 # Size of the database
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
                            dist, vals, name = parse(argument[0], islist=True,istuple=istuple)
                            return (dist,vals,name)
                        elif isinstance(argument, tuple):
                            dist, vals, name = zip(*[parse(arg, islist=islist,istuple=True) for arg in argument])
                            return (dist, vals, name)
                        elif isinstance(argument, int):
                            if islist:
                                dist, vals, name = zip(*[data.draw(i) for i in data.draw(ProbabilityGenerators.IntList("IntList", N))])
                                return (dist,vals,name)
                            else:
                                dist,vals,name = data.draw(ProbabilityGenerators.IntDist("IntDist"))
                                return (dist,vals,name)
                        elif isinstance(argument, float):
                            if islist:
                                dist, vals, name = zip(*[data.draw(i) for i in data.draw(ProbabilityGenerators.FloatList("FloatList", N))])
                                return (dist, vals, name)
                            else:
                                dist, vals, name = data.draw(ProbabilityGenerators.FloatGenerator("FloatDist", N))
                                return (dist,vals,name)
                        else:
                            raise Exception("Type is currently not supported")
                    dist, vals, name = zip(*[parse(x) for x in args])
                    # print(dist)
                    dist = zip(*dist[0])
                    for i,d in enumerate(dist):
                        # print(tuple([j[0] for j in d]))
                        x[i+1] = d
                    # print(parameters)
                    average = pm.Deterministic("average", func(x))

                    num_samples = 10
                    # # prior = pm.sample_prior_predictive(num_samples)
                    trace = pm.sample(num_samples, cores=1) 
                    traces.append(trace)                  
            helper()
            return (lambda x=traces:x)
        return inner
