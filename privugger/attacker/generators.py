"""
Probability distributions generators
"""
import pymc3 as pm
import pymc3.distributions as dist
from privugger.attacker.distributions import *
from hypothesis import strategies as st
import numpy as np
import scipy

def IntList(name, data, length=1, possible_dist=POSSIBLE_INTS, ranges=(0, np.inf)):
    """
    Generates a list of probabilistics distributions to mimic all possible int values
    
    Returns: Tuple[List[Pymc3.distributions], List[Tuple[String, ]]
    ----------------------------------------------------------------
    Returns a tuple containing a list of pymc3 distributions and a list containing information about each distributions
    
    Parameters:
    ------------
    name: String
        - The general name the list of distributions should take. In case of multiple distributions than names will have appended # within name.
    data: hypothesis.data
        - The hypothesis data used to draw the distributions
    length: Int
        - The length of the list
    possible_dist: List[Int]
        - A list of ints to be chosen from Privugger.distributions, indicating which distributions to choose from.
    """
    rand = data.draw(st.randoms(use_true_random=True))
    use_multiple_dist = rand.choice([0])
    if use_multiple_dist and len(possible_dist) < 0:
        dist, info = tuple(zip(*[IntGenerator(data=data, name=f"{name}{i}", shape=1) for i in range(length)]))
        return (dist, info)
    else:
        return IntGenerator(data=data, name=name, shape=length, ranges=ranges)

def FloatList(name, data, length=1, possible_dist=POSSIBLE_FLOATS, ranges=(-np.inf, np.inf)):
    """
    Generates a list of probabilistics distributions to mimic all possible float values
    
    Returns: Tuple[List[Pymc3.distributions], List[Tuple[String, ]]
    ----------------------------------------------------------------------
    Returns a tuple containing a list of pymc3 distributions and a list containing information about each distributions

    Parameters:
    -----------
    name: String
        - Name of the distributions. In case of multiple dist the names will also have appended the #
    data: hypothesis.data
        - The hypothesis data used to draw the distributions
    length: Int
        - The length of the list
    possible_dist: List[Int]
        - A list of ints to be chosen from privugger.distributions, indicating which distributions to choose from
    """
    rand = data.draw(st.randoms(use_true_random=True))
    use_same_shape = rand.choice([1])
    if use_same_shape:
        return FloatGenerator(name, data, possible_dist=[TRUNCATED_NORMAL], shape=length, ranges=ranges)
    else:
        dist, info = tuple(zip(*[FloatGenerator(name+str(i), data, possible_dist=possible_dist, ranges=ranges) for i in range(length)]))
        return (dist, info)


def IntGenerator(data, name, possible_dist = POSSIBLE_INTS, shape=1, ranges=(0, np.inf)):
    """
    A method for generating a single probabilistic distributions to mimic int distribution

    Returns: Tuple[Pymc3.distributions, Tuple[String, ]]
    ----------
        - Returns a tuple containint the distribution and a tuple with information about the distribution

    Parameters
    ----------
    data: hypothesis.data
        - The hypothesis data used to draw the distributions
    name: str
        - The name of the ditributions
    possible_dist: List[ints]
        - A list of the desired privugger.distributions to be chosen random from
    shape: Int
        - The dimensionality of the distribution
    """
    if ranges[0] < 0 or ranges[0] >= ranges[1]:
        raise ValueError("The ranges has to be greater than or equal to 0 and in increasing order. E.g. (0,100)")
    rand = data.draw(st.randoms(use_true_random=True))
    if ranges[1] > 1 and BERNOULLI in possible_dist:
        possible_dist.remove(BERNOULLI)
    dist = rand.choice(possible_dist)
    if dist == BINOMIAL:
        return Binomial(data=data, name=name, shape=shape, ranges=ranges)
    elif dist == BERNOULLI:
        return Bernoulli(data=data, name=name, shape=shape, ranges=ranges) 
    elif dist == GEOMETRIC:
        return Geometric(data=data, name=name, shape=shape, ranges=ranges)
    elif dist == BETA_BINOMIAL:
        return BetaBinomial(name=name, data=data, shape=shape, ranges=ranges)
    elif dist == POISSON:
        return Poisson(name=name, data=data, shape=shape, ranges=ranges)
    elif dist == DISCRETE_UNIFORM:
        return DiscreteUniform(name=name, data=data, shape=shape, ranges=ranges)
    else:
        raise ValueError("The possible distribution is not supported for Int Generators")

def FloatGenerator(name, data, possible_dist = POSSIBLE_FLOATS, shape=1, ranges=(-np.inf, np.inf)):
    """
    A method for generating a single distributions to represent float data

    Returns: Tuple[Pymc3.distributions, Tuple[String, ]]
    ----------
        - Returns a tuple containint the distribution and a tuple with information about the distribution

    Parameters
    ----------
    name: str
        - The name of the ditributions
    data: hypothesis.data
        - The hypothesis data used to draw the distributions
    possible_dist: List[ints]
        - A list of the desired privugger.distributions to be chosen random from
    shape: Int
        - The dimensionality of the distribution
    """
    rand = data.draw(st.randoms(use_true_random=True))
    dist = rand.choice(possible_dist)
    if dist == NORMAL:
        return Normal(data=data, name=name, shape=shape, ranges=ranges)
    elif dist == UNIFORM:
        return Uniform(data=data, name=name, shape=shape, ranges=ranges)
    elif dist == TRUNCATED_NORMAL:
        return TruncatedNormal(name=name, data=data, shape=shape)
    elif dist == BETA:
        return Beta(name=name,data=data, shape=shape)
    elif dist == EXPONENTIAL:
        return Exponential(name=name, data=data, shape=shape)
    elif dist == LAPLACE:
        return Laplace(name=name, data=data, shape=shape)
    elif dist == STUDENT_T:
        return StudentT(name=name, data=data, shape=shape)
    elif dist == CAUCHY:
        return Cauchy(name=name, data=data, shape=shape)
    elif dist == GAMMA:
        return Gamma(name=name, data=data, shape=shape)
    else:
        raise ValueError("The possible distribution is not supported for Int Generators")

# Int Distributions

def Binomial(data, name, shape=1, ranges=(1, np.inf)):
    """
    Constructs a binomial distributions with RV = X ~ Binomial(n,p)

    Returns: Tuple[Pymc3.distributions.Binomial, Tuple[String, int, float]]
    ----------
        - Returns a tuple with binomial distributions paired with the name, numper of test and probability
    
    Parameters:
    ----------
    name: str
        - The name of the ditributions
    data: hypothesis.data
        - The hypothesis data used to draw the distributions
    shape: int
        - The dimensionality of the distribution
    """
    l,h = ranges
    mean = lambda n,p: n*p
    ints = st.integers(min_value=l, max_value=10000)
    probability = st.floats(min_value=0.0010004043579101562, max_value=0.990234375, allow_infinity=False, allow_nan=False, width=16)
    n,p = data.draw(st.tuples(ints,probability).map(sorted).filter(lambda x: l <= mean(x[0],x[1]) <= h))
    if shape > 1:
        a = dist.Binomial(name=name, n=n, p=p, shape=shape)
    else:
        a = dist.Binomial(name=name, n=n, p=p)
    b = ["Binomial", n,p]
    return (a,b)


def Bernoulli(data, name, shape=1, ranges=(0,1)):
    """
    Constructs a bernoulli distributions with RV = X ~ Bernoulli(p)

    Returns: Tuple[Pymc3.distributions.Bernoulli, Tuple[String, float]]
    ----------
        - Returns a tuple with Bernoulli distributions paired with the name and probability
    
    Parameters:
    ----------
    name: str
        - The name of the ditributions
    data: hypothesis.data
        - The hypothesis data used to draw the distributions
    shape: int
        - The dimensionality of the distribution
    """
    probability = st.floats(min_value=0.0010004043579101562, max_value=0.990234375,allow_infinity=False, allow_nan=False,width=16)
    p = data.draw(probability)
    if shape > 1:
        a = dist.Bernoulli(name=name, p=p, shape=shape)
    else:
        a = dist.Bernoulli(name=name, p=p)
    b = ["Bernoulli", p]
    return (a,b)
    

def Geometric(data, name, shape=1, ranges=(1, np.inf)):
    """
    Constructs a geometric distributions with RV = X ~ Geometric(p)

    Returns: Tuple[Pymc3.distributions.Geometric, Tuple[String, float]]
    ----------
        - Returns a tuple with geometric distributions paired with the name and probability
    
    Parameters:
    ----------
    name: str
        - The name of the ditributions
    data: hypothesis.data
        - The hypothesis data used to draw the distributions
    shape: int
        - The dimensionality of the distribution
    """
    l,h = ranges
    mean = lambda p: 1/p
    probability = st.floats(min_value=0.0010004043579101562, max_value=0.990234375,allow_infinity=False, allow_nan=False,width=16).filter(lambda x: l <= mean(x) <= h)
    p = data.draw(probability)
    if shape > 1:
        a = dist.Geometric(name=name, p=p, shape=shape)
    else:
        a = dist.Geometric(name=name, p=p)
    b = ["Geometric", p]
    return (a,b)


def BetaBinomial(data, name, shape=1, ranges=(1, np.inf)):
    """
    Constructs a BetaBinomial distributions with RV = X ~ BetaBinomial(n, a, ß)

    Returns: Tuple[Pymc3.distributions.BetaBinomial, Tuple[String, int, float, float]]
    ----------
        - Returns a tuple with distributions paired with the [name, number, alpha, beta]
    
    Parameters:
    ----------
    name: str
        - The name of the ditributions
    data: hypothesis.data
        - The hypothesis data used to draw the distributions
    shape: int
        - The dimensionality of the distribution
    """
    l,h = ranges
    mean = lambda n,a,b: (n*a)/(a+b)
    ints = st.integers(min_value=l, max_value=10000)
    positive_float = st.floats(min_value=0.0999755859375, max_value=10000,allow_infinity=False, allow_nan=False,width=16)
    tuples = st.tuples(ints,positive_float,positive_float).map(sorted).filter(lambda x: l <= mean(x[0],x[1],x[2]) <= h)
    n = data.draw(ints)
    alpha = data.draw(positive_float)
    beta = data.draw(positive_float)
    if shape > 1:
        a = dist.BetaBinomial(name, alpha=alpha, beta=beta, n=n, shape=shape)
    else:
        a = dist.BetaBinomial(name, alpha=alpha, beta=beta, n=n)
    b = ["BetaBinomial", n, alpha, beta]
    return (a,b)


def Poisson(data, name, shape=1, ranges=(0, np.inf)):
    """
    Constructs a Poisson distributions with RV = X ~ Poisson(µ)

    Returns: Tuple[Pymc3.distributions.Poisson, Tuple[String, float]]
    ----------
        - Returns a tuple with distributions paired with the [name, mu]
    
    Parameters:
    ----------
    name: str
        - The name of the ditributions
    data: hypothesis.data
        - The hypothesis data used to draw the distributions
    shape: int
        - The dimensionality of the distribution
    """
    l,h = ranges
    non_negativ_float = st.floats(min_value=l, max_value=10000,allow_infinity=False, allow_nan=False,width=16).filter(lambda x: l <= x <= h)
    mu = data.draw(non_negativ_float)
    if shape > 1:
        a = dist.Poisson(name, mu=mu, shape=shape)
    else:
        a = dist.Poisson(name, mu=mu)
    b = ["Poisson", mu]
    return (a,b) 


def DiscreteUniform(data, name, ranges=(-np.inf, np.inf), shape=1):
    """
    Constructs a DiscreteUniform distributions with RV = X ~ DiscreteUniform(l,u)

    Returns: Tuple[Pymc3.distributions.DiscreteUniform, Tuple[String, int, int]]
    ----------
        - Returns a tuple with distributions paired with the [name, lower, upper]
    
    Parameters:
    ----------
    name: str
        - The name of the ditributions
    data: hypothesis.data
        - The hypothesis data used to draw the distributions
    shape: int
        - The dimensionality of the distribution
    """
    low, high = min(ranges), max(ranges)
    low = low if low != -np.inf else -1000
    high = high if high != np.inf else 1000
    values = st.integers(min_value=low, max_value=high)
    mean = lambda l,h: (h+l)/2
    size = (st.tuples(values, values)
                .map(sorted)
                .filter(lambda x: x[0] < x[1] and  low <= mean(x[0],x[1]) <= high))
    lower, upper = data.draw(size)
    a = dist.DiscreteUniform(name, lower, upper)
    b = ["DiscreteUniform", lower, upper]
    return (a,b)

# Float Distributions
def Normal(data, name, shape=1, ranges=(0, 100)):
    """
    Constructs a Normal distributions with RV = X ~ Normal(µ,sigma)

    Returns: Tuple[Pymc3.distributions.Normal, Tuple[String, float, float]]
    ----------
        - Returns a tuple with distributions paired with the [name, mu, sigma]
    
    Parameters:
    ----------
    name: str
        - The name of the ditributions
    data: hypothesis.data
        - The hypothesis data used to draw the distributions
    shape: int
        - The dimensionality of the distribution
    range: Tuple[int] 
        - The possible values that the PMF can mimic
    """
    low, high = ranges
    low = low if low != -np.inf else -1000
    high = high if high != np.inf else 1000

    floats = st.floats(allow_infinity=False, allow_nan=False, min_value=low, max_value=high,width=16)
    positive_floats = st.floats(min_value=0.0999755859375,allow_infinity=False, allow_nan=False,max_value=high-low,width=16)

    mean = lambda mu, sigma: mu
    values = st.tuples(floats, positive_floats).filter(lambda x: low <= mean(x[0], x[1]) <= high)
    mu, sigma = data.draw(values)
    a = dist.Normal(name=name, mu=mu, sigma=sigma, shape=shape)
    b = ["Normal", mu,sigma]
    return (a,b)


def Uniform(data, name, shape=1, ranges=(0,100)):
    """
    Constructs a Uniform distributions with RV = X ~ Uniform(l,u)

    Returns: Tuple[Pymc3.distributions.Uniform, Tuple[String, int, int]]
    ----------
        - Returns a tuple with distributions paired with the [name, lower, upper]
    
    Parameters:
    ----------
    name: str
        - The name of the ditributions
    data: hypothesis.data
        - The hypothesis data used to draw the distributions
    shape: int
        - The dimensionality of the distribution
    """
    low, high = ranges
    low = low if low != -np.inf else -1000
    high = high if high != np.inf else 1000
    floats = st.floats(allow_infinity=False, allow_nan=False, min_value=low, max_value=high,width=16)
    mean = lambda a,b: (a+b)/2
    size = (st.tuples(floats, floats)
                .map(sorted)
                .filter(lambda x: x[0] < x[1] and low <= mean(x[0],x[1]) <= high))
    lower, upper = data.draw(size)
    a = dist.Uniform(name, lower=lower, upper=upper, shape=shape)
    b = ["Uniform", lower, upper]
    return (a,b)


def TruncatedNormal(data, name, shape=1, ranges=(-np.inf, np.inf)):
    """
    Constructs a TruncatedNormal distributions with RV = X ~ TruncatedNormal(mu, sigma, lower, upper)

    Returns: Tuple[Pymc3.distributions.TruncatedNormal, Tuple[String, float, float, int, int]]
    ----------
        - Returns a tuple with distributions paired with the [name, mu, sigma, lower, upper]
    
    Parameters:
    ----------
    name: str
        - The name of the ditributions
    data: hypothesis.data
        - The hypothesis data used to draw the distributions
    shape: int
        - The dimensionality of the distribution
    """
    low, high = ranges
    low = low if low != -np.inf else -1000
    high = high if high != np.inf else 1000
    floats = st.floats(allow_infinity=False, allow_nan=False, min_value=low, max_value=high,width=16).filter(lambda x: low <= x <= high)
    positive_floats = st.floats(min_value=0.0999755859375,allow_infinity=False, allow_nan=False,max_value=high-low,width=16)
    size = (st.tuples(st.integers(min_value=low, max_value=high), st.integers(min_value=low, max_value=high))
                .map(sorted)
                .filter(lambda x: x[0] < x[1]))
    mu = data.draw(floats)
    sigma = data.draw(positive_floats)
    lower, upper = data.draw(size)
    a = dist.TruncatedNormal(name, mu=mu, sigma=sigma, lower=lower, upper=upper, shape=shape)
    b = ["TruncatedNormal", mu, sigma, lower, upper]
    return (a,b)


def Beta(data, name, shape=1):
    """
    Constructs a Beta distributions with RV = X ~ Beta(alpha, beta)

    Returns: Tuple[Pymc3.distributions.Beta, Tuple[String, float, float]]
    ----------
        - Returns a tuple with distributions paired with the [name, alpha, beta]
    
    Parameters:
    ----------
    name: str
        - The name of the ditributions
    data: hypothesis.data
        - The hypothesis data used to draw the distributions
    shape: int
        - The dimensionality of the distribution
    """
    positive_floats = st.floats(min_value=0.0999755859375,allow_infinity=False, allow_nan=False,max_value=40,width=16)
    alpha = data.draw(positive_floats)
    beta = data.draw(positive_floats)
    a = dist.Beta(name, alpha=alpha, beta=beta, shape=shape)
    b = ["Beta", alpha, beta]
    return (a,b)


def Exponential(data, name, shape=1, ranges=(-np.inf, np.inf)):
    """
    Constructs a Exponential distributions with RV = X ~ Exponential(lambda)

    Returns: Tuple[Pymc3.distributions.Exponential, Tuple[String, float]]
    ----------
        - Returns a tuple with distributions paired with the [name, lambda]
    
    Parameters:
    ----------
    name: str
        - The name of the ditributions
    data: hypothesis.data
        - The hypothesis data used to draw the distributions
    shape: int
        - The dimensionality of the distribution
    """
    low, high = ranges
    low = low if low != -np.inf else 0
    high = high if high != np.inf else 1000
        
    positive_floats = st.floats(min_value=0.0999755859375,allow_infinity=False, allow_nan=False,max_value=50,width=16).filter(lambda x: low <= 1/x <= high)
    lam = data.draw(positive_floats)
    a = dist.Exponential(name, lam, shape=shape)
    if not low:
        shift = pm.Deterministic(name+"_shifted", a+low)
        b = ["Exponential", lam]
        return (a,b)
    b = ["Exponential", lam]
    return (a,b)


def Laplace(data, name, shape=1, ranges=(-np.inf, np.inf)):
    """
    Constructs a Laplace distributions with RV = X ~ Laplace(mu, b)

    Returns: Tuple[Pymc3.distributions.Laplace, Tuple[String, float, float]]
    ----------
        - Returns a tuple with distributions paired with the [name, mu, b]
    
    Parameters:
    ----------
    name: str
        - The name of the ditributions
    data: hypothesis.data
        - The hypothesis data used to draw the distributions
    shape: int
        - The dimensionality of the distribution
    """
    low, high = ranges
    low = low if low != -np.inf else -1000
    high = high if high != np.inf else 1000
    floats = st.floats(allow_infinity=False, allow_nan=False, min_value=low, max_value=high,width=16).filter(lambda x: low <= x <= high)
    positive_floats = st.floats(min_value=0.0999755859375,allow_infinity=False, allow_nan=False,max_value=50,width=16)
    mu = data.draw(floats)
    bi = data.draw(positive_floats)
    a = dist.Laplace(name, mu=mu, b=bi, shape=shape)
    b = ["Laplace", mu, bi]
    return (a,b)


def StudentT(data, name, shape=1, ranges=(-np.inf, np.inf)):
    """
    Constructs a StudentT distributions with RV = X ~ StudentT(nu, mu, sigma)

    Returns: Tuple[Pymc3.distributions.StudentT, Tuple[String, float, float, float]]
    ----------
        - Returns a tuple with distributions paired with the [name, nu, mu, sigma]
    
    Parameters:
    ----------
    name: str
        - The name of the ditributions
    data: hypothesis.data
        - The hypothesis data used to draw the distributions
    shape: int
        - The dimensionality of the distribution
    """
    low, high = ranges
    low = low if low != -np.inf else -1000
    high = high if high != np.inf else 1000
    positive_floats = st.floats(min_value=0.0999755859375,allow_infinity=False, allow_nan=False,max_value=high-low,width=16)
    floats = st.floats(allow_infinity=False, allow_nan=False, min_value=low, max_value=high,width=16)
    nu = data.draw(positive_floats)
    mu = data.draw(floats)
    sigma = data.draw(positive_floats)
    a = dist.StudentT(name, nu=nu, mu=mu, sigma=sigma, shape=shape)
    b = ["StudentT", nu, mu, sigma]
    return (a,b)


def Cauchy(data, name, shape=1, ranges=(-np.inf, np.inf)):
    """
    Constructs a Cauchy distributions with RV = X ~ Cauchy(alpha, beta)

    Returns: Tuple[Pymc3.distributions.Cauchy, Tuple[String, float, float]]
    ----------
        - Returns a tuple with distributions paired with the [name, alpha, beta]
    
    Parameters:
    ----------
    name: str
        - The name of the ditributions
    data: hypothesis.data
        - The hypothesis data used to draw the distributions
    shape: int
        - The dimensionality of the distribution
    """
    low, high = ranges
    low = low if low != -np.inf else -1000
    high = high if high != np.inf else 1000
    positive_floats = st.floats(min_value=0.0999755859375,allow_infinity=False, allow_nan=False,max_value=10,width=16)
    floats = st.floats(allow_infinity=False, allow_nan=False, min_value=low, max_value=high,width=16)
    alpha = data.draw(floats)
    beta = data.draw(positive_floats)
    a = dist.Cauchy(name, alpha=alpha, beta=beta, shape=shape)
    b = ["Cauchy", alpha, beta]
    return (a,b)


def Gamma(data, name, shape=1):
    """
    Constructs a Gamma distributions with RV = X ~ Gamma(alpha, beta)

    Returns: Tuple[Pymc3.distributions.Gamma, Tuple[String, float, float]]
    ----------
        - Returns a tuple with distributions paired with the [name, alpha, beta]
    
    Parameters:
    ----------
    name: str
        - The name of the ditributions
    data: hypothesis.data
        - The hypothesis data used to draw the distributions
    shape: int
        - The dimensionality of the distribution
    """
    positive_floats = st.floats(min_value=0.0999755859375,allow_infinity=False, allow_nan=False,max_value=40,width=16)
    values = st.tuples(positive_floats, positive_floats).map(sorted).filter(lambda x: low <= (x[0]/x[1]) <= high)
    alpha, beta = data.draw(values)
    a = dist.Gamma(name, alpha, beta, shape=shape)
    b = ["Gamma", alpha, beta]
    return (a,b)