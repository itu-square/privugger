"""
Probability distributions generators
"""

import pymc3.distributions as dist
from privugger.distributions import *
from hypothesis import strategies as st

def IntList(name, data, length=1, possible_dist=POSSIBLE_INTS):
    """
    Generates a list of probabilistics distributions to mimic all possible int values
    
    Returns: Tuple[List[Pymc3.distributions], List[Tuple[String, ]]
    ----------
    Returns a tuple containing a list of pymc3 distributions and a list containing information about each distributions

    Parameters:
    ----------
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
    use_multiple_dist = bool(rand.getrandbits(1))
    if use_multiple_dist and len(possible_dist) > 1:
        dist, info = tuple(zip(*[IntGenerator(data=data, name=f"{name}{i}", shape=1) for i in range(length)]))
        return (dist, info)
    else:
        return IntGenerator(data=data, name=name, shape=length)

def FloatList(name, data, length=1, possible_dist=POSSIBLE_FLOATS):
    """
    Generates a list of probabilistics distributions to mimic all possible float values
    
    Returns: Tuple[List[Pymc3.distributions], List[Tuple[String, ]]
    ----------
    Returns a tuple containing a list of pymc3 distributions and a list containing information about each distributions

    Parameters:
    ----------
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
    use_same_shape = rand.choice([0,1])
    if use_same_shape:
        return FloatGenerator(name, data, shape=length)
    else:
        dist, info = tuple(zip(*[FloatGenerator(name+str(i), data, possible_dist=possible_dist) for i in range(length)]))
        return (dist, info)


def IntGenerator(data, name, possible_dist = POSSIBLE_INTS, shape=1):
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
    rand = data.draw(st.randoms(use_true_random=True))
    dist = rand.choice(possible_dist)
    if dist == BINOMIAL:
        return Binomial(data=data, name=name, shape=shape)
    elif dist == BERNOULLI:
        return Bernoulli(data=data, name=name, shape=shape) 
    elif dist == GEOMETRIC:
        return Geometric(data=data, name=name, shape=shape)
    elif dist == BETA_BINOMIAL:
        return BetaBinomial(name=name, data=data, shape=shape)
    elif dist == POISSON:
        return Poisson(name=name, data=data, shape=shape)
    else:
        return DiscreteUniform(name=name, data=data, shape=shape)

def FloatGenerator(name, data, possible_dist = POSSIBLE_FLOATS, shape=1):
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
        return Normal(data=data, name=name, shape=shape)
    elif dist == UNIFORM:
        return Uniform(data=data, name=name, shape=shape)
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
    elif dist == LOG_NORMAL:
        return LogNormal(name=name, data=data, shape=shape)
    elif dist == CHI_SQUARED:
        return ChiSquared(name=name, data=data, shape=shape)
    elif dist == TRIANGULAR:
        return Triangular(name=name, data=data, shape=shape)
    else:
        return Logistic(name=name, data=data, shape=shape)

# Int Distributions

def Binomial(data, name, shape=1):
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
    ints = st.integers(min_value=1, max_value=10000)
    probability = st.floats(min_value=0.001, max_value=0.9999,allow_infinity=False, allow_nan=False)
    n = data.draw(ints)
    p = data.draw(probability)
    a = dist.Binomial(name=name, n=n, p=p, shape=shape)
    b = ["Binomial", n,p]
    return (a,b)


def Bernoulli(data, name, shape=1):
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
    probability = st.floats(min_value=0.001, max_value=0.9999,allow_infinity=False, allow_nan=False)
    p = data.draw(probability)
    a = dist.Bernoulli(name=name, p=p, shape=shape)
    b = ["Bernoulli", p]
    return (a,b)
    

def Geometric(data, name, shape=1):
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
    probability = st.floats(min_value=0.001, max_value=0.9999,allow_infinity=False, allow_nan=False)
    p = data.draw(probability)
    a = dist.Geometric(name=name, p=p, shape=shape)
    b = ["Geometric", p]
    return (a,b)


def BetaBinomial(data, name, shape=1):
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
    ints = st.integers(min_value=1, max_value=10000)
    positive_float = st.floats(min_value=0.001, max_value=10000,allow_infinity=False, allow_nan=False)
    n = data.draw(ints)
    alpha = data.draw(positive_float)
    beta = data.draw(positive_float)
    a = dist.BetaBinomial(name, alpha=alpha, beta=beta, n=n, shape=shape)
    b = ["BetaBinomial", n, alpha, beta]
    return (a,b)


def Poisson(data, name, shape=1):
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
    non_negativ_float = st.floats(min_value=0, max_value=10000,allow_infinity=False, allow_nan=False)
    mu = data.draw(non_negativ_float)
    a = dist.Poisson(name, mu=mu, shape=shape)
    b = ["Poisson", mu]
    return (a,b) 


def DiscreteUniform(data, name, shape=1):
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
    size = (st.tuples(st.integers(min_value=-10000, max_value=10000), st.integers(min_value=-10000, max_value=10000))
                .map(sorted)
                .filter(lambda x: x[0] < x[1]))
    lower, upper = data.draw(size)
    a = dist.DiscreteUniform(name, lower, upper, shape=shape)
    b = ["DiscreteUniform", lower, upper]
    return (a,b)

# Float Distributions


def Normal(data, name, shape=1):
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
    """
    floats = st.floats(allow_infinity=False, allow_nan=False, min_value=-100, max_value=200)
    positive_floats = st.floats(min_value=0.1,allow_infinity=False, allow_nan=False,max_value=100)
    mu = data.draw(floats)
    sigma = data.draw(positive_floats)
    a = dist.Normal(name=name, mu=mu, sigma=sigma, shape=shape)
    b = ["Normal", mu,sigma]
    return (a,b)


def Uniform(data, name, shape=1):
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
    size = (st.tuples(st.integers(min_value=-100, max_value=200), st.integers(min_value=-100, max_value=200))
                .map(sorted)
                .filter(lambda x: x[0] < x[1]))
    lower, upper = data.draw(size)
    a = dist.Uniform(name, lower=lower, upper=upper, shape=shape)
    b = ["Uniform", lower, upper]
    return (a,b)


def TruncatedNormal(data, name, shape=1):
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
    floats = st.floats(allow_infinity=False, allow_nan=False, min_value=-100, max_value=200)
    positive_floats = st.floats(min_value=0.1,allow_infinity=False, allow_nan=False,max_value=200)
    size = (st.tuples(st.integers(min_value=-100, max_value=200), st.integers(min_value=-100, max_value=200))
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
    positive_floats = st.floats(min_value=0.1,allow_infinity=False, allow_nan=False,max_value=40)
    alpha = data.draw(positive_floats)
    beta = data.draw(positive_floats)
    a = dist.Beta(name, alpha=alpha, beta=beta, shape=shape)
    b = ["Beta", alpha, beta]
    return (a,b)


def Exponential(data, name, shape=1):
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
    positive_floats = st.floats(min_value=0.1,allow_infinity=False, allow_nan=False,max_value=50)
    lam = data.draw(positive_floats)
    a = dist.Exponential(name, lam, shape=shape)
    b = ["Exponential", lam]
    return (a,b)


def Laplace(data, name, shape=1):
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
    floats = st.floats(allow_infinity=False, allow_nan=False, min_value=-100, max_value=200)
    positive_floats = st.floats(min_value=0.1,allow_infinity=False, allow_nan=False,max_value=50)
    mu = data.draw(floats)
    bi = data.draw(positive_floats)
    a = dist.Laplace(name, mu=mu, b=bi, shape=shape)
    b = ["Laplace", mu, bi]
    return (a,b)


def StudentT(data, name, shape=1):
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
    positive_floats = st.floats(min_value=0.1,allow_infinity=False, allow_nan=False,max_value=40)
    floats = st.floats(allow_infinity=False, allow_nan=False, min_value=-100, max_value=200)
    nu = data.draw(positive_floats)
    mu = data.draw(floats)
    sigma = data.draw(positive_floats)
    a = dist.StudentT(name, nu=nu, mu=mu, sigma=sigma, shape=shape)
    b = ["StudentT", nu, mu, sigma]
    return (a,b)


def Cauchy(data, name, shape=1):
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
    positive_floats = st.floats(min_value=0.1,allow_infinity=False, allow_nan=False,max_value=40)
    floats = st.floats(allow_infinity=False, allow_nan=False, min_value=-100, max_value=200)
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
    positive_floats = st.floats(min_value=0.1,allow_infinity=False, allow_nan=False,max_value=40)
    alpha = data.draw(positive_floats)
    beta = data.draw(positive_floats)
    a = dist.Gamma(name, alpha, beta, shape=shape)
    b = ["Gamma", alpha, beta]
    return (a,b)


def LogNormal(data, name, shape=1):
    """
    Constructs a LogNormal distributions with RV = X ~ LogNormal(mu, sigma)

    Returns: Tuple[Pymc3.distributions.LogNormal, Tuple[String, float, float]]
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
    """
    positive_floats = st.floats(min_value=0.1,allow_infinity=False, allow_nan=False,max_value=40)
    mu = data.draw(positive_floats)
    sigma = data.draw(positive_floats)
    a = dist.Lognormal(name, mu=mu, sigma=sigma, shape=shape)
    b = ["LogNormal", mu, sigma]
    return (a,b)


def ChiSquared(data, name, shape=1):
    """
    Constructs a ChiSquared distributions with RV = X ~ ChiSquared(nu)

    Returns: Tuple[Pymc3.distributions.ChiSquared, Tuple[String, int]]
    ----------
        - Returns a tuple with distributions paired with the [name, nu]
    
    Parameters:
    ----------
    name: str
        - The name of the ditributions
    data: hypothesis.data
        - The hypothesis data used to draw the distributions
    shape: int
        - The dimensionality of the distribution
    """
    positive_int = st.integers(min_value = 1, max_value=40)
    nu = data.draw(positive_int)
    a = dist.ChiSquared(name, nu, shape=shape)
    b = ["ChiSquared", nu]
    return (a,b)


def Triangular(data, name, shape=1):
    """
    Constructs a Triangular distributions with RV = X ~ Triangular(lower, middle, upper)

    Returns: Tuple[Pymc3.distributions.Triangular, Tuple[String, float, float, float]]
    ----------
        - Returns a tuple with distributions paired with the [name, lower, middle, upper]
    
    Parameters:
    ----------
    name: str
        - The name of the ditributions
    data: hypothesis.data
        - The hypothesis data used to draw the distributions
    shape: int
        - The dimensionality of the distribution
    """
    floats = st.floats(allow_infinity=False, allow_nan=False, min_value=-100, max_value=200)
    float_size = (st.tuples(floats, floats,floats)).map(sorted).filter(lambda x: x[0] < x[1] < x[2])
    lower, middle, upper = data.draw(float_size)
    a = dist.Triangular(name, lower=lower, c=middle, upper=upper, shape=shape)
    b = ["Triangular", lower, middle, upper]
    return(a,b)


def Logistic(data, name, shape=1):
    """
    Constructs a Logistic distributions with RV = X ~ Logistic(mu, s)

    Returns: Tuple[Pymc3.distributions.Logistic, Tuple[String, float, float]]
    ----------
        - Returns a tuple with distributions paired with the [name, mu, s]
    
    Parameters:
    ----------
    name: str
        - The name of the ditributions
    data: hypothesis.data
        - The hypothesis data used to draw the distributions
    shape: int
        - The dimensionality of the distribution
    """
    floats = st.floats(allow_infinity=False, allow_nan=False, min_value=-100, max_value=200)
    positive_floats = st.floats(min_value=0.1,allow_infinity=False, allow_nan=False,max_value=40)
    mu = data.draw(floats)
    s = data.draw(positive_floats)
    a = dist.Logistic(name, mu=mu, s=s, shape=shape)
    b = ["Logistic", mu, s]
    return (a,b)