import numpy as np
import matplotlib.pyplot as plt
import scipy as sc

def map_int_to_cont_dist(name, info, domain, shape,rng=None):
    """
    Maps an integer to a method which returns discrete distribution

    :param name: specifies the name of the distribution (Optional, left in to support extension to pymc3)
    :param info: An array specifing the information for the distribution where info[0] acts as an indicator for values
    :param domain: The domain constraints for the underlying distributions (and to manually set the secret)
    :param shape: How many of the same distribution to generate (# of n in thesis)
    :param rng: A randomstate seed to fix the random generator (can in some scenarios help the optimizer)
    :returns: A method of that when called return a distribution samples n times
    """
    dist = None
    ids = info[0]
    
    if ids == 0:
        #Normal

        mu, std = info[1:]
        dist = [lambda siz: sc.stats.norm(mu, std).rvs(siz,random_state=rng) for _ in range(shape)]
    elif ids == 1:
        #UNIFORM
        
        a,b = info[1:]
        dist = [lambda siz: sc.stats.uniform(a, b).rvs(siz,random_state=rng) for _ in range(shape)]
    elif ids == 2:
        #Half normal
        mu, std = info[1:]
        dist = [lambda siz: sc.stats.halfnorm(mu, std).rvs(siz,random_state=rng) for _ in range(shape)]

    #     dist = pm.Gamma(name, mu=abs(mu), sigma=std, shape=shape)
    if shape == 1:
        return dist
    db = np.empty(shape+1, dtype=object)
    if "alice" in domain and isinstance(domain["alice"], sc.stats._distn_infrastructure.rv_frozen):
        db[0] = lambda siz: domain["alice"].rvs(siz, random_state=rng)
    else:
        db[0] = lambda siz: sc.stats.uniform(domain["lower"], domain["upper"]-domain["lower"]).rvs(siz, random_state=rng)
    for i in range(shape):
        db[i+1] = dist[i]
    return db

def map_int_to_discrete_dist(name, info, domain, shape, rng=None):
    """
    Maps an integer to a method which returns continuous distribution

    :param name: specifies the name of the distribution (Optional, left in to support extension to pymc3)
    :param info: An array specifing the information for the distribution where info[0] acts as an indicator for values
    :param domain: The domain constraints for the underlying distributions (and to manually set the secret)
    :param shape: How many of the same distribution to generate (# of n in thesis)
    :param rng: A randomstate seed to fix the random generator (can in some scenarios help the optimizer)
    :returns: A method of that when called return a distribution samples n times
    """
    ids = info[0]
    dist = None
    if ids == 1:
        a,b = info[1:]
        dist = [lambda siz: sc.stats.randint(a, a+b).rvs(siz) for _ in range(shape)]
    elif ids == 0:
        mu,loc = info[1:]
        dist = [lambda siz: sc.stats.poisson(mu=mu,loc=loc).rvs(siz) for _ in range(shape)]

    if shape == 1:
        return dist
    db = np.empty(shape+1, dtype=object)
    if "alice" in domain and isinstance(domain["alice"], sc.stats._distn_infrastructure.rv_frozen):
        db[0] = lambda siz: domain["alice"].rvs(siz, random_state=rng)
    else:
        db[0] = lambda siz: sc.stats.randint(domain["lower"], domain["upper"]+1).rvs(siz)
    for i in range(shape):
        db[i+1] = dist[i]
    return db

