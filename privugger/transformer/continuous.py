import pymc3 as pm

"""
By specifying our own interface for distributions we could ideally hide which specific backend is used to model the distributions

"""

class Continuous():
    
    def pymc3_dist(self, name):
        return None


__all__ = [
    "Uniform",
    "Normal",
    "Exponential",
    "Beta"

]


class Uniform(Continuous):
    
    def __init__(self,lower=0, upper=1, num_elements=-1):
        self.lower = lower
        self.upper = upper
        self.num_elements = num_elements
    
    def pymc3_dist(self, name):
        if(self.num_elements==-1):
            return pm.Uniform(name, lower=self.lower, upper=self.upper)
        else:
            return pm.Uniform(name, lower=self.lower, upper=self.upper, shape=self.num_elements)



class Normal(Continuous):

    def __init__(self, mu=0, std=1, num_elements=-1):
        self.mu = mu
        self.std = std
        self.num_elements = num_elements
   
    def pymc3_dist(self, name):
        if(self.num_elements==-1):
            return pm.Normal(name, mu=self.mu, sigma=self.std)
        else:
            return pm.Normal(name, mu=self.mu, sigma=self.std, shape=self.num_elements)
    


class Exponential(Continuous):
    
    def __init__(self, lam=1, num_elements=-1):
        self.lam=lam
        self.num_elements=num_elements

    def pymc3_dist(self, name):
        if(self.num_elements==-1):
            return pm.Exponential(name, lam=self.lam)
        else:
            return pm.Exponential(name, lam=self.lam, shape=self.num_elements)


class Beta(Continuous):
    
    def __init__(self, alpha=1, beta=1, num_elements=-1):
        self.alpha=alpha
        self.beta=beta
        self.num_elements=num_elements

    def pymc3_dist(self, name):
        if(self.num_elements==-1):
            return pm.Beta(name, alpha=self.alpha, beta=self.beta)
        else:
            return pm.Beta(name, alpha=self.alpha, beta=self.beta, shape=self.num_elements)
