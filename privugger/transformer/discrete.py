
import pymc3 as pm

"""
By specifying our own interface for distributions we could ideally hide which specific backend is used to model the distributions

"""

class Discrete():
    
    def pymc3_dist(self, name):
        return None



__all__ = [

    "Bernoulli",
    "Categorical",
    "Binomial",
    "DiscreteUniform",
    "Geometric"
]


class Bernoulli(Discrete):
    def __init__(self, p=0.5, num_elements=2):
        self.p = p
        self.num_elements=num_elements
    
    def pymc3_dist(self, name):
        return pm.Bernoulli(name, p=self.p, shape=self.num_elements)



class Categorical(Discrete):
    
    def __init__(self, p=None, num_elements=2):
        
        if (p==None):
            raise TypeError("please specify p")
        else:
            self.p=p

        self.num_elements=num_elements

    def pymc3_dist(self, name):
        return pm.Categorical(name, p=self.p, shape=self.num_elements)



class Binomial(Discrete):
    
    def __init__(self, n=2, p=0.5, num_elements=2):
        self.n=n
        self.p=p
        self.num_elements=num_elements
    
    def pymc3_dist(self, name):
        return pm.Binomial(name, n=self.n, p=self.p, shape=self.num_elements)



class DiscreteUniform(Discrete):
    def __init__(self,lower=0, upper=1, num_elements=2):
        self.lower = lower
        self.upper = upper
        self.num_elements=num_elements

    def pymc3_dist(self, name):
        return pm.DiscreteUniform(name, lower=self.lower, upper=self.upper, shape=self.num_elements)


class Geometric(Discrete):
    
    def __init__(self, p=0.5, num_elements=2):
        self.p=p
        self.num_elements=num_elements

    def pymc3_dist(self, name):
        pm.Geometric(name, p=self.p, shape=self.num_elements)


















