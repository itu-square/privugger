

import pymc3 as pm
from scipy import stats as st

"""
By specifying our own interface for distributions we could ideally hide which specific backend is used to model the distributions

"""

class Discrete():
    
    def pymc3_dist(self, name):
        return None

    def scipy_dist(self, name):
        return None


__all__ = [

    "Bernoulli",
    "Categorical",
    "Binomial",
    "DiscreteUniform",
    "Geometric",
    "Constant"
]

#NOTE the convention is that num_elements -1 means that it is not set 
class Bernoulli(Discrete):
    def __init__(self, p=0.5, num_elements=-1):
        self.p = p
        self.num_elements=num_elements
    
    def pymc3_dist(self, name):
        if(self.num_elements==-1):
            return pm.Bernoulli(name, p=self.p)
        else:
            return pm.Bernoulli(name, p=self.p, shape=self.num_elements)
    
    def scipy_dist(self, name):
        dist = (lambda siz : st.bernoulli(p=self.p).rvs(siz)) if self.num_elements == -1 else (lambda siz: st.bernoulli(p=self.p).rvs((self.num_elements, siz)))
        return name,dist



class Categorical(Discrete):
    
    def __init__(self, p=None, num_elements=-1):
        
        if (p==None):
            raise TypeError("please specify p")
        else:
            self.p=p

        self.num_elements=num_elements

    def pymc3_dist(self, name):
        if(self.num_elements==-1):
            return pm.Categorical(name, p=self.p)
        else:
            return pm.Categorical(name, p=self.p, shape=self.num_elements)

    def scipy_dist(self, name):
        theta = self.p
        dist = (lambda siz : st.rv_discrete(values=(range(len(theta)), theta)).rvs(siz)) if self.num_elements == -1 else (lambda siz: st.rv_discrete(values=(range(len(theta)), theta)).rvs((self.num_elements, siz)))
        return name, dist


class Binomial(Discrete):
    
    def __init__(self, n=2, p=0.5, num_elements=-1):
        self.n=n
        self.p=p
        self.num_elements=num_elements
    
    def pymc3_dist(self, name):
        if(self.num_elements==-1):
            return pm.Binomial(name, n=self.n, p=self.p)
        else:
            return pm.Binomial(name, n=self.n, p=self.p, shape=self.num_elements)

    def scipy_dist(self, name):
        dist = (lambda siz : st.binom(n=self.n, p=self.p).rvs(siz)) if self.num_elements == -1 else (lambda siz: st.binom(n=self.n, p=self.p).rvs((self.num_elements, siz)))
        return name, dist

class DiscreteUniform(Discrete):
    def __init__(self,lower=0, upper=1, num_elements=-1):
        self.lower = lower
        self.upper = upper
        self.num_elements=num_elements

    def pymc3_dist(self, name):
        if(self.num_elements==-1):
            return pm.DiscreteUniform(name, lower=self.lower, upper=self.upper)
        else:
            return pm.DiscreteUniform(name, lower=self.lower, upper=self.upper, shape=self.num_elements)

    def scipy_dist(self, name):
        dist = (lambda siz : st.randint(lower=self.lower, upper=self.upper).rvs(siz)) if self.num_elements == -1 else (lambda siz: st.randint(lower=self.lower, upper=self.upper).rvs((self.num_elements, siz)))
        return name, dist

class Geometric(Discrete):
    
    def __init__(self, p=0.5, num_elements=-1):
        self.p=p
        self.num_elements=num_elements

    def pymc3_dist(self, name):
        if(self.num_elements==-1):
            return pm.Geometric(name, p=self.p)
        else:
            return pm.Geometric(name, p=self.p, shape=self.num_elements)
        
    def scipy_dist(self, name):
        dist = (lambda siz : st.geom(self.p).rvs(siz)) if self.num_elements == -1 else (lambda siz: st.geom(self.p).rvs((self.num_elements, siz)))
        return name, dist


class Constant(Discrete):
    
    def __init__(self, val, num_elements=-1):
        self.val = val
        self.num_elements = num_elements

    def pymc3_dist(self, name):
        return pm.ConstantDist(name, self.val)
    
    def scipy_dist(self, name):
        return lambda siz: np.array([self.val for _ in range(siz)])













