import pymc3 as pm
from scipy import stats as st

"""
By specifying our own interface for distributions we could ideally hide which specific backend is used to model the distributions

"""

class Continuous():
    
    def pymc3_dist(self, name):
        return None

    def scipy_dist(self, name):
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

    def scipy_dist(self, name):
        dist = (lambda siz : st.uniform(self.lower, self.upper-self.lower).rvs(siz)) if self.num_elements == -1 else (lambda siz: st.uniform(self.lower, self.upper-self.lower).rvs((self.num_elements, siz)))
        return name,dist



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

    def scipy_dist(self, name):
        dist = (lambda siz : st.norm(self.mu, self.std).rvs(siz)) if self.num_elements == -1 else (lambda siz: st.norm(self.mu, self.std).rvs((self.num_elements, siz)))
        return name,dist


class Exponential(Continuous):
    
    def __init__(self, lam=1, num_elements=-1):
        self.lam=lam
        self.num_elements=num_elements

    def pymc3_dist(self, name):
        if(self.num_elements==-1):
            return pm.Exponential(name, lam=self.lam)
        else:
            return pm.Exponential(name, lam=self.lam, shape=self.num_elements)

    def scipy_dist(self, name):
        dist = (lambda siz : st.expon(self.lam).rvs(siz)) if self.num_elements == -1 else (lambda siz: st.expon(self.lam).rvs((self.num_elements, siz)))
        return name,dist


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

    def scipy_dist(self, name):
        dist = (lambda siz : st.beta(self.alpha,self.beta).rvs(siz)) if self.num_elements == -1 else (lambda siz: st.beta(self.alpha,self.beta).rvs((self.num_elements, siz)))
        return name,dist