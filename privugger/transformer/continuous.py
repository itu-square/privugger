import pymc3 as pm
from scipy import stats as st
from abc import abstractmethod
"""
By specifying our own interface for distributions we could ideally hide which specific backend is used to model the distributions

"""

class Continuous():

    @abstractmethod
    def pymc3_dist(self, name, hypers):
        return None
	
    @abstractmethod
    def get_params(self):
    	return None
    	
    	
    @abstractmethod
    def scipy_dist(self, name):
        return None

__all__ = [
    "Uniform",
    "Normal",
    "Exponential",
    "Beta"

]


class Uniform(Continuous):
    
    def __init__(self,name, lower=0, upper=1, num_elements=-1, is_hyper_param=False):
        self.lower = lower
        self.name = name
        self.upper = upper
        self.num_elements = num_elements
        self.is_hyper_param = is_hyper_param
    
    def pymc3_dist(self, name, hypers):
        lower = self.lower
        upper = self.upper
        if(len(hypers) == 1):
                hyper_dist = hypers[0][0]
                hyper_name = hypers[0][1]
                idx = hypers[0][2]
                if(idx == 0):
                    lower = hyper_dist.pymc3_dist(hyper_name, [])
                else:
                    upper = hyper_dist.pymc3_dist(hyper_name, [])
        elif(len(hypers) == 2):
                hyper_dist_1 = hypers[0][0]
                hyper_name_1 = hypers[0][1]
                hyper_dist_2 = hypers[1][0]
                hyper_name_2 = hypers[1][1]

                lower = hyper_dist_1.pymc3_dist(hyper_name_1, [])
                upper = hyper_dist_2.pymc3_dist(hyper_name_2, [])
        if(self.num_elements==-1):
            return pm.Uniform(name, lower=lower, upper=upper)
        else:
            
            return pm.Uniform(name, lower=lower, upper=upper, shape=self.num_elements)
	
    def get_params(self):
    	return (self.lower, self.upper)
    
    def scipy_dist(self, name):
        dist = (lambda siz : st.uniform(self.lower, self.upper-self.lower).rvs(siz)) if self.num_elements == -1 else (lambda siz: st.uniform(self.lower, self.upper-self.lower).rvs((self.num_elements, siz)))
        return name,dist



class Normal(Continuous):

    def __init__(self,name, mu=0, std=1, num_elements=-1, is_hyper_param=False):
        self.mu = mu
        self.name = name
        self.std = std
        self.num_elements = num_elements
        self.is_hyper_param = is_hyper_param
   
    def pymc3_dist(self, name, hypers):
        mu = self.mu
        std = self.std
        if(len(hypers) == 1):
                hyper_dist = hypers[0][0]
                hyper_name = hypers[0][1]
                idx = hypers[0][2]
                if(idx == 0):
                    mu = hyper_dist.pymc3_dist(hyper_name, [])
                else:
                    std = hyper_dist.pymc3_dist(hyper_name, [])
        elif(len(hypers) == 2):
                hyper_dist_1 = hypers[0][0]
                hyper_name_1 = hypers[0][1]
                hyper_dist_2 = hypers[1][0]
                hyper_name_2 = hypers[1][1]
                mu = hyper_dist_1.pymc3_dist(hyper_name_1, [])
                std = hyper_dist_2.pymc3_dist(hyper_name_2, [])

                
        if(self.num_elements==-1):
            return pm.Normal(name, mu=mu, sigma=std)
        else:
            return pm.Normal(name, mu=mu, sigma=std, shape=self.num_elements)
    def get_params(self):
    	return (self.mu, self.std)

    def scipy_dist(self, name):
        dist = (lambda siz : st.norm(self.mu, self.std).rvs(siz)) if self.num_elements == -1 else (lambda siz: st.norm(self.mu, self.std).rvs((self.num_elements, siz)))
        return name,dist


class Exponential(Continuous):
    
    def __init__(self,name, lam=1, num_elements=-1, is_hyper_param=False):
        self.lam=lam
        self.name = name
        self.num_elements=num_elements
        self.is_hyper_param = is_hyper_param

    def pymc3_dist(self, name, hypers):
        if(self.num_elements==-1):
            return pm.Exponential(name, lam=self.lam)
        else:
            return pm.Exponential(name, lam=self.lam, shape=self.num_elements)

    def get_params(self):
    	return (self.lam)
    	
    def scipy_dist(self, name):
        dist = (lambda siz : st.expon(self.lam).rvs(siz)) if self.num_elements == -1 else (lambda siz: st.expon(self.lam).rvs((self.num_elements, siz)))
        return name,dist


class Beta(Continuous):
    
    def __init__(self, name, alpha=1, beta=1, num_elements=-1, is_hyper_param=False):
        self.alpha=alpha
        self.name = name
        self.beta=beta
        self.num_elements=num_elements
        self.is_hyper_param = is_hyper_param

    def pymc3_dist(self, name, hypers):
        if(self.num_elements==-1):
            return pm.Beta(name, alpha=self.alpha, beta=self.beta)
        else:
            return pm.Beta(name, alpha=self.alpha, beta=self.beta, shape=self.num_elements)
	
    def get_params(self):
    	return (self.alpha, self.beta)
    	
    def scipy_dist(self, name):
        dist = (lambda siz : st.beta(self.alpha,self.beta).rvs(siz)) if self.num_elements == -1 else (lambda siz: st.beta(self.alpha,self.beta).rvs((self.num_elements, siz)))
        return name,dist
