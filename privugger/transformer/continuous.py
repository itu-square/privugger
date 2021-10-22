import pymc3 as pm
import torch as t
from scipy import stats as st
from abc import abstractmethod
"""
By specifying our own interface for distributions we could ideally hide which specific backend is used to model the distributions

"""

class Continuous():

    """
    An abstract class representing all the continuous distributions that privugger supports
    """
    @abstractmethod
    def pymc3_dist(self, name, hypers):
        return None
	
    @abstractmethod
    def pytorch(self, name):
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

        """
        Class for the Uniform distribution 
        Parameters
        -----------
        name: String of the name of the random variable
        lower: int for the lower bound. Default: 0
        upper: int for the upper bound. Default: 1
        num_elements: int specifying number of RV's
        is_hyper_param: Boolean specifying if this RV is used as a hyper parameter. Default: False
        """
        
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
    	return [self.lower, self.upper]

    def uniform(self, name):
        low = t.torch(self.lower)
        high = t.torch(self.upper)
        sample_shape= t.Size([self.num_elements])
        dist = t.distribution.uniform.Uniform(low, high).sample(sample_shape=sample_shape)
        return name, dist
    
    def scipy_dist(self, name):
        dist = (lambda siz : st.uniform(self.lower, self.upper-self.lower).rvs(siz)) if self.num_elements == -1 else (lambda siz: st.uniform(self.lower, self.upper-self.lower).rvs((self.num_elements, siz)))
        return name,dist



class Normal(Continuous):

    def __init__(self,name, mu=0, std=1, num_elements=-1, is_hyper_param=False):

        """
        Class for the Gaussian distribution 
        Parameters
        -----------
        name: String of the name of the random variable
        mu: value for the mean of the distribution. Default: 0
        std: value for the standard deviation. Default: 1
        num_elements: int specifying number of RV's
        is_hyper_param: Boolean specifying if this RV is used as a hyper parameter. Default: False
        """
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


    def pytorch(self, name):
        mu = t.torch(self.mu)
        std = t.torch(self.std)
        sample_shape= t.Size([self.num_elements])
        dist = t.distribution.normal.Normal(mu, std).sample(sample_shape=sample_shape)
        return name, dist

    def get_params(self):
    	return [self.mu, self.std]

    def scipy_dist(self, name):
        dist = (lambda siz : st.norm(self.mu, self.std).rvs(siz)) if self.num_elements == -1 else (lambda siz: st.norm(self.mu, self.std).rvs((self.num_elements, siz)))
        return name,dist


class Exponential(Continuous):
    
    def __init__(self,name, lam=1, num_elements=-1, is_hyper_param=False):
        """
        Class for the Exponential distribution 
        Parameters
        -----------
        name: String of the name of the random variable
        lam: value for the lambda parameter. Default: 1
        num_elements: int specifying number of RV's
        is_hyper_param: Boolean specifying if this RV is used as a hyper parameter. Default: False
        """
        self.lam=lam
        self.name = name
        self.num_elements=num_elements
        self.is_hyper_param = is_hyper_param

    def pymc3_dist(self, name, hypers):
        lam = self.lam
        if(len(hypers) == 1):
            hyper_dist = hypers[0][0]
            hyper_name = hypers[0][1]
            lam = hyper_dist.pymc3_dist(hyper_name, [])
        if(self.num_elements==-1):
            return pm.Exponential(name, lam=lam)
        else:
            return pm.Exponential(name, lam=lam, shape=self.num_elements)
    
    def pytorch(self, name):
        lam = t.torch(self.lam)
        sample_shape= t.Size([self.num_elements])
        dist = t.distribution.exponential.Exponential(lam).sample(sample_shape=sample_shape)
        return name, dist

    def get_params(self):
    	return [self.lam]
    	
    def scipy_dist(self, name):
        dist = (lambda siz : st.expon(self.lam).rvs(siz)) if self.num_elements == -1 else (lambda siz: st.expon(self.lam).rvs((self.num_elements, siz)))
        return name,dist


class Beta(Continuous):
    
    def __init__(self, name, alpha=1, beta=1, num_elements=-1, is_hyper_param=False):
        """
        Class for the Beta distribution 
        Parameters
        -----------
        name: String of the name of the random variable
        alpha: value for the alpha parameter. Default: 1
        beta: value for the beta parameter. Default: 1
        num_elements: int specifying number of RV's
        is_hyper_param: Boolean specifying if this RV is used as a hyper parameter. Default: False
        """

        self.alpha = alpha
        self.name  = name
        self.beta  = beta
        self.num_elements=num_elements
        self.is_hyper_param = is_hyper_param

    def pymc3_dist(self, name, hypers):
        alpha = self.alpha
        beta  = self.beta
        if(len(hypers) == 1):
                hyper_dist = hypers[0][0]
                hyper_name = hypers[0][1]
                idx = hypers[0][2]
                if(idx == 0):
                    alpha = hyper_dist.pymc3_dist(hyper_name, [])
                else:
                    beta = hyper_dist.pymc3_dist(hyper_name, [])
        elif(len(hypers) == 2):
                hyper_dist_1 = hypers[0][0]
                hyper_name_1 = hypers[0][1]
                hyper_dist_2 = hypers[1][0]
                hyper_name_2 = hypers[1][1]
                alpha = hyper_dist_1.pymc3_dist(hyper_name_1, [])
                beta = hyper_dist_2.pymc3_dist(hyper_name_2, [])


        if(self.num_elements==-1):
            return pm.Beta(name, alpha=self.alpha, beta=self.beta)
        else:
            return pm.Beta(name, alpha=self.alpha, beta=self.beta, shape=self.num_elements)
    
    def pytorch(self, name):
        alpha = t.torch(self.alpha)
        beta = t.torch(self.beta)
        sample_shape= t.Size([self.num_elements])
        dist = t.distribution.beta.Beta(alpha, beta).sample(sample_shape=sample_shape)
        return name, dist
	
    def get_params(self):
    	return [self.alpha, self.beta]
    	
    def scipy_dist(self, name):
        dist = (lambda siz : st.beta(self.alpha,self.beta).rvs(siz)) if self.num_elements == -1 else (lambda siz: st.beta(self.alpha,self.beta).rvs((self.num_elements, siz)))
        return name,dist
