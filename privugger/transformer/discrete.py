import pymc3 as pm
import torch as t
import pyro as pr
from scipy import stats as st
from abc import abstractmethod
"""
By specifying our own interface for distributions we could ideally hide which specific backend is used to model the distributions

"""

class Discrete():

    """
    An abstract class that represents the discrete distributions that privugger supports
    """
    @abstractmethod
    def pymc3_dist(self, name):
        return None

    @abstractmethod
    def pyro(self, name):
        return None

    @abstractmethod
    def get_params(self):
        return None
    
    @abstractmethod
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

    def __init__(self,name, p=0.5, num_elements=-1, is_hyper_param=False):
        """
        Class for the Bernoulli distribution
        Parameters
        -----------
        name: String of the name of the random variable
        p: float value [0,1] giving the probability. Default: 0.5
        num_elements: int specifying number of RV's
        is_hyper_param: Boolean specifying if this RV is used as a hyper parameter. Default: False
        """

        self.p = p
        self.name =name
        self.num_elements=num_elements
        self.is_hyper_param = is_hyper_param
    
    def pymc3_dist(self, name, hypers):
        p = self.p
        if(len(hypers) == 1):
                hyper_dist = hypers[0][0]
                hyper_name = hypers[0][1]
                p = hyper_dist.pymc3_dist(hyper_name, [])

        if(self.num_elements==-1):
            return pm.Bernoulli(name, p=p)
        else:
            return pm.Bernoulli(name, p=p, shape=self.num_elements)
    
    def pyro(self, name):
        prob = t.tensor(self.p)
        sample_shape= t.Size([self.num_elements])
        dist = pr.distributions.Bernoulli(probs=prob).sample(sample_shape=sample_shape)
        return name, dist

    def get_params(self):
        return [self.p]
    
    def scipy_dist(self, name):
        dist = (lambda siz : st.bernoulli(p=self.p).rvs(siz)) if self.num_elements == -1 else (lambda siz: st.bernoulli(p=self.p).rvs((self.num_elements, siz)))
        return name,dist



class Categorical(Discrete):
    
    def __init__(self, name, p=None, num_elements=-1, is_hyper_param=False):
        """
        Class for the Categorical distribution
        Parameters
        -----------
        name: String of the name of the random variable
        p: Float list of probabilities
        num_elements: int specifying number of RV's
        is_hyper_param: Boolean specifying if this RV is used as a hyper parameter. Default: False
        """


        if (p==None):
            raise TypeError("please specify p")
        else:
            self.p=p

        self.num_elements=num_elements
        self.name = name
        self.is_hyper_param = is_hyper_param

    def pymc3_dist(self, name, hypers):
        p = self.p
        if(len(hypers) == 1):
            hyper_dist = hypers[0][0]
            hyper_name = hypers[0][1]
            p = hyper_dist.pymc3_dist(hyper_name, [])
            
        if(self.num_elements==-1):
            return pm.Categorical(name, p=p)
        else:
            return pm.Categorical(name, p=p, shape=self.num_elements)
    
    def pyro(self, name):
        prob= t.tensor(self.p)
        sample_shape= t.Size([self.num_elements])
        dist = pr.distributions.Categorical(probs=prob).sample(sample_shape=sample_shape)
        return name, dist

    def get_params(self):
        return [self.p]
    
    def scipy_dist(self, name):
        theta = self.p
        dist = (lambda siz : st.rv_discrete(values=(range(len(theta)), theta)).rvs(siz)) if self.num_elements == -1 else (lambda siz: st.rv_discrete(values=(range(len(theta)), theta)).rvs((self.num_elements, siz)))
        return name, dist


class Binomial(Discrete):
    
    def __init__(self, name, n=2, p=0.5, num_elements=-1, is_hyper_param=False):
        """
        Class for the Binomial distribution
        Parameters
        -----------
        name: String of the name of the random variable
        n: int specifying the number of trials. Default: 2
        p: float value [0,1] giving the probability. Default: 0.5
        num_elements: int specifying number of RV's
        is_hyper_param: Boolean specifying if this RV is used as a hyper parameter. Default: False
        """


        self.n=n
        self.name = name
        self.p=p
        self.num_elements=num_elements
        self.is_hyper_param = is_hyper_param
    
    def pymc3_dist(self, namei, hypers):
        n = self.n
        p = self.p

        if(len(hypers) == 1):
                hyper_dist = hypers[0][0]
                hyper_name = hypers[0][1]
                idx = hypers[0][2]
                if(idx == 0):
                    n = hyper_dist.pymc3_dist(hyper_name, [])
                else:
                    p = hyper_dist.pymc3_dist(hyper_name, [])
        elif(len(hypers) == 2):
                hyper_dist_1 = hypers[0][0]
                hyper_name_1 = hypers[0][1]
                hyper_dist_2 = hypers[1][0]
                hyper_name_2 = hypers[1][1]
                n = hyper_dist_1.pymc3_dist(hyper_name_1, [])
                p = hyper_dist_2.pymc3_dist(hyper_name_2, [])


        if(self.num_elements==-1):
            return pm.Binomial(name, n=n, p=p)
        else:
            return pm.Binomial(name, n=n, p=p, shape=self.num_elements)
    
    def pyro(self, name):
        n = t.tensor(self.n)
        prob= t.tensor(self.p)
        sample_shape= t.Size([self.num_elements])
        dist = pr.distributions.Binomial(total_count=n , probs=prob).sample(sample_shape=sample_shape)
        return name, dist

    def get_params(self):
        return [self.n, self.p]
    
    def scipy_dist(self, name):
        dist = (lambda siz : st.binom(n=self.n, p=self.p).rvs(siz)) if self.num_elements == -1 else (lambda siz: st.binom(n=self.n, p=self.p).rvs((self.num_elements, siz)))
        return name, dist

class DiscreteUniform(Discrete):
    def __init__(self, name, lower=0, upper=1, num_elements=-1, is_hyper_param=False):
        """
        Class for the Discrete Uniform distribution
        Parameters
        -----------
        name: String of the name of the random variable
        lower: int value giving the lower bound of the values. Default: 0
        upper: ine value giving the upper bound of the values. Default: 1
        num_elements: int specifying number of RV's
        is_hyper_param: Boolean specifying if this RV is used as a hyper parameter. Default: False
        """


        self.lower = lower
        self.upper = upper
        self.name = name
        self.num_elements=num_elements
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
            return pm.DiscreteUniform(name, lower=lower, upper=upper)
        else:
            return pm.DiscreteUniform(name, lower=lower, upper=upper, shape=self.num_elements)
    """
    class PyroDist(pr.distributions.TorchDistribution):
        arg_constraints = {"Lower": constraints.positive, "Upper": constraints.positive}
        support = constraints.real_vector
        
        def __init__(self, lower, upper, num_elements):
            self.lower, self.upper, self.num_elements = broadcast_all(lower, upper, num_elements)
            self.disu = t.randint(lower, upper, (num_elements, ))
        super().__init__(event_shape = (num_elements, ))

        def sample(self, sample_shape=()):
            u = self.mvn.sample(sample_shape)
            u0, u1 = u[..., 0], u[..., 1]
            a, b = self.a, self.b
            x = a * u0
            y = (u1 / a) + b * (u0 ** 2 + a ** 2)
            return torch.stack([x, y], -1)

    def get_params(self):
        return [self.lower, self.upper]

    def scipy_dist(self, name):
        dist = (lambda siz : st.randint(lower=self.lower, upper=self.upper).rvs(siz)) if self.num_elements == -1 else (lambda siz: st.randint(lower=self.lower, upper=self.upper).rvs((self.num_elements, siz)))
        return name, dist
    """

class Geometric(Discrete):
    
    def __init__(self, p=0.5, num_elements=-1, is_hyper_param=False):
        """
        Class for the Geometric distribution
        Parameters
        -----------
        name: String of the name of the random variable
        p: float value [0,1] giving the probability. Default: 0.5
        num_elements: int specifying number of RV's
        is_hyper_param: Boolean specifying if this RV is used as a hyper parameter. Default: False
        """


        self.p=p
        self.num_elements=num_elements
        self.is_hyper_param = is_hyper_param

    def pymc3_dist(self, name, hypers):
        p = self.p
        if(len(hypers) == 1):
            hyper_dist = hypers[0][0]
            hyper_name = hypers[0][1]
            p = hyper_dist.pymc3_dist(hyper_name, [])

        if(self.num_elements==-1):
            return pm.Geometric(name, p=p)
        else:
            return pm.Geometric(name, p=p, shape=self.num_elements)

    def pyro(self, name):
        prob= t.tensor(self.p)
        sample_shape= t.Size([self.num_elements])
        dist = pr.distributions.Geometric(probs=prob).sample(sample_shape=sample_shape)
        return name, dist
        
    def get_params(self):
        return [self.p]
    
    def scipy_dist(self, name):
        dist = (lambda siz : st.geom(self.p).rvs(siz)) if self.num_elements == -1 else (lambda siz: st.geom(self.p).rvs((self.num_elements, siz)))
        return name, dist


class Constant(Discrete):
    
    def __init__(self, name, val, num_elements=-1, is_hyper_param=False):
        """
        Class for the Constant distribution
        Parameters
        -----------
        name: String of the name of the random variable
        val: The constant value 
        num_elements: int specifying number of RV's
        is_hyper_param: Boolean specifying if this RV is used as a hyper parameter. Default: False
        """


        self.val = val
        self.name = name
        self.num_elements = num_elements
        self.is_hyper_param = is_hyper_param

    def pymc3_dist(self, name, hypers):
        val = self.val
        if(len(hypers) == 1):
            hyper_dist = hypers[0][0]
            hyper_name = hypers[0][1]
            val = hyper_dist.pymc3_dist(hyper_name, [])
        if(self.num_elements==-1):
            return pm.ConstantDist(name, self.val)
        else:
            return pm.ConstantDist(name, self.val, shape=self.num_elements)

    def get_params(self):
        return [self.val]
    
    def scipy_dist(self, name):
        return lambda siz: np.array([self.val for _ in range(siz)])













