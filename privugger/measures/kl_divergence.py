import numpy as np
from scipy.special import kl_div, entr, rel_entr

__all__ = [
    "Entropies",
    "InferenceData"
]
class InferenceData:
    def __init__(self, trace, priors=[]):
        self.trace = trace
        self.priors = priors
    
    def X(self):
        p = self.priors
        x = [self.trace['posterior'][f'{p}'].values for str in p]
        return x
    
    def Y(self):
        y = self.trace['posterior']['output'].values
        return  

class Entropies(InferenceData):
    def __init__(self, trace, priors):
        super().__init__(trace, priors)
        self.x = super().X()
        self.y = super().Y()
        
    def entropy(self):
        return entr(self.x)
            
    def relative_entropy(self):
        return rel_entr(self.x, self.y)
            
    def kl_divergence(self):
        return kl_div(self.x, self.y)
