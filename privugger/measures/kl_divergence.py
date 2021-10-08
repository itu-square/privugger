import numpy as np
import portion as p
from scipy.special import kl_div, entr, rel_entr


class Divergence:
    """ 
    Calculate the divergences between two probablility measures P and Q. 
    Using scipy.special functions 'entr', 'rel_entr' and 'kl_div' for discrete data.
    Using the estimation of the Radon-Nikodym derivative via a data-dependent partition as described in;
    https://doi-org.ep.ituproxy.kb.dk/10.1109/TIT.2005.853314
    """
    
    def __init__(self, data_class=None):
        self.data_class = data_class
    """
    Parameter
    ------------
    Data Class : String
        Data classification, 'discrete' or 'continuous'. Default Value is None.
    """
    
    if self.data_class == 'discrete':
        """
        Parameters
        -----------

        P, Q : Array_like
        Probability measures 

        Discrete 
        ---------------------

        entr(P) = -P*log(P) for P > 0, 0 for P = 0 and -∞ otherwise
        
        rel_entr(P, Q) = P*log(P/Q) for P>0 and Q>0, 0 for P = 0 and Q ≥ 0 and ∞  otherwise
        
        Kl_div(P, Q) = P*log(P/Q) - P + Q for P>0 and Q>0, 0 for P = 0 and Q ≥ 0 and ∞   otherwise
        
        """
        @property
        def entropy(self, P):
            return entr(P)    
        
        @property
        def rel_entropy(self, P, Q):
            return rel_entr(P, Q)
        
        @property
        def kl_divergence(self, P, Q):
            return  kl_div(P, Q) 

    elif self.data_class == 'continous' :

        def kl_divergence(self, P, Q, l_m):
            
            n = np.size(P)
            m = np.size(Q)
            
            T_n = n/l_m
            T_m = m/l_m
            
            P_I = p.iterate(p.openclosed(0, P[i]), step=1), p.iterate(p.openclosed(P[i], P[(2*i)]), step=, p.openclosed(P[(i*(T_m-1))], P[T_m])
            
            
            
        # Q_I = [np.arange(-np.inf, Q[l_m])]