import numpy as np
import pandas as pd
import portion as p
from scipy.special import kl_div, entr, rel_entr

    
""" 
    Parameters
    ------------
    P = Array of samples  array or list
    Q = Array of samples, array or list

    l_m = Partion size, integer
    dim = Dimension of sample, integer
    step = Step size, integer or float

"""
 
 
 

def discrete(P, Q):
    H = lambda P: entr(P)
    H_cross = lambda P, Q: rel_entr(P, Q)
    kl_divergence = lambda P, Q: kl_div(P, Q)

    return  pd.Series({'Entropy':sum(H(P)), 'Cross Entropy': sum(H_cross(P,Q)), 'KL divergence':sum(kl_divergence(P,Q))})

  
  
def continuous(P, Q, l_m, dim=1, step=0.01):
        
    m = np.size(P)
    n = np.size(Q)
        
    t_m = int(np.power((m/l_m),(1/dim)))

    dt = l_m*(t_m-1)

    def I_mid(X):
            
        i = l_m
        j = 2*l_m
            
        I = list(p.iterate(p.openclosed(X[i], X[j]), step=step))
            
        while  j == dt is False:
            i = 2*i
            j = 2*j
            I = I.append(list(p.iterate(p.openclosed(X[i], X[j]), step=step)))
        else:
            pass
        return I
        
    I_start = lambda X: list(p.iterate(p.openclosed(min(X), X[l_m]), step=step))

    I_end = lambda X: list(p.iterate(p.open(X[dt], max(X)), step=step))
        
    I = lambda X:[*I_start(X),*I_mid(X),*I_end(X)]

    P_n = I(np.sort(P, axis=None))
    Q_m = I(np.sort(Q, axis=None))

    D_kl = sum([P_n[i]*np.log(P_n[i]/Q_m[i]) for i in range(0, np.size(P_n))])

    return pd.Series({'KL Divergence': D_kl})
       

