import math
import torch as t
import numpy as np
import matplotlib 
import matplotlib.pyplot as plt
from torch.distributions import Normal
from torch.distributions.kl import kl_divergence, register_kl

def h_rt(X):

    n = X.size()[0]
        
    def iqr(x):
        q1 = t.quantile(x, q=t.tensor(0.25))
        q3 = t.quantile(x, q=t.tensor(0.75))
        return q3 - q1
        
    def quantile(p):
        p1 = t.tensor(((2*p)-1))
        return (t.sqrt(t.tensor(2)) * t.erfinv(p1)) 
            
    sigma_iqr = iqr(X) / (quantile(0.75) - quantile(0.25))
    std = t.std(X)
        
    sigma_hat = t.min(t.tensor([std, sigma_iqr]))

    return 1.06*(n**-(1/5)) * sigma_hat
    
    
def H(X, h):
    
    n = X.size()[0]
        
    pi = lambda:t.tensor(np.pi)

    phi = lambda x :t.reciprocal(t.sqrt(2*pi()))*t.exp(-1/2 * (x**2))

    def iqr(x):
        q1 = t.quantile(x, q=t.tensor(0.25))
        q3 = t.quantile(x, q=t.tensor(0.75))
        return q3 - q1

    a = 0.920*iqr(X)*t.float_power(t.tensor(n), -(1/7))
    b = 0.912*iqr(X)*t.float_power(t.tensor(n), -(1/9))

    H4 = lambda x: x**4 - 6*(x**2) + 3
    H6 = lambda x: x**6 - 15*(x**4) + 45*(x**2) - 15

    def Phi(x, r):
        if r == b:
            return phi(x) * H6(x)
        else:
            return phi(x) * H4(x)

    
    x_ij = (1/h) * t.stack([X[i] - t.stack([X[j] for j in range(0, n)]) for i in range(0, n)])
    
    SD_a = ((n*(n-1))**-1) * (a**-5) * t.sum(Phi(x_ij, a))
    TD_b = -((n*(n-1))**-1) * (b**-7) * t.sum(Phi(x_ij, b))

    
    alpha =  1.357 * (SD_a/TD_b)**(1/7) * (h**(5/7))
    
    
    R = (t.sqrt(pi())/4*pi()) * (t.erf(X[n-1]) - t.erf(X[0]))
    
    sigma = ((1/(2*pi())) * (t.sqrt((pi()/t.tensor(2)))*t.erf(X[n-1]/t.sqrt(t.tensor(2))) - (X[n-1] * t.exp(-(0.5*(X[n-1]**2)))))**2)  - ((1/(2*pi())) * (t.sqrt((pi()/t.tensor(2)))*t.erf(X[0]/t.sqrt(t.tensor(2))) - (X[0] * t.exp(-(0.5*(X[0]**2)))))**2)
    
    SD_alpha = ((n*(n-1))**-1) * (alpha**-5) * t.sum(Phi(x_ij, alpha))

    return t.abs((R/(sigma*SD_alpha)))

def f(X, h):
    x = t.tensor(h)
    
    y = H(X, x)**(1/5) * (X.size()[0])**(-1/5) - x
    
    return y

def f_prime(X, h):
    
    x = t.tensor(h, requires_grad=True)
    
    y = (H(X, x)**(1/5)) * (X.size()[0])**(-1/5) - x
    
    y.backward()
    
    return x.grad

from collections import deque


def newton(X):    
    
    h_n = deque([h_rt(X)])
    
    while True:
        
        h_n1 = h_n[0] - (f(X, h_n[0].detach().numpy())/f_prime(X, h_n[0].detach().numpy()))
        
        if h_n1 >= 0.0:
            
            h_n.append(h_n1)
            
            if math.isclose(h_n[0].numpy(), h_n[1].numpy()) == True:
                
                h_n.popleft()
                
                break
            
            else:
                h_n.popleft()
        
        else:
            break

    return h_n