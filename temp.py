from scipy.stats import *
from scipy import special
import numpy as np
import matplotlib.pyplot as plt

def BetaBinom(a, b, n, x):
    pmf = special.binom(n, x) * (special.beta(x+a, n-x+b) / special.beta(a, b))
    return pmf

#Ints Value
BINOMIAL, BERNOULLI, GEOMETRIC, BETA_BINOMIAL, POISSON, DISCRETE_UNIFORM = range(6)

#Floats Value
NORMAL, UNIFORM, TRUNCATED_NORMAL = range(6,9)
BETA, EXPONENTIAL, LAPLACE, STUDENT_T = range(9, 13)
CAUCHY, GAMMA, LOG_NORMAL = range(13, 16)
CHI_SQUARED, TRIANGULAR, LOGISTIC = range(16,19)

_, ax = plt.subplots(2,1)

x = np.linspace(0,10,11)

#Discrete domain:
y1 = binom.pmf(x, 10, 0.5)
y2 = bernoulli.pmf(x, 0.5)
y3 = geom.pmf(x[1:], 0.5)
y4 = BetaBinom(0.5,0.5,10,x)
y5 = poisson.pmf(x, 5)
y6 = np.linspace(1/len(x), 1/len(x), len(x))

ax[0].plot(x,y1, "-o", label=r"X ~ Binomial($n$ = 10, $p$ = 0.5)")
ax[0].plot(x[1:2],y2[1:2], "-o", label=r"X ~ Bernoulli($p$ = 0.5)")
ax[0].plot(x[1:],y3, "-o", label=r"X ~ Geometric($p$ = 0.5)")
ax[0].plot(x,y4, "-o", label=r"X ~ BetaBinomial($\alpha$ = 2,$\alpha$ = 2,$n$ = 10)")
ax[0].plot(x,y5, "-o", label=r"X ~ Poisson($\mu$ = 5)")
ax[0].plot(x,y6, "-o", label=r"X ~ DiscreteUniform($l$ = 0,$h$ = 10)")
ax[0].legend()

#Continous domain
x = np.linspace(0,10,1000)
y7 = norm.pdf(x, 5, 0.5)
y8 = np.zeros(1000)
y8[(x<5) & (x>1)] = 1/4
y9 = np.zeros(1000)
y9[(x<5) & (x>3)] = norm.pdf(x,3,0.5)[300:500]
y10 = beta.pdf(x[0:100], 2,2)
y11 = expon.pdf(x, 1/2)
y12 = laplace.pdf(x, 5, 2)
y13 = t.pdf(x, 3, loc=5, scale=0.5)

ax[1].plot(x,y7, label=r"X ~ Normal($\mu$ = 5, $\sigma$ = 0.5)")
ax[1].plot(x,y8, label=r"X ~ Uniform(l = 1, h = 5)")
ax[1].plot(x,y9, label=r"X ~ TruncatedNormal($\mu$ = 5, $\sigma$ = 0.5, l = 3, h = 5)")
ax[1].plot(x[0:100], y10, label=r"X ~ Beta($\alpha$ = 2, $\beta$ = 2)")
ax[1].plot(x, y11, label=r"X ~ Exponential($\lambda$ = 2)")
ax[1].plot(x, y12, label=r"X ~ Laplace($\mu$ = 5, $b$ = 2)")
ax[1].plot(x,y13, label=r"X ~ StudentT($v$ = 3, $\mu$ = 5, $\sigma$ = 0.5")
ax[1].legend()

ax[0].set_title("Various supported distributions in the discrete domain")
ax[0].set_ylabel("p(x)")
ax[0].set_xlabel("x")
ax[1].set_title("Various supported distributions in the discrete domain")
ax[1].set_ylabel("f(x)")
ax[1].set_xlabel("x")
plt.tight_layout()
plt.show()

