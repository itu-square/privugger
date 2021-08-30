import pymc3 as pm
import arviz as az
from matplotlib import pyplot as plt


# Hard condition > 50
with pm.Model() as model:
    age = pm.Normal("age", 50,10, testval=60.)
    t = pm.Deterministic("output", 1.*(age >= 50.))
    cond = pm.Bernoulli("output_cond", t, observed=1)

    trace = pm.sample(500, tune=1000, cores=1, chains=10)
az.plot_posterior(trace)
plt.show()

# Soft condition
with pm.Model() as model:
    age = pm.Normal("age", 50,10, testval=60.)
    t = pm.Normal("output", age >= 50., 0.1)

    trace = pm.sample(500, tune=1000, cores=1, chains=10)
az.plot_posterior(trace)
plt.show()
