import numpy as np
from matplotlib import pyplot as plt
import pymc3 as pm
import seaborn as sns
from hypothesis import given, settings, Phase, HealthCheck, strategies as st
from prior_generator import ProbabilityGenerators as pg
from prior_generator import ProbabilityDistributions as pd
from functools import reduce
from sklearn.feature_selection import mutual_info_regression
import logging
logger = logging.getLogger("pymc3")
logger.setLevel(logging.ERROR)
########################
## Program to analyze ##
########################
def alpha(database):
    return (reduce((lambda i, j: i + j),
                   list(map(lambda i: i[1], database)))
            /
            len(database))

hist = []
labels = []

@settings(max_examples=1, deadline=None, phases=[Phase.generate],suppress_health_check=[HealthCheck.too_slow])
@given(st.data())
def test_with_given(data):
    """
    A way of using the probability genrators to estimate the age of alice
    """
    with pm.Model() as model:

        ###########
        ## Prior ##
        ###########
        # Database
        N = 20 # number of entries in the database
        x = np.empty(N+1, dtype=object)  

        # Alice entry
        age_alice_database = pm.Uniform("alice_age", lower=0, upper=100)
        name_age_alice = "alice_age"
        name_alice_database = pm.Constant("name_alice_database", 0)   

        # Add Alice in database
        x[0] = (name_alice_database, age_alice_database)

        # Other users
        # age = pm.distributions.Normal(name="Age", mu=0, sigma=0.001,shape=N)
        age,age_info = pg.Normal(data, "alice", shape=N)
        # age = pg.FloatGenerator("Age", data=data, shape=N)
        name,name_info = pg.IntList(name="Name", data=data, length=N)
        # Add users to the database
        for i in range(0, N):
            x[i+1] = (name[i], age[i])

            
        #########################
        ## Output distribution ##
        #########################    
        average = pm.Deterministic("average", alpha(x))

        
        # #################
        # ## Observation ##
        # #################
        # obs = pm.Normal('obs', mu=average, sigma=.05, observed=55.3)

        
        ##############
        ## Sampling ##
        ##############
        num_samples = 5000
        # # prior = pm.sample_prior_predictive(num_samples)
        trace = pm.sample(num_samples, cores=1, step=pm.NUTS())
        
        alice_age = trace[name_age_alice]
        output = trace["average"]

        mututal_info = mutual_info_regression([[i] for i in alice_age], output, discrete_features=False)
        outputs = open("output.csv", "a")
        outputs.write(f"{N}!{age_info}!{name_info}!{mututal_info}")
        print(f"Mutual entropy: {mututal_info}")

test_with_given()