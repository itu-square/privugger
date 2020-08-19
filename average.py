import numpy as np
from matplotlib import pyplot as plt
import pymc3 as pm
import seaborn as sns
from hypothesis import given, settings, Phase, HealthCheck, strategies as st
from prior_generator import ProbabilityGenerators as pg
from prior_generator import ProbabilityDistributions as pd
from functools import reduce
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

@settings(max_examples=4, deadline=None, phases=[Phase.generate],suppress_health_check=[HealthCheck.too_slow])
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
        N = 1 # number of entries in the database
        x = np.empty(N+1, dtype=object)  

        # Alice entry
        age_alice_database = pm.Uniform("alice_age", lower=0, upper=100)
        name_age_alice = "alice_age"
        # age_alice_database = data.draw(pg.FloatGenerator(name="alice_age"))
        # name_age_alice = age_alice_database[2]
        name_alice_database = pm.Constant("name_alice_database", 0)   

        # Add Alice in database
        x[0] = (name_alice_database, age_alice_database)

        # Other users
        age = [data.draw(i) for i in data.draw(pg.FloatList(name="Age",length=N, possible_dist=[pd.Normal,pd.Triangular]))]
        # age = data.draw(pg.NormalDist(name="Age", shape=N))
        name = [data.draw(i) for i in data.draw(pg.IntList(name="Name", length=N, possible_dist=[pd.Binomial]))]

        # Add users to the database
        for i in range(0, N):
            x[i+1] = (name[i][0], age[i][0])

            
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
        num_samples = 10
        # # prior = pm.sample_prior_predictive(num_samples)
        trace = pm.sample(num_samples, cores=1)

        
        # #############
        # ## Queries ##
        # #############
        # print("P(22.4 < average(x) < 22.5): " + str(np.mean((22.4 < trace["average"])*(trace["average"] < 22.5))))
        print("P(alice < 18): " + str(np.mean(trace[name_age_alice] < 18)))
        string_to_add = []
        for dist in age:
            for i, parameters in enumerate(dist[1]):
                string_to_add.append(f"{parameters} ")
        ind = set(string_to_add)
        name = ""
        for n in ind:
            name += f"{n} : {string_to_add.count(n)}"
        labels.append(n)
        hist.append(np.mean(trace[name_age_alice] < 18))
        
        # ##############
        # ## Plotting ##
        # ##############
        # sns.distplot(prior["alice_age"], label="Prior", hist=False)
        # ax = sns.distplot(trace["alice_age"], label="Posterior", hist=False)
        # ax.legend()
        # plt.show()

test_with_given()
print(hist)
print(labels)

fig,ax = plt.subplots()
ax.barh(np.arange(len(hist)), hist,align="center")
ax.set_yticks(np.arange(len(hist)))
ax.set_yticklabels(labels)
ax.set_xlabel("p(alice < 18)")
ax.set_ylabel("Number of various distributions used to mimic the Database")
ax.set_title("Performance of difference distributions as the database")

plt.show()