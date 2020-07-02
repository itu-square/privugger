import numpy as np
from matplotlib import pyplot as plt
import pymc3 as pm
import seaborn as sns
from functools import reduce

########################
## Program to analyze ##
########################
def alpha(database):
    return (reduce((lambda i, j: i + j),
                   list(map(lambda i: i[1], database)))
            /
            len(database))


with pm.Model() as model:

    ###########
    ## Prior ##
    ###########
    # Database
    N = 20 # number of entries in the database
    x = np.empty(N+1, dtype=object)  

    # Alice entry
    age_alice_database = pm.Uniform("alice_age", lower=0, upper=100)
    name_alice_database = pm.Constant("name_alice_database", 0)   

    # Add Alice in database
    x[0] = (name_alice_database, age_alice_database)

    # Other users
    age = pm.Normal("age", mu=55.2, sigma=3.5, shape=N)
    name = pm.DiscreteUniform("name", 0, 5, shape=N)

    # Add users to the database
    for i in range(0, N):
        x[i+1] = (name[i], age[i])

        
    #########################
    ## Output distribution ##
    #########################    
    average = pm.Deterministic("average", alpha(x))

    
    #################
    ## Observation ##
    #################
    obs = pm.Normal('obs', mu=average, sigma=.05, observed=55.3)

    
    ##############
    ## Sampling ##
    ##############
    num_samples = 5000
    prior = pm.sample_prior_predictive(num_samples)
    trace = pm.sample(num_samples, cores=4)

    
    #############
    ## Queries ##
    #############
    print("P(22.4 < average(x) < 22.5): " + str(np.mean((22.4 < trace["average"])*(trace["average"] < 22.5))))
    print("P(alice < 18): " + str(np.mean(trace["alice_age"] < 18)))

    
    ##############
    ## Plotting ##
    ##############
    sns.distplot(prior["alice_age"], label="Prior", hist=False)
    ax = sns.distplot(trace["alice_age"], label="Posterior", hist=False)
    ax.legend()
    plt.savefig("result-alice-age.pdf")
