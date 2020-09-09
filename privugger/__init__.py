from hypothesis import settings, given, Phase, HealthCheck, strategies as st
import numpy as np
import pymc3 as pm
from privugger.generators import IntGenerator, IntList, FloatGenerator, FloatList
from sklearn.feature_selection import mutual_info_regression

"""
The data privacy debugger, PRIVUGER, is a privacy risk analysis tool.
"""

def Analyze(*args, **kwargs):
    """
    ***A decorator used to probabilistically analyse the method***

    **Returns: List[float] **
    ----------
        - Returns a list containing mutual information based on each test

    **Parameters:**
    ----------
    Types: Types
        - The types that your method takes
        - Example: Tuple[int, float]
    *number_of_test: int*
        - Number of test to be executed
        - Default: 1
    *size: int*
        - Size of the database to simulate
        - Default: 4
    *samples: int*
        - Number of samples per. execution
        - Default: 1000
    *ranges: list[tuple[int,int]]*
        - A list of ranges that the distributions should mimic
        - Default: (0, 100)
    """
    def inner(func):
        traces = []
        max_examples = 1 if "max_examples" not in kwargs else kwargs["max_examples"]
        n = 2 if "N" not in kwargs else kwargs["N"]
        samples = 1000 if "num_samples" not in kwargs else kwargs["num_samples"]
        @settings(max_examples=max_examples, deadline=None, phases=[Phase.generate],suppress_health_check=[HealthCheck.too_slow])
        @given(st.data())
        def helper(data):
            with pm.Model() as model:
                N = n # Size of the database
                x = np.empty(N+1, dtype=object)
                #Test subject
                age_alice_database = pm.Uniform("alice_age", lower=0, upper=100)
                name_age_alice = "alice_age"
                name_alice_database = pm.Constant("name_alice_database", 0)  
                
                x[0] = (name_alice_database, age_alice_database)
                def parse(argument, islist=False, istuple=False):
                    if isinstance(argument, list):
                        if len(argument) != 1:
                            raise Exception("The size of the list has to contain only one element")
                        dist, info = parse(argument[0], islist=True,istuple=istuple)
                        return (dist,info)
                    elif isinstance(argument, tuple):
                        dist, info = zip(*[parse(arg, islist=islist,istuple=True) for arg in argument])
                        return (dist, info)
                    elif isinstance(argument, int):
                        if islist:
                            dist, info = IntList(name="IntList", data=data, length=N)
                            return (dist,info)
                        else:
                            dist,info = data.draw(IntGenerator(data=data, name="IntDist"))
                            return (dist,info)
                    elif isinstance(argument, float):
                        if islist:
                            dist, info = FloatList(name="FloatList", data=data, length=N)
                            return (dist, info)
                        else:
                            dist, info = FloatGenerator(name="FloatDist", data=data, shape=1)
                            return (dist,info)
                    else:
                        raise Exception("Type is currently not supported")
                dist, info = zip(*[parse(x) for x in args])
                dist = dist[0]
                for i in range(0,N):
                    x[i+1] = tuple([d[i] for d in dist])

                average = pm.Deterministic("average", func(x))
                num_samples = samples

                trace = pm.sample(num_samples, cores=1, step=pm.NUTS()) 
                output = trace["average"]
                alice_age = trace["alice_age"]
                mututal_info = mutual_info_regression([[i] for i in alice_age], output, discrete_features=False)
                traces.append(mututal_info[0])                  
        helper()
        return (lambda x=traces:x)
    return inner