from hypothesis import settings, given, Phase, HealthCheck, strategies as st
import numpy as np
import pymc3 as pm
from privugger.attacker.generators import IntGenerator, IntList, FloatGenerator, FloatList, DiscreteUniform, Uniform
from sklearn.feature_selection import mutual_info_regression
import typing
import inspect

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
        # Check that all parameters have type annotation:
        if len(inspect.signature(func).parameters) != len(func.__annotations__)-1:
            raise TypeError("You need to specify all types in your method before analyzing")
            """
            TODO: Investigate if TypeError is the correct error to throw
            """
        # The output trace:
        traces = {}

        #Values for KWARGS
        max_examples = 1 if "max_examples" not in kwargs else kwargs["max_examples"]
        N = 2 if "N" not in kwargs else kwargs["N"]
        samples = 1000 if "num_samples" not in kwargs else kwargs["num_samples"]
        """
        TODO: Add support for ranges
        """
        @settings(max_examples=max_examples, deadline=None, phases=[Phase.generate], suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much])
        @given(st.data())
        def helper(data):
            with pm.Model() as model:
                def parse( argument, islist=False, istuple=False, parameter_pos=0):
                    """
                    PSEUDO CODE FOR PARSE:
                    values <- Ø
                    For each p <- parameter type in argument
                        case p == list
                            x <- Ø
                            s <- Construct a secret to cover all possible values as p
                            x[0] <- s
                            for n=1, n<=N, n++
                                d <- parse(p)
                        return x
                        case p == tuple
                            x <- Ø
                            for t <- type in p
                                x += parse(t)
                        case p == float
                            d <- float generator
                            return d
                        case p == int
                            d <- int generator
                            return d
                    """
                    if argument == int:
                        name = f"Alice-int_{parameter_pos}"
                        if islist:
                            #Figure out how to fetch range:
                            alice_int = pm.distributions.DiscreteUniform(name, 0, 100)
                            dist, info = IntList(name=f"IntList_{parameter_pos}", data=data, length=N)
                            return ((alice_int, dist,info), parameter_pos)
                        else:
                            alice_int = pm.distributions.DiscreteUniform(name, 0, 100)
                            dist,info = IntGenerator(data=data, name=f"IntDist_{parameter_pos}")
                            return ((alice_int, dist,info), parameter_pos)
                    elif argument == float:
                        if islist:
                            alice_float = pm.distributions.Uniform(f"Alice-float_{parameter_pos}", 0, 100)
                            dist, info = FloatList(name=f"FloatList_{parameter_pos}", data=data, length=N)
                            return ((alice_float, dist,info), parameter_pos)
                        else:
                            alice_float = pm.distributions.Uniform(f"Alice-float_{parameter_pos}", 0, 100)
                            dist, info = FloatGenerator(name=f"FloatDist_{parameter_pos}", data=data, shape=1)
                            return ((alice_float, dist,info), parameter_pos)
                    elif argument.__origin__ == list or argument.__origin__ == typing.List:
                        # parameter is a list
                        x = np.empty(N+1, dtype=object)
                        alice_info, d, inf = [],[],[]
                        for p in argument.__args__:
                            parameter_pos += 1
                            (alice, dist, info), pos = parse(p, islist=True, istuple=istuple, parameter_pos=parameter_pos)
                            parameter_pos = pos
                            alice_info = alice
                            d.append(dist)
                            inf.append(info)
                        x[0] = alice
                        inputs = [[] for _ in range(N)]
                        for i in range(N):
                            for par in d:
                                if len(par) > 1:
                                    for dist in par:
                                        inputs[i].append(dist[i])
                                else:
                                    inputs[i].append(par[i])
                            x[i+1] = inputs[i]
                        return ((alice_info, x, info), parameter_pos)
                    elif argument.__origin__ == tuple or argument.__origin__ == typing.Tuple:
                        alice_info, d, i = [],[],[]
                        for p in argument.__args__:
                            parameter_pos += 1
                            (alice, dist, info),pos = parse( p, islist=islist, istuple=True, parameter_pos=parameter_pos)
                            parameter_pos = pos
                            alice_info.append(alice)
                            d.append(dist)
                            i.append(info)
                        return ((tuple(alice_info), tuple(d), tuple(i)), parameter_pos)
                parameters = list(func.__annotations__.values())
                pos = 0
                alice_names = []
                outputs = []
                info = []
                for p in parameters[:-1]:
                    (alice, dist, temp_info), pos = parse(p, parameter_pos=pos)
                    for n in alice:
                        alice_names.append(n)
                    outputs.append(dist)
                    info.append(temp_info)
                output = pm.Deterministic("Output", func(*outputs))
                trace = pm.sample(samples, cores=1, step=pm.NUTS())


                for name in alice_names:
                    alice = trace[name]
                    mututal_info = mutual_info_regression([[i] for i in alice], trace["Output"], discrete_features=False)
                    max_entropy = mutual_info_regression([[i] for i in alice], alice, discrete_features=False)
                    traces[name] = (mututal_info[0]/max_entropy[0], info)                
        helper()
        return (lambda x=traces:x)
    return inner

