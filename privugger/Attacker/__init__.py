from hypothesis import settings, given, Phase, HealthCheck, strategies as st
import numpy as np
import pymc3 as pm
from privugger.attacker.generators import IntGenerator, IntList, FloatGenerator, FloatList, DiscreteUniform, Uniform
from privugger.attacker.metrics import SimulationMetrics
from sklearn.feature_selection import mutual_info_regression
import typing
import inspect

"""
The data privacy debugger, PRIVUGER, is a privacy risk analysis tool.
"""

def simulate(function, *args, **kwargs) -> SimulationMetrics:
    """
    The main method used to simulate a method

    **Returns: SimulationMetrics **
    ----------
        - Returns a SimulationMetrics, a method helped to assist the data analyst

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
    # Check that all parameters have type annotation:
    if len(inspect.signature(function).parameters) != len(function.__annotations__)-1:
        raise TypeError("You need to specify all types in your method before analyzing")

    # The output trace:
    traces = []

    #Values for KWARGS
    max_examples = 1 if "max_examples" not in kwargs else kwargs["max_examples"]
    N = 2 if "N" not in kwargs else kwargs["N"]
    samples = 1000 if "num_samples" not in kwargs else kwargs["num_samples"]
    current_test = []
    ranges = [(-10000, 10000) for _ in range(len(inspect.signature(function).parameters))] if "ranges" not in kwargs else kwargs["ranges"]
    logging = True if "logging" not in kwargs else kwargs["logging"]

    #Logging
    if not logging:
        import logging
        logger = logging.getLogger("pymc3")
        logger.setLevel(logging.ERROR)
        logger.propagate = False

    @settings(max_examples=max_examples, deadline=None, phases=[Phase.generate], suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much])
    @given(st.data())
    def helper(data):

        def parse(argument, islist=False, istuple=False, parameter_pos=0, ranges=ranges):
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
                    alice_int = pm.distributions.DiscreteUniform(name, ranges[0][0], ranges[0][1])
                    dist, info = IntList(name=f"intList_{parameter_pos}", data=data, length=N, ranges=ranges[0])
                    return ((alice_int, dist,info), parameter_pos)
                else:
                    dist,info = IntGenerator(data=data, name=f"intDist_{parameter_pos}", ranges=ranges[0])
                    if not istuple:
                        return (([dist], dist,info), parameter_pos)
                    alice_int = pm.distributions.DiscreteUniform(name, ranges[0][0], ranges[0][1])
                    return ((alice_int, dist,info), parameter_pos)
            elif argument == float:
                if islist:
                    alice_float = pm.distributions.Uniform(f"Alice-float_{parameter_pos}", ranges[0][0], ranges[0][1])
                    dist, info = FloatGenerator(name=f"floatList_{parameter_pos}", data=data, shape=N, ranges=ranges[0])
                    return ((alice_float, dist,info), parameter_pos)
                else:
                    dist, info = FloatGenerator(name=f"floatDist_{parameter_pos}", data=data, shape=1, ranges=ranges[0])
                    if not istuple:
                        return (([dist], dist,info), parameter_pos)
                    alice_float = pm.distributions.Uniform(f"Alice-float_{parameter_pos}", ranges[0][0], ranges[0][1])
                    return ((alice_float, dist,info), parameter_pos)
            elif argument.__origin__ == list or argument.__origin__ == typing.List:
                # parameter is a list
                x = np.empty(N+1, dtype=object)
                alice_info, d, inf = [],[],[]
                for i, p in enumerate(argument.__args__):
                    (alice, dist, info), pos = parse(p, islist=True, istuple=istuple, parameter_pos=parameter_pos, ranges=ranges[i:])
                    parameter_pos = pos+1
                    alice_info = alice
                    d.append(dist)
                    inf.append(info)
                x[0] = alice
                inputs = [[] for _ in range(N)]
                for k in range(N):
                    for par in d:
                        if len(par) > 1:
                            for dist in par:
                                try:
                                    inputs[k].append(dist[k])
                                except:
                                    break
                        else:
                            inputs[k].append(par[k])
                    x[k+1] = inputs[k]
                return ((alice_info, x, info), parameter_pos)
            elif argument.__origin__ == tuple or argument.__origin__ == typing.Tuple:
                alice_info, d, i = [],[],[]
                for j, p in enumerate(argument.__args__):
                    (alice, dist, info),pos = parse(p, islist=islist, istuple=True, parameter_pos=parameter_pos, ranges=ranges[j:])
                    parameter_pos = pos+1
                    alice_info.append(alice)
                    d.append(dist)
                    i.append(info)
                return ((tuple(alice_info), tuple(d), tuple(i)), parameter_pos)

        #Progress bar
        current_test.append([])
        percentage = (len(current_test)/max_examples)
        width = 100
        fill = "█"*(int(width*percentage))
        tail = "-"*(int(width*(1-percentage)))
        print("\r["+fill+tail+"] " + str(len(current_test)) + "/" + str(max_examples), end="\r")

        #Parameters setup
        parameters = list(function.__annotations__.values())
        pos = 0
        alice_names = []
        outputs = []
        info = []

        #A single data type
        if len(parameters) == 2:
            with pm.Model() as model:
                p = parameters[0]
                (alice, dist, temp_info), pos = parse(p, parameter_pos=pos, ranges=ranges)
                for n in alice:
                    alice_names.append(n)
                outputs.append(dist)
                info.append(temp_info)
                pos+=1
                output = pm.Deterministic("Output", function(*outputs))
                trace = pm.sample(samples, cores=1)
                traces.append((trace, alice_names, info))      
        else:
            TEST_PER_PARAMETER = 10
            # generate for the first parameter and keep the other locked
            for p_pos, p in enumerate(parameters[:-1]):
                pos = 0
                with pm.Model() as model:
                    # p becomes the one to test on
                    alice_info_pairs = [[] for _ in range(TEST_PER_PARAMETER)]
                    info_pairs = []
                    output_pairs = []
                    # Generate 10 samples to test versus one specific distribution
                    for i in range(TEST_PER_PARAMETER):
                        (alice, dist, temp_info), pos = parse(p, parameter_pos=pos)
                        for n in alice:
                            alice_info_pairs[i].append(n)
                        output_pairs.append([dist])
                        info_pairs.append([temp_info])
                        pos+=1
                    for p2_pos, p_2 in enumerate(parameters[:-1]):
                        if p2_pos != p_pos:
                            (alice, dist, temp_info), pos = parse(p, parameter_pos=pos)
                            for i in range(TEST_PER_PARAMETER):
                                for n in alice:
                                    alice_info_pairs[i].append(n)
                                output_pairs[i].append(dist)
                                info_pairs[i].append(temp_info)
                            pos+=1
                    for i in range(TEST_PER_PARAMETER):
                        pm.Deterministic(f"Output_{i}", function(*output_pairs[i]))
                    trace = pm.sample(samples, cores=1, steps=pm.NUTS())
                    for i in range(TEST_PER_PARAMETER):
                        traces.append((trace, alice_info_pairs[i], info_pairs[i]))   
    helper()
    return SimulationMetrics(traces)

def simulate_decorator(*args, **kwargs):
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
        trace = simulate(func, args, kwargs)
        return (lambda x=trace:x)
    return inner

