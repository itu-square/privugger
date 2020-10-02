import pymc3 as pm
from hypothesis import given, settings, HealthCheck, Phase, strategies as st
from privugger.attacker import Analyze
import matplotlib.pyplot as plt
import random
from functools import reduce
from typing import * 
import inspect
from typing import List, Tuple

@Analyze(N=20, max_examples=1, num_samples=1000)
def alpha(database : List[Tuple[int, float]]) -> List[Tuple[int, float]]:
    return (reduce((lambda i, j: i + j),
                   list(map(lambda i: i[1], database)))
            /
            len(database))
# @Analyze(N=20, max_examples=1, num_samples=1000)
# def alpha_dp(database: List[Tuple[int, float]],param_ε: float) -> List[Tuple[int, float]]:
#     Δalpha=100/len(database)
#     ε=param_ε
#     b = Δalpha/ε
#     laplace_noise = pm.Laplace("laplace_noise",mu=0,b=b)
#     return (reduce((lambda i, j: i + j),
#                    list(map(lambda i: i[1], database)))
#             /
#             len(database)) + laplace_noise

trac = alpha()
print(trac)