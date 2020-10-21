import pymc3 as pm
from hypothesis import given, settings, HealthCheck, Phase, strategies as st
from privugger.Attacker import simulate, SimulationMetrics
from privugger.Attacker.generators import IntGenerator
import matplotlib.pyplot as plt
import random
from functools import reduce
from typing import * 
import inspect
from typing import List, Tuple
from privugger.Transformer.type_decoration import load
import privugger.Transformer.typed as typed

# @Analyze(N=20, max_examples=1, num_samples=1000)
# def alpha(database : List[Tuple[int, float]]) -> List[Tuple[int, float]]:
#     return (reduce((lambda i, j: i + j),
#                    list(map(lambda i: i[1], database)))
#             /
#             len(database))
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
import theano
import theano.tensor as tt
import numpy as np
from typing import List

load("program_to_be_analysed.py")

def alpha(database: List[Tuple[int, float]]) -> List[Tuple[int, float]]:
    return (reduce((lambda i, j: i + j),
                   list(map(lambda i: i[1], database)))
            /
            len(database))

# def outer(password: int) ->int:
#     @theano.compile.ops.as_op(itypes=[tt.lscalar], otypes=[tt.lscalar])
#     def original_pwd_checker(password: int) -> int:
#         PWD = 1024
#         return np.int64(password == PWD)
#     return original_pwd_checker(password)


import typed
def outer1(a : int, b : int) -> int:
    return typed.meth(a,b)
#trac = outer()
# print(trac)
trace = simulate(alpha, max_examples=1, num_samples=100, ranges=[(0,1_000_000),(0,100)], logging=False)
# trace = SimulationMetrics(path="metrics2.priv")
trace.plot_mutual_information()
# trace.save_to_file("")
# SimulationMetrics
# trace.plot_distributions()



