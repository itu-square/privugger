import pymc3 as pm
from hypothesis import given, settings, HealthCheck, Phase, strategies as st
from privugger.attacker import simulate
from privugger.attacker.generators import IntGenerator
import matplotlib.pyplot as plt
import random
from functools import reduce
from typing import * 
import inspect
from typing import List, Tuple

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

def outer(password: int) ->int:
    @theano.compile.ops.as_op(itypes=[tt.lscalar], otypes=[tt.lscalar])
    def original_pwd_checker(password: int) -> int:
        PWD = 1024
        return np.int64(password == PWD)
    return original_pwd_checker(password)
# trac = outer()
# print(trac)
trace = simulate(outer, max_examples=2, num_samples=10000, ranges=[(0,10)])
print(trace())
# @settings(max_examples=1, deadline=None, phases=[Phase.generate],suppress_health_check=[HealthCheck.too_slow,  HealthCheck.filter_too_much])
# @given(st.data())
# def test(data):
#     with pm.Model() as model:
#         prior,info = IntGenerator(data, "prior")
#         print(prior)
#         output = pm.Deterministic("output", outer(prior))
#         trace = pm.sample(1000, cores=1)
#         pm.traceplot(trace)
#         plt.show()
# test()


