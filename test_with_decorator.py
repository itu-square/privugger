import pymc3 as pm
from hypothesis import given, settings, HealthCheck, Phase, strategies as st
from privugger import Analyze
import matplotlib.pyplot as plt
import random
from functools import reduce

import inspect
from typing import List, Tuple
@Analyze([(1,2.0)],N=20, max_examples=1, num_samples=1000)
def alpha(database):
    return (reduce((lambda i, j: i + j),
                   list(map(lambda i: i[1], database)))
            /
            len(database))

trac = alpha()
print(trac)