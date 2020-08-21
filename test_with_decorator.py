import pymc3 as pm
from hypothesis import given, settings, HealthCheck, Phase, strategies as st
from prior_generator import ProbabilityGenerators as pg
from prior_generator import ProbabilityDistributions as pd
import matplotlib.pyplot as plt
import random
from functools import reduce

import inspect
from typing import List, Tuple
@pg.Analyze([(1,2.0)])

def alpha(database):
    return (reduce((lambda i, j: i + j),
                   list(map(lambda i: i[1], database)))
            /
            len(database))


trac = alpha()
print(trac)