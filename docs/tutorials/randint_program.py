import numpy as np
import pymc as pm
import aesara.tensor as at
import randint as ra


def method(SECRET):
    X = SECRET
    pm.Deterministic('lign1', X)
    var20 = pm.DiracDelta('var20', np.int64(100))
    Y = ra.method(var20)
    pm.Deterministic('lign2', Y)
    x = X + Y
    pm.Deterministic('lign3', x)
    return x
