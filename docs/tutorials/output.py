import numpy as np
import pymc as pm
import aesara.tensor as at


def method(ages):
    return ages.sum() / ages.size
