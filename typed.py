import theano
import theano.tensor as tt
import numpy as np
from typing import *


@theano.compile.ops.as_op(itypes=[tt.lscalar, tt.lscalar], otypes=[tt.lscalar])
def meth(v: int, t: int) ->int:
    return np.int64(v == t)

