import theano
import theano.tensor as tt
import numpy as np
from typing import List


@theano.compile.ops.as_op(itypes=[tt.lscalar], otypes=[tt.lscalar])
def original_pwd_checker(password: int) ->int:
    PWD = 1024
    return np.int64(password == PWD)

