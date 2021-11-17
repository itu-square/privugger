import theano
import theano.tensor as tt
import numpy as np


def method(age):

    @theano.compile.ops.as_op(itypes=[tt.dscalar], otypes=[tt.dscalar])
    def name(age):
        return np.array(age)
    return name(age)
