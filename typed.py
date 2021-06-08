import theano
import theano.tensor as tt
import numpy as np


def method(age, height):

    @theano.compile.ops.as_op(itypes=[tt.dscalar, tt.dscalar], otypes=[tt.
        dscalar])
    def name(age, height):
        return np.array(age + height)
    return name(age, height)
