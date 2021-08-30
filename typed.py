import theano
import theano.tensor as tt
import numpy as np


def method(age):

    @theano.compile.ops.as_op(itypes=[tt.dvector], otypes=[tt.dscalar])
    def alpha(age):
        return np.array(age.sum() / age.size)
    return alpha(age)
