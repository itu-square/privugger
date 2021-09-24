import theano
import theano.tensor as tt
import numpy as np


def method(age, height, weight):

    @theano.compile.ops.as_op(itypes=[tt.dvector, tt.dvector, tt.dvector],
        otypes=[tt.dscalar])
    def alpha(age, height, weight):
        return age.mean(), height.mean(), weight.mean()
    return alpha(age, height, weight)
