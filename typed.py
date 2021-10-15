import theano
import theano.tensor as tt
import numpy as np


def method(priors):

    @theano.compile.ops.as_op(itypes=[tt.dvector, tt.dvector, tt.dvector],
        otypes=[tt.dscalar])
    def function(priors):
        return [np.mean(priors['age']), np.mean(priors['height']), np.mean(
            priors['weight'])]
    return function(priors)
