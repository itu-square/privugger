import theano
import theano.tensor as tt
import numpy as np


def method(names, zips, ages, genders, diagnosis):

    @theano.compile.ops.as_op(itypes=[tt.lvector, tt.lvector, tt.dvector,
        tt.lvector, tt.lvector], otypes=[tt.lscalar])
    def zero(names, zips, ages, genders, diagnosis):
        return np.int64(0)
    return zero(names, zips, ages, genders, diagnosis)
