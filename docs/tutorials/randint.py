import aesara
import aesara.tensor as at
import numpy as np


@aesara.compile.ops.as_op(itypes=[at.lscalar], otypes=[at.lscalar])
def method(arg0):

    def randint(arg0):
        y = np.random.randint(arg0)
        return np.array(y)
    return randint(arg0)
