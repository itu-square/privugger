from theano import gof
import numpy as np

class IntFloatTuples(gof.Type):

    def filter(self, tuples, strict=True, allow_downcast=None):
        #pass
        if strict:
            if( not isinstance(tuples[0][0], np.int64) or not (isinstance(tuples[0][1], np.float64))):
                raise TypeError("Expected a List of Tuples(int, float)")
            else:
                return tuples
        elif allow_downcast:
            raise TypeError("can not down cast this type")
        else: 
            #print(tuples[0][0])
            #print(type(tuples[0][0]))
            if( not isinstance(tuples[0][0], np.int64) or not (isinstance(tuples[0][1], np.float64))):
                raise TypeError("Expected a List of Tuples(int, float)")
            else:
                return tuples
            #raise TypeError("can not do this")

    def values_eq_approx(self, x, y, tolerance=0):
        pass


intfloattuples = IntFloatTuples()
