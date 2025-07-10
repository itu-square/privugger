def elem_sum(x):
    res = 0
    for i in x:
        res+=i
    #  Having to add np.array here seems to be an problem with
    #  pytensor types. The same problem occurs (namily, having to wrap
    #  the output as np.array()) if we replace the implementation with
    #  x.sum() or sum([i for i in x]).
    return np.array(res) 
