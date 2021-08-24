from typing import * 
import numpy as np
from privugger.attacker import parameters, optimizer as opt
from numba import njit

def convert_primitive_to_dist(p, i, is_list, rng=None):
    """
    Converts a primitive data-type to a distribution.

    :param p: The parameter of the underlying PPM
    :param i: id into the domain og constraints
    :param is_list: specifies if the distribution should be generated as a list or single distribution
    :param rng: A RandomState to fix the program, e.g. np.random.RandomState(12345)
    :returns: A method that when called returns a scipy distribution (that has not been sampled on yet) e.g. f(x) = scipy.stats.norm(x)
    """
    if p == float:
        def float_convert(x, domain):
            size = parameters.DB_SIZE if is_list else 1
            return opt.map_int_to_cont_dist(domain[i]["name"], x, domain[i], shape=size, rng=rng)
        return float_convert
    elif p == int:
        def int_convert(x, domain):
            size = parameters.DB_SIZE if is_list else 1
            return opt.map_int_to_discrete_dist(domain[i]["name"], x, domain[i], shape=size)
        return int_convert

def convert_non_primitive_to_dist(p,i, rng=None):
    """
    Converts a non-primitive data-type to a distribution.

    :param p: The parameter of the underlying PPM
    :param i: id into the domain of constraints
    :param rng: A RandomState to fix the program, e.g. np.random.RandomState(12345)
    :returns: all parameters unwrapped into methods of distributions that when called returns the samples distributions
    """
    inner = p.__args__
    partial = []
    for pi in inner:
        if pi == int or pi == float:
            is_list = p.__origin__ == List or p.__origin__ == list
            partial.append(convert_primitive_to_dist(pi, i, is_list, rng=rng))
            i+=1
        else:
            temp, i = convert_non_primitive_to_dist(pi, i, rng=rng)
            partial.append(temp)
    return partial, i

def parse(f,rng=None):
    """
    Parses the ppm and converts it to a list of distributions that

    :param f: The Privacy Preserving Mechanism (PPM)
    :param rng: A RandomState to fix the program, e.g. np.random.RandomState(12345)
    :returns: A list of methods that when call returns a distribution in accordance to the PPM
    """
    methods = []
    parameters = list(f.__annotations__.values())
    i = 0
    for p in parameters[:-1]:
        if p == float or p == int:
            methods.append(convert_primitive_to_dist(p, i, False, rng=rng))
        else:
            # Under this assumption we now have a list or tuple
            temp, i = convert_non_primitive_to_dist(p,i,rng=rng)
            methods.append(temp)
    return methods

def make_output(f, dist):
    """
    Calls a leakage measurement with distributions

    :param f: A method representing leakage measurement
    :param dist: A list of distributions
    :returns: An estimate of the posterior after calling f
    """
    db = np.zeros(len(dist), )
    for i,d in enumerate(dist):
        db[i] = f(d)
    return db

@njit
def unwrap_fast(X):
    """
    A numba version that unwraps a method from [[A,A],[B,B]] to [[A,B],[A,B]]

    :param X: The method to be unwrapped
    :returns: list(zip(*X))
    """
    db = np.empty((len(X[0]), len(X)))
    for i in range(len(X[0])):
        for j in range(len(X)):
            db[i,j] = X[j][i]
    return db

def create_analytical_method(f, q, domain, random_state=None):
    """
    Parses a PPM and converts it to a maximization method

    :param f: A leakage measurement
    :param q: A Privacy Preserving Mechanism (PPM)
    :param domain: Domain specifying parameters range
    :param random_state: A random state to fixate the underlying distribution (e.g. np.random.RandomState(12345))
    :returns: A function with input X that has to be maximised. That is "argmax create_analytical_method" = Best leakage
    """
    if isinstance(random_state, int):
        methods = parse(f,random_state)
    else:
        methods = parse(f)
    if len(list(f.__annotations__.values())) == 2 and len(domain) == 1:
        if isinstance(methods[0], list):
            #Method with one List as input
            def inner(x, i, return_trace=False):
                """
                The inner method for a single parameter of type List[float] or List[int]

                :param x: A vector specifying the parameters
                :param i: Specifying which distribution to generate in accordance to the value it is given in dist.py
                :param return_trace: Default false, but can be called after the optimizer to return the trace
                :returns: The leakage using that particular distributions
                """
                x = np.append(np.asarray([[i]]), x).reshape((1,-1))
                dist = eval_methods(methods[0], x, domain)
                dist_reshape = unwrap_fast(dist)
                out = make_output(f, dist_reshape)
                trace = {
                    f"Alice_{domain[0]['name']}": dist[0],
                    f"Rest_{domain[0]['name']}": dist[1:],
                    "out": out
                }
                if return_trace:
                    return q(trace),trace
                return q(trace)
            return inner
        else:
            #Method with one primitive data type
            def inner(x, i, return_trace=False):
                """
                The inner method for a single primitive data-type PPM (int or float)

                :param x: A vector specifying the parameters
                :param i: Specifying which distribution to generate in accordance to the value it is given in dist.py
                :param return_trace: Default false, but can be called after the optimizer to return the trace
                :returns: The leakage using that particular distributions
                """
                x = np.append(np.asarray([[i]]), x).reshape((1,-1))
                items = methods[0](x[0], domain)
                items = items[0]
                dist = items(parameters.SAMPLES)
                try:
                    out = f(dist)
                except:
                    out = [f(di) for di in dist]
                trace = {
                    f"{domain[0]['name']}": dist,
                    "out": out
                }
                if return_trace:
                    return q(trace), trace
                return q(trace)
            return inner
    else:
        # We will have to do a larger test, such that every parameter becomes a 
        def inner(x, i, return_trace=False):
            """
            The inner method for a multiple parameters

            :param x: A vector specifying the parameters
            :param i: An array specifying which distribution to generate in accordance to the value it is given in dist.py
            :param return_trace: Default false, but can be called after the optimizer to return the trace
            :returns: The leakage using that particular distributions
            """

            # Insert i into array i.e. [24,12,14,13] becomes [0,24,12,0,14,13]
            x_new = np.zeros(len(x[0])+len(x[0])//2)
            pos = 0
            for j in range(len(x_new)):
                if not j % 3:
                    x_new[j] = i[pos]
                    pos += 1
                else:
                    pos = j-((j//3)+1)
                    x_new[j] = x[0][pos]
            # Convert method dist

            
            res = eval_methods(methods, x_new.reshape((1,-1)), domain)
            siz = 1
            for j in range(len(res)):
                if list(res[j].shape)[0] > 1:
                    siz = parameters.DB_SIZE+1
            data = np.empty(tuple([len(res), parameters.SAMPLES, siz]))
            r = res
            
            if len(list(f.__annotations__.values())) == 2:
                while len(r) == 1:
                    r = r[0]
                for j in range(len(r)):
                    r[j] = r[j].flatten()
                data = np.array(list(zip(*r)))
            else:
                for j in range(len(r)):
                    data[j] = list(zip(*res[j]))
            out = np.zeros(parameters.SAMPLES)
            for j in range(len(out)):
                if len(list(f.__annotations__.values())) == 2:
                    out[j] = f(data[j])
                else:
                    out[j] = f(*data[:,j])
            trace = {"out": out}
            for j,d in enumerate(domain):
                #Tuple
                if len(r[j]) == parameters.SAMPLES:
                    trace["Alice_" + d["name"]] = r[j]
                else:
                    trace["Alice_" + d["name"]] = r[j][0]
                    trace["Rest_" + d["name"]] = r[j][1:]
            if return_trace:
                return q(trace), trace
            return q(trace)
        return inner

def eval_methods(method, x, domain, i=0):
    """
    Calls a list of method with the parameters specifying the underlying methods
        - In terms of the report, this method takes I and P as input and calls each I_i with P_i

    :param method: a list of functions that returns distributions (I, in terms of thesis)
    :param x: The parameters for the underlying distribution (P, in terms of thesis)
    :param domain: The domain specifying constraints of the underlying distribution (can be seen in dist.py)
    :param i: An id specifying the what part of x to be used
    :returns: A list of scipy distributions
    """
    res = np.empty(len(method), dtype=object)
    for j, m in enumerate(method):
        if isinstance(m, list):
            res[j] = eval_methods(m, x, domain, i)
        else:
            c = np.asarray([di(parameters.SAMPLES) for di in m(x[0][i:i+3], domain)])
            i+=3
            if len(method) == 1:
                return c
            res[j]=c
    return res