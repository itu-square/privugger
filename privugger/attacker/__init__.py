import numpy as np
import GPyOpt
from GPyOpt.methods import BayesianOptimization
from typing import *
from privugger.attacker import parameters, parse, optimizer as opt
from tqdm import tqdm
from privugger.attacker.dist import *
from scipy import stats as st
from joblib import Parallel, delayed


def fix_domain(domain):
    """
    Ensures that the domain given is correct domain

    :param domain: A dictionary specified by the user
    :returns: The domain fixed
    :raises TypeError: If a wrong key was used for a value
    """
    LEGAL_VALS = {
        "name" : lambda x: isinstance(x, str),
        "lower": lambda x: isinstance(x, int) or isinstance(x, float),
        "upper": lambda x: isinstance(x, int) or isinstance(x, float),
        "type": lambda x: x == "int" or x == "float"
    }

    for k, v in LEGAL_VALS.items():
        if not v(domain[k]):
            #Fix the input
            raise TypeError(f"Wrong variable for domain with key {k}, recieved {domain[k]}")
    return domain

def domain_to_dist_ids(d, ids):
    """
    Parameterize an id to a distribution (similar to that in Table 4 in thesis)

    :param d: the domain to be converted to a distribution
    :param ids: A location of what parameter to convert
    """
    res = [fix_domain(domain) for domain in d]
    resulting_domain = []
    constraints = []
    pos = 0
    for i, domain in zip(ids,d):
        if i == 0:
            if domain["type"] == "float":
                dom, cons = normal_domain(domain)
            else:
                dom, cons = poisson_domain(domain, pos)
            pos += 2
        elif i == 1:
            dom, cons = uniform_domain(domain, pos)
            pos += 2
        elif i == 2:
            dom, cons = half_normal_domain(domain, pos)
            pos += 2

        for di in dom:
            resulting_domain.append(di)
        for c in cons:
            constraints.append(c)
    return resulting_domain, constraints

def work_on_random_alice(q):
    """
    A way to look for a random secret in List[...] without altering the program

    :param q: The ppm
    :returns: An altered q from q(a List[int]) to q(a: int, b: List[int])
    """
    if list(q.__annotations__.values())[0].__args__[0] == int:
        def inner(a: int, b: List[int]) -> int:
            return q([a[0]]+b[1:])
    else:
        def inner(a: float, b: List[float]) -> float:
            return q([a[0]]+b[1:])
    return inner

def construct_analysis(q, domain, f, random_state=None, cores=1):
    """
    Analyse a program for any leakage attacks

    :param q: The PPM 
    :param domain: A dictionary specifying parameters range
    :param f: The leakage measurement
    :param random_state: Default none, but can be used to fix a random seed
    :param cores: how many cores to run the analysis on
    :returns: A Wrapper class specifying leakage found
    """
    pars = list(q.__annotations__.values())
    if len(pars) == 2 and len(domain) == 1 and "alice" not in domain[0]:
        if pars[0] != int and pars != float:
            domain.append({
                "name": domain[0]["name"] + "_2",
                "lower": domain[0]["lower"],
                "upper": domain[0]["upper"],
                "type": domain[0]["type"],
                })
            q = work_on_random_alice(q)
    f,q = q,f
    method = parse.create_analytical_method(f, q, domain, random_state)

    comb = np.array([np.arange(parameters.CONT_DIST) if d["type"] == "float" else np.arange(parameters.DISC_DIST) for d in domain])

    combs = np.array(np.meshgrid(*comb)).T.reshape(-1, len(comb))

    # X, Y, fs = [],[], []
    def run_analysis(dist):
        cur_dist, constraint = domain_to_dist_ids(domain, dist)
        feasible_region = GPyOpt.Design_space(space = cur_dist, constraints = constraint) 
        initial_design = GPyOpt.experiment_design.initial_design('random', feasible_region, 10)

        f = lambda x: method(x,dist)
        # fs.append(f)
        #CHOOSE the objective
        objective = GPyOpt.core.task.SingleObjective(f)

        # CHOOSE the model type
        model = GPyOpt.models.GPModel(exact_feval=True,optimize_restarts=10,verbose=False)

        #CHOOSE the acquisition optimizer
        aquisition_optimizer = GPyOpt.optimization.AcquisitionOptimizer(feasible_region)

        #CHOOSE the type of acquisition
        if parameters.ACQUISITION == "LCB":
            acquisition = GPyOpt.acquisitions.AcquisitionLCB(model, feasible_region, optimizer=aquisition_optimizer)
        elif parameters.ACQUISITION == "PI":
            acquisition = GPyOpt.acquisitions.AcquisitionMPI(model, feasible_region, optimizer=aquisition_optimizer)
        elif parameters.ACQUISITION == "EI":
            acquisition = GPyOpt.acquisitions.AcquisitionEI(model, feasible_region, optimizer=aquisition_optimizer)
        else:
            raise TypeError("The parameters for acquisition has to be either EI, PI or LCB")

        #CHOOSE a collection method
        evaluator = GPyOpt.core.evaluators.Sequential(acquisition)
        bo = GPyOpt.methods.ModularBayesianOptimization(model, feasible_region, objective, acquisition, evaluator, initial_design)

        bo.run_optimization(max_iter = parameters.MAX_ITER, eps = parameters.EPS, verbosity=False) 

        

        return bo.X, bo.Y, dist, bo, method
    res = Parallel(n_jobs=cores)(delayed(run_analysis)(dist) for dist in tqdm(combs))

    return wrapper(res, [d["type"] for d in domain])



class wrapper:
    """
    A class to hold all information about the leakage found
    """
    def __init__(self, res, types):
        self.X = [r[0] for r in res]
        self.Y = [r[1] for r in res]
        self.dist = [r[2] for r in res]
        self.bos = [r[3] for r in res]
        self.functions = [r[4] for r in res]
        self.types = types

    def __str__(self):
        return self.best_dist(False)

    def print_dist(self, val, di, t, leakage):
        """
        A method for showing the best distribution found
        """
        if t == "float":
            if di == 0:
                return(f"Normal(mu={val[0]}, sigma={val[1]})")
            elif di == 1:
                return(f"Uniform(lower={val[0]}, scale={val[1]})")
            elif di == 2:
                return(f"HalfNormal(mu={val[0]}, sigma={val[1]})")
        else:
            if di == 0:
                return(f"Poisson(lambda={val[0]}, loc={val[1]})")
            elif di == 1:
                return(f"Discrete Uniform(lower={val[0]}, scale={val[1]}) ")

    def best_dist(self, should_print=True):
        """
        Prints the best distribution based on the Y found
        """
        res = []
        for i in range(len(self.X)):
            best_id = np.argmin(self.Y[i])
            best_y = self.Y[i][best_id]
            best_x = self.X[i][best_id]
            dists = self.dist[i]
            s = ""
            for di, t in zip(dists, self.types):
                s += self.print_dist(best_x, di, t, -best_y)
            s += f" - maximum of {-best_y}"
            res.append(s)
        if should_print:
            for v in res:
                print(v)
        return res

    def maximum(self):
        """ 
        Find the best value acchieved
        """
        best = [min(yi) for yi in self.Y]
        return best
    
    def run(self, i, return_trace=False):
        """
        Executes the program again with the best values

        :param i: An indicator of which dist to use
        """
        return self.functions[i]([self.X[i][np.argmin(self.Y[i])]],i, return_trace=return_trace)
