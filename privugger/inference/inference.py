
from privugger.transformer.PyMC3.type_decoration import *
from privugger.distributions.continuous import Continuous
from privugger.distributions.discrete import Discrete, Constant
from privugger.transformer.PyMC3.theano_types import TheanoToken
from privugger.transformer.PyMC3.program_output import *

import astor
import pymc3 as pm
import theano.tensor as tt
import arviz as az
import os
import importlib

## Create a global pymc3 model and list of priors
global_model  = None
global_priors = []

## Set global variables for indicating if there was a concat/stack before infer
concatenated     = False
stacked          = False
global_model_set = False

def _from_distributions_to_theano(input_specs, output):
    
    itypes = []
    otype = []
    
    
    if(input_specs == None):
        itypes.append(TheanoToken.float_matrix)
    else:
        for s in input_specs:
            if(isinstance(s, str)):
                if(s == "continuous"):
                    itypes.append(TheanoToken.float_vector)
                else:
                    itypes.append(TheanoToken.int_vector)

            elif(s.is_hyper_param):
                continue

            elif(issubclass(s.__class__, Continuous)):
                if(s.num_elements == -1):
                    itypes.append(TheanoToken.float_scalar)
                elif(s.num_elements==1):
                    itypes.append(TheanoToken.single_element_float_vector)
                else:
                    itypes.append(TheanoToken.float_vector)

            else:
                if(s.num_elements == -1):
                    itypes.append(TheanoToken.int_scalar)
                elif(s.num_elements==1):
                    itypes.append(TheanoToken.single_element_int_vector)
                else:
                    itypes.append(TheanoToken.int_vector)

    #NOTE: This gets the output type.
    if(type(output).__name__ ==  "type"):
        if(output.__name__ == "Float"):
           otype.append(TheanoToken.float_scalar)
        elif(output.__name__ == "Int"):
           otype.append(TheanoToken.int_scalar)
    elif(type(output).__name__ == "List"):
        if(output.output.__name__ == "Int"):  
           otype.append(TheanoToken.int_vector)
        elif(output.output.__name__ == "Float"): 
           otype.append(TheanoToken.float_vector)
    else:
        if(output.output.__name__ == "Int"):  
           otype.append(TheanoToken.int_matrix)
        elif(output.output.__name__ == "Float"): 
           otype.append(TheanoToken.float_matrix)

            

    return (itypes, otype)

def concatenate(distribution_a, distribution_b,  type_of_dist, axis=0):

    """
    
    Parameters
    ----------- 

    distribution_a: The first distribution
    
    distribution_b: The second distribution
    
    type_of_dist: String that specifies if it is a continuous or discrete distribution
    
    axis: Int value giving the axis to stack
     
    Returns
    -----------
    Type of distribution: String
    """
    
    #NOTE we just return a tuple and then actually concat later. First element is the distributions and second specify the axis and
    #if we are concatenating or stacking
    global global_model
    global global_model_set
    global global_priors
    
    if(not global_model_set):
        global_model     = pm.Model()
        global_priors    = []
        global_model_set = True
    with global_model as model:
        val = pm.math.concatenate( (distribution_a.pymc3_dist(distribution_a.name, []), distribution_b.pymc3_dist(distribution_b.name, [])), axis=axis )
        global_priors.append(val)

    global concatenated
    concatenated = True
    return type_of_dist
    #return ((distribution_a, distribution_b), (axis, "concat"))


def stack(distributions,  type_of_dist, axis=0):
    """
    
    Parameters
    -----------

    distributions: A list of distributions
     
    type_of_dist: String that specifies if it is a continuous or discrete distribution

    axis: Int value giving the axis to stack
     
    Returns
    -----------
    Type of distribution: String
    """
     
    #NOTE we just return a tuple and then actually stack later. First element is the distributions and second specify the axis and
    #if we are concatenating or stacking
    global global_model
    global global_model_set
    global global_priors
    
    if(not global_model_set):
        global_model     = pm.Model()
        global_priors    = []
        global_model_set = True
        
    with global_model as model:
        stacked_variables = []
        for i in range(len(distributions)):
            stacked_variables.append(distributions[i].pymc3_dist(distriutions[i].name, []))
            global_priors.append(pm.math.stack(stacked_variables, axis=axis))

    global stacked
    stacked = True
    return type_of_dist
    #return (distributions, (axis, "stack"))


def get_model():
    if (not global_model_set):
        return None
    else:
        return global_model

def sample_prior(model, samples=50):

    """
    This method does the inference when provided a PyMC3 model

    Parameters
    -------------
    model : PyMC3 model
    
    samples : int number of samples

    Returns
    ------------
    
    Samples from the priors: Priors
    
    """
    with model as sample_prior:
        prior_checks = pm.sample_prior_predictive(samples=samples)

        return prior_checks
    
def infer(prog, cores=2 , chains=2, draws=500, method="pymc3", return_model=False):
    """
    
    Parameters
    -----------
    
    prog: the program type specified as a privugger.Program type
    
    cores: Int number of cores to use for sampling. Default 500
    
    chains: Int number of chains. Default 2

    draws: Int number of draws. Default 2

    method: String specifying which backend to use

    return_model: Boolean. Returns the probabilistic model if true and the trace if false

    Returns
    ----------
    Trace produced by the probabilistic programming inference: Arviz trace
    """
    data_spec      = prog.dataset
    output         = prog.output_type
    num_specs      = len(data_spec.input_specs)
    input_specs    = data_spec.input_specs
    program        = prog.program

    global global_priors
    global global_model

    global concatenated
    global stacked
    global global_model_set
    
    if not (concatenated or stacked or global_model_set):
        
        global_model = pm.Model()
        global_priors = []
        global_model_set = True

    # global_model = pm.Model()
    # global_priors = []
    
    #### ##################
    ###### Lift program ###
    #######################
    if method == "pymc3":
        if(program is not  None):
            ftp = FunctionTypeDecorator()
            decorators = _from_distributions_to_theano(input_specs, output)
            
            lifted_program = ftp.lift(program, decorators)
            lifted_program_w_import = ftp.wrap_with_theano_import(lifted_program)
        
            #print(astor.to_source(lifted_program_w_import))
        
            f = open("typed.py", "w")
            f.write(astor.to_source(lifted_program_w_import))
            f.close()
            import typed as t 
            importlib.reload(t)
            
            #################
            ## Create model #
            #################
            trace = None
            with global_model:
                
                priors = []
                hyper_params = []
                
                for idx in range(num_specs):
                    prior = input_specs[idx]

                    #This is for the case when our prior comes from a concatenated/stacked distribution
                    if(isinstance(prior, str)):
                        continue
                  
                    if(prior.is_hyper_param):
                        hyper_params.append((prior, prior.name))
                    else:
                        params = prior.get_params()
                        hypers_for_prior = []
                        for p_idx in range(len(params)):
                            p = params[p_idx]
                            if(isinstance(p, Continuous) or isinstance(p, Discrete)):
                                for hyper in hyper_params:
                                    if(p.name == hyper[1]):
                                        hypers_for_prior.append((hyper[0],hyper[1], p_idx))
                            
                        #priors.append(prior.pymc3_dist(prior.name, hypers_for_prior))
                        global_priors.append(prior.pymc3_dist(prior.name, hypers_for_prior))
                
                if(program is not None):
                    #output = pm.Deterministic("output", t.method(*priors) )
                    output = pm.Deterministic(prog.name, t.method(*global_priors) )

                # Add observations
                prog.execute_observations(prior, output)

                if(return_model):
                    return global_model
                else:
                    trace = pm.sample(draws=draws, chains=chains, cores=cores,return_inferencedata=True)

                concatenated     = False
                stacked          = False
                global_model_set = False
                del global_model
                del global_priors
                #global_model = pm.Model()
                return trace
            
    elif method == "scipy":
        if isinstance(program, str):
            import re
            f = open(program, "r")
            new = open("typed.py", "w")
            for l in f.readlines():
                res = re.findall("def [a-zA-Z]+\(", l)
                if len(res):
                    new.write(re.sub("def [a-zA-Z]+\(", "def method(", l))
                else:
                    new.write(l)
            f.close()
            new.close()
            import typed as t
            importlib.reload(t)
            f = t.method
        else:
            f = program
        priors = []
        trace = {}
        for idx in range(num_specs):
            name, dist = input_specs[idx].scipy_dist(input_sepcs[idx].name)
            dist = dist(draws)
            priors.append(dist)
            trace[name] = dist
        outputs = []
        for pi in list(zip(*priors)):
            if len(pi) == 1:
                pi = pi[0]
            outputs.append(f(*pi))
        trace["output"] = outputs
        return az.convert_to_inference_data(trace)
    else:
        raise TypeError("Unsupported probabilistic framework")


    
