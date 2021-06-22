
from privugger.transformer.type_decoration import *
from privugger.transformer.continuous import Continuous
from privugger.transformer.discrete import Discrete 
from privugger.transformer.theano_types import TheanoToken
from privugger.transformer.program_output import *

import astor
import pymc3 as pm
import theano.tensor as tt

def from_distributions_to_theano(input_specs, output):
    
    itypes = []
    otype = []
    
    
    if(input_specs == None):
        itypes.append(TheanoToken.float_matrix)
    else:
        for s in input_specs:
            
            if(issubclass(s.__class__, Continuous)):
                if(s.num_elements == -1):
                    itypes.append(TheanoToken.float_scalar)
                elif(s.num_elements==1):
                    itypes.append(TheanoToken.single_element_float_vector)
                else:
                    itypes.append(TheanoToken.float_vector)

            #NOTE Tuple means that we are concatenating the distributions
            elif(s.__class__ is tuple):

                if(issubclass(s[0][0].__class__, Continuous) and issubclass(s[0][1].__class__, Continuous)):
                    itypes.append(TheanoToken.float_vector)

                elif(issubclass(s[0][0].__class__, Discrete) and issubclass(s[0][1].__class__, Discrete)):
                    itypes.append(TheanoToken.int_vector)
                else:
                    raise TypeError("When concatenating the distributions must have the same domain")
                
            else:
                if(s.num_elements == -1):
                    itypes.append(TheanoToken.int_scalar)
                elif(s.num_elements==1):
                    itypes.append(TheanoToken.single_element_int_vector)
                else:
                    itypes.append(TheanoToken.int_vector)

    #NOTE: This gets the output type
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


    return (itypes, otype)


def concatenate(distribution_a, distribution_b, axis=0):
    #NOTE we just return a tuple and then actually concat later
    return ((distribution_a, distribution_b), axis)


def stack(distributions, axis=0):
    return pm.math.stack([*distributions], axis=axis)

def infer(data_spec, program_output,
          program=None, cores=2 ,
          chains=2, draws=500):
    """
    
    Parameters
    -----------
    
    data_spec: A list of the specifications for the input to the program
    
    program_output: The ouput of the program as a normal type

    program: String with a path to the target program for analysis. Default None
   
    cores: Int number of cores to use for sampling. Default 500
    
    chains: Int number of chains. Default 2

    draws: Int number of draws. Default 2

    Returns
    ----------
    trace: Trace produced by the probabilistic programming inference 
    """
    num_specs      = len(data_spec.input_specs)
    input_specs    = data_spec.input_specs
    var_names      = data_spec.var_names
    output         = program_output

    
    #### ##################
    ###### Lift program ###
    #######################
    if(program is not  None):
        ftp = FunctionTypeDecorator()
        decorators = from_distributions_to_theano(input_specs, output)
        
        lifted_program = ftp.lift(program, decorators)
        lifted_program_w_import = ftp.wrap_with_theano_import(lifted_program)
    
        print(astor.to_source(lifted_program_w_import))
    
        
        f = open("typed.py", "w")
        f.write(astor.to_source(lifted_program_w_import))
        f.close()
        
        import typed as t 

    #################
    ## Create model #
    #################
    
    with pm.Model() as model:
        
        priors = []
        for idx in range(num_specs):
            prior = input_specs[idx]
            
             #NOTE Tuple means that we are concatenating the distributions
            if(prior.__class__ is tuple):
                dist_a = prior[0][0].pymc3_dist(var_names[idx] + "1")
                dist_b = prior[0][1].pymc3_dist(var_names[idx] + "2")
                axis = prior[1]
                priors.append( pm.math.concatenate( (dist_a, dist_b), axis=axis) )
            else:
                priors.append(prior.pymc3_dist(var_names[idx]))
        
        print(priors)


      
        if(program is not None):
            output = pm.Deterministic("output", t.method(*priors) )

        trace = pm.sample(draws=draws, chains=chains, cores=cores)
        
        return trace


    
