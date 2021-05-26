
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
            else:
                if(s.num_elements == -1):
                    itypes.append(TheanoToken.int_scalar)
                elif(s.num_elements==1):
                    itypes.append(TheanoToken.single_element_int_vector)
                else:
                    itypes.append(TheanoToken.int_vector)

    #TODO: get the correct output type
    
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

def infer(data_spec,program=None, cores=2 , chains=2, draws=500, concat=False, stack=False):
    """
    
    Parameters
    -----------
    
    data_spec: A list of the specifications for the input to the program
    
    program: String with a path to the target program for analysis. Default None
   
    cores: Int number of cores to use for sampling. Default 500
    
    chains: Int number of chains. Default 2

    draws: Int number of draws. Default 2

    concat: Boolean indicating if the input should be concatenated into single list. Default false

    stack: Boolean indicating if the input should be stacked into a matrix. Default false
    
    Returns
    ----------
    trace: Trace produced by the probabilistic programming inference 
    """
    num_specs      = len(data_spec.input_specs)
    input_specs    = data_spec.input_specs
    var_names      = data_spec.var_names
    output         = data_spec.program_output
    
    #### ##################
    ###### Lift program ###
    #######################
    if(program == None):
        pass
    else:
        ftp = FunctionTypeDecorator()
        if(concat):
            #TODO do this correct
            #decorators = from_distributions_to_theano([input_specs[0]])
            decorators = from_distributions_to_theano(None, output)
        elif(stack):
            #super hacky but None means that the input is a matrix of floats
            decorators = from_distributions_to_theano(None, output)
        else:
            decorators = from_distributions_to_theano(input_specs, output)
        #print(decorators)
        lifted_program = ftp.lift(program, decorators)
        lifted_program_w_import = ftp.wrap_with_theano_import(lifted_program)
    
        print(astor.to_source(lifted_program_w_import))
    
        #c = compile(astor.to_source(lifted_program_w_import), "lifted", "exec")
        #exec(c)
        f = open("typed.py", "w")
        f.write(astor.to_source(lifted_program_w_import))
        f.close()
        #res = exec(astor.to_source(lifted_program_w_import), {"arguments": a, "arguments": b})
        import typed as t 

    #################
    ## Create model #
    #################
    
    with pm.Model() as model:
        
        priors = []
        for idx in range(num_specs):
            priors.append(input_specs[idx].pymc3_dist(var_names[idx]))
        
        print(priors)
        if(concat):
            argument = pm.math.stack([*priors], axis=0)
            print(argument[0])
            if(program==None):
                pass
            else:
                output = pm.Deterministic("output", t.method(argument) )


        elif(stack):
            join = []
            for p in priors:
                print(p)
                join.append(p.reshape((-1,1)))
            argument = pm.math.stack(join, axis=1)
            if(program==None):
                pass
            else:
                output = pm.Deterministic("output", t.method(argument) )

        
        else:
            if(program==None):
                pass
            else:
                output = pm.Deterministic("output", t.method(*priors) )
        trace = pm.sample(draws=draws, chains=chains, cores=cores)
        
        return trace


    
