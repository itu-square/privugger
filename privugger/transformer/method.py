
from privugger.transformer.type_decoration import *
from privugger.transformer.continuous import Continuous
from privugger.transformer.discrete import Discrete 
from privugger.transformer.theano_types import TheanoToken

import astor
import pymc3 as pm
import theano.tensor as tt

def from_distributions_to_theano(input_specs):
    
    itypes = []
    otype = []
    if(input_specs == None):
        itypes.append(TheanoToken.float_matrix)
    else:
        for s in input_specs:
            if(issubclass(s.__class__, Continuous)):
                itypes.append(TheanoToken.float_vector)
            else:
                itypes.append(TheanoToken.int_vector)

    #if(issubclass(output["dist"], Continuous)):
     #   otype.append("dvector")
    #else:
     #   otype.append("lvector")
    otype.append(TheanoToken.float_scalar)
    return (itypes, otype)

def infer(data_spec, program=None, concat=False, stack=False):
    """

    :param program: the targer program for analysis
   
    :param data_spec: the specifications for the input to the program
   
    """
    
    num_specs      = len(data_spec.input_specs)
    input_specs    = data_spec.input_specs
    var_names      = data_spec.var_names

    #### ##################
    ###### Lift program ###
    #######################
    if(program == None):
        pass
    else:
        ftp = FunctionTypeDecorator()
        if(concat):
            decorators = from_distributions_to_theano([input_specs[0]])
        elif(stack):
            #super hacky but None means that the input is a matrix of floats
            decorators = from_distributions_to_theano(None)
        else:
            decorators = from_distributions_to_theano(input_specs)
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

    with pm.Model() as model:
        
        priors = []
        for idx in range(num_specs):
            priors.append(input_specs[idx].pymc3_dist(var_names[idx]))
        
        print(priors)
        if(concat):
            argument = pm.math.concatenate([*priors])
            if(program==None):
                pass
            else:
                output = pm.Deterministic("output", t.method(argument) )


        elif(stack):
            join = []
            for p in priors:
                join.append(p.reshape(-1,1))
            argument = pm.stack(join, axis=1)
            if(program==None):
                pass
            else:
                output = pm.Deterministic("output", t.method(argument) )

        
        else:
            if(program==None):
                pass
            else:
                output = pm.Deterministic("output", t.method(argument) )
        trace = pm.sample()
        
        return trace


    
