import privugger
import privugger.transformer.datastructures as pvds
from privugger.transformer.discrete import *
from privugger.transformer.continuous import *
from privugger.transformer.program_output import *

from privugger.transformer.method import *
from privugger.measures.mutual_information import *

import arviz as az


import pymc3 as pm

# Specify distributions
age  = Bernoulli(p=0.5)

program = "program_to_analyse.py"
# Create dataset and specify program output
# For now output type can be: Int, Float, List(Float), List(Int)
ds = pvds.Dataset(input_specs = [age],
                  var_names   = ["age"],
                  program_output = Int)

# Call infer
trace = infer(ds,  program, draws= 1000, cores=4, method="scipy")
print(trace)