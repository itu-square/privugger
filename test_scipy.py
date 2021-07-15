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
name = Normal(mu=10,std=2)

program = "program_to_analyse.py"
# Create dataset and specify program output
# For now output type can be: Int, Float, List(Float), List(Int)
ds = pvds.Dataset(input_specs = [age, name],
                  var_names   = ["age", "name"])

# Call infer
trace = infer(ds,  program=program, draws= 1000, cores=4, method="scipy", program_output = Int)
print(trace)