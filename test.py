import privugger.transformer.datastructures as pvds
from privugger.transformer.discrete import *
from privugger.transformer.continuous import *
from privugger.transformer.program_output import *

from privugger.transformer.method import *
from privugger.measures.mutual_information import *


def alpha(age):
    return (age.sum()) / (age.size)

#################
## AGE EXAMPLE ##
#################

# Database size
N    = 10

# Specify distributions
age  = Normal(mu=55.2, std=3.5, num_elements=N)


age = add_observation(age, ">4")


#This is an example of how to concatenate
# con  = concatenate(age_alice, age)


# Create dataset. Refer to "age_alice" as "age1" in the trace and "age" as "age2" in the trace. This is the general naming convention. 
ds   = pvds.Dataset(input_specs = [age],
                  var_names   = ["age"])

# Call infer and specify program output
# For now output type can be: Int, Float, List(Float), List(Int)
trace = infer(ds, Float,  alpha, cores=2, draws=1000)

print(trace["age"] <= 4)