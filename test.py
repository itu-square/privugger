import privugger as pv
import numpy as np

def alpha(age):
    return (age.sum()) / (age.size)

#################
## AGE EXAMPLE ##
#################

# Database size
N    = 10

# Specify distributions
age  = pv.Normal(mu=55.2, std=3.5, num_elements=N)

# Create dataset. Refer to "age_alice" as "age1" in the trace and "age" as "age2" in the trace. This is the general naming convention. 
ds   = pv.Dataset(input_specs = [age],
                  var_names   = ["age"])

# For now output type can be: Int, Float, List(Float), List(Int)
program = pv.Program(dataset=ds, output_type=pv.Float, method=alpha)

# Add observations
program.add_observation("output>56")

# Call infer and specify program output
trace = pv.infer(program, cores=2, draws=1000)

print(trace["output"])
print(all(np.array(trace["output"] > 56).flatten()))