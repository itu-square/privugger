import pymc3 as pm
from hypothesis import given, settings, HealthCheck, Phase, strategies as st
from privugger.Attacker import simulate
from privugger.Attacker.generators import IntGenerator
import matplotlib.pyplot as plt
import random
from functools import reduce
from typing import * 
import inspect
from typing import List, Tuple
from privugger.Transformer.type_decoration import load
import importlib
import astor


# @Analyze(N=20, max_examples=1, num_samples=1000)
# def alpha_dp(database: List[Tuple[int, float]],param_ε: float) -> List[Tuple[int, float]]:
#     Δalpha=100/len(database)
#     ε=param_ε
#     b = Δalpha/ε
#     laplace_noise = pm.Laplace("laplace_noise",mu=0,b=b)
#     return (reduce((lambda i, j: i + j),
#                    list(map(lambda i: i[1], database)))
#             /
#             len(database)) + laplace_noise



#lift the program
#program variable is the new AST of the transformed program
#in the "load"-function specify the path to the file, and the function to analyse
program = load("privugger/Transformer/password-program.py", "original_pwd_checker")

#Write this to a file called "typed.py". This file need to exist somewhere
with open("privugger/Transformer/typed.py", "w") as decorated_file: 
    decorated_file.write(astor.to_source(program))

#Dynamically import the moydle called typed and use the generic function "method" which wraps the transformed function
typed = importlib.import_module("privugger.Transformer.typed")

#Run the analysis
trace = simulate(typed.method, max_examples=10, num_samples=100, ranges=[(0,100),(0,100)])
trace.plot_mutual_information()



