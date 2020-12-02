import theano
import theano.tensor as tt
import numpy as np
from typing import List, Tuple
from privugger.Attacker import simulate, SimulationMetrics
from privugger.Attacker.generators import IntGenerator

def outer1(data: List[Tuple[int, float,int,int]]) -> float:
    def PPM(data: List[Tuple[int, float,int,int]]) -> float:
        data = data[:2]
        data = list(map(lambda x: x[0] + x[1], data))
        return sum(data)    
    return PPM(data)

trace = simulate(outer1, max_examples=20, num_samples=10000, ranges=[(0,120),(0,300), (0,10), (0,10)], logging=False)
trace.save_to_file("")