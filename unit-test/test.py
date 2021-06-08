import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

import privugger.transformer.datastructures as pvds
from privugger.transformer.discrete import *
from privugger.transformer.continuous import *
from privugger.transformer.program_output import *
from privugger.transformer.method import *
from privugger.measures.mutual_information import *
import unittest

class TestProbabilityGenerators(unittest.TestCase):
    program = "unit-test/average_age.py"

    def test_normal_gives_normal(self):
        """
        Test that the mean of the normal actually is close to the mean given
        A small epsilon of 1% is used to argue for incosistency
        """
        mean = 10
        n = Normal(mu=mean, std=3.5)
    
        # Create dataset and specify program output
        ds = pvds.Dataset(input_specs = [n, n],
                        var_names   = ["age", "height"],
                        program_output = Float)

        #Program

        # Call infer
        trace = infer(ds,  self.program, draws= 10000, cores=1)

        eps = 1/100*(mean)
        accuracy_age = mean+eps >= np.mean(trace["age"]) >= mean-eps
        accuracy_height = mean+eps >= np.mean(trace["height"]) >= mean-eps

        self.assertTrue(accuracy_age)
        self.assertTrue(accuracy_height)

    def test_sampling_samples_correct_order(self):
        """
        Ensures that when sampling for distribution A and B then output is made of [(a_0,b_0), ..., (a_i, b_i)]
        """
        a = Normal(mu=10, std=3.5)
        b = Normal(mu=40, std=3.5)

    
        # Create dataset and specify program output
        ds = pvds.Dataset(input_specs = [a,b],
                        var_names   = ["age", "height"],
                        program_output = Float)

        #Program

        # Call infer
        trace = infer(ds, self.program, draws= 1000, cores=1)
        for a,b, o in zip(trace["age"], trace["height"], trace["output"]):
            self.assertEqual(a+b, o)

    def test_k_samples_gives_k_samples(self):
        """
        Ensures that the number of samples given also corresponds to the number of samples returned
        """
        sample_size = 1000
        
        # Specify distributions
        age  = Normal(mu=55.2, std=3.5)
        height = Normal(mu=55.2, std=3.5)

        # Create dataset and specify program output
        ds = pvds.Dataset(input_specs = [age, height],
                        var_names   = ["age", "height"],
                        program_output = Float)

        #Program

        # Call infer
        trace = infer(ds, self.program, draws= sample_size, cores=1, chains=1)

        self.assertEqual(len(trace["age"]), sample_size)
        self.assertEqual(len(trace["height"]), sample_size)

if __name__ == '__main__':
    unittest.main()