import os
import sys
import inspect


#Use this on Windows
#currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
#parentdir = os.path.dirname(currentdir)
#sys.path.insert(0, parentdir)


#It appears that this is the way to do it when on Linux
sys.path.append(os.path.join(".."))

import privugger.transformer.datastructures as pvds
from privugger.transformer.discrete import *
from privugger.transformer.continuous import *
from privugger.transformer.program_output import *
from privugger.transformer.method import *
from privugger.measures.mutual_information import *
import unittest

class TestProbabilityGenerators(unittest.TestCase):
    program = "average_age.py"

    def create_file(self, f):
    ## SAVE TO FILE
        res = "".join(inspect.getsourcelines(f)[0])
        res = res.strip()
        if "lambda" in res:
            res = res[len("self.create_file("):-1]
            res = res.strip("\\n")
            if len(res.split("=")) > 1 and "lambda" in res.split("=")[1]:
                res = res[res.find("=")+1:]
        if "lambda" == res[:6]:
            form = res.strip().split(':')
            res = f"def fun({form[0][6:]}): \n  return {form[1]}"
        elif "def" != res[:3]:
            raise TypeError("Method has to be either lambda of def type")
        with open("temp.py", "w") as file:
            file.write(res)


    def test_sampling_samples_correct_order(self):
        """
        Ensures that when sampling for distribution A and B then output is made of [(a_0,b_0), ..., (a_i, b_i)]
        """
        a = Normal(mu=10, std=3.5)
        b = Normal(mu=40, std=3.5)

    
        # Create dataset and specify program output
        ds = pvds.Dataset(input_specs = [a,b],
                        var_names   = ["age", "height"])

        #Program
        # self.create_file(lambda a,b: a+b)


        # Call infer
        trace = infer(ds, Float, "addition.py", draws= 1000, cores=1)
        for a,b, o in zip(trace["age"], trace["height"], trace["output"]):
            self.assertEqual(a+b, o)
        # os.remove("temp.py")

    def test_multi_samples_correct_order(self):
        """
        Ensures that when sampling for distribution A and B then output is made of [(a_0,b_0), ..., (a_i, b_i)]
        """
        a = Normal(mu=10, std=3.5)
        b = Normal(mu=40, std=3.5)

    
        # Create dataset and specify program output
        ds = pvds.Dataset(input_specs = [a,b],
                        var_names   = ["age", "height"])

        #Program
        # self.create_file(lambda a,b: a*b)


        # Call infer
        trace = infer(ds, Float, "multiplication.py", draws= 1000, cores=1)
        for a,b, o in zip(trace["age"], trace["height"], trace["output"]):
            self.assertEqual(a*b, o)
        # os.remove("temp.py")

    def test_uniform_cutoff(self):
        """
        Ensures that when sampling for continuous uniform, no value exceeds domain
        """
        a = Uniform(10,50)

    
        # Create dataset and specify program output
        ds = pvds.Dataset(input_specs = [a],
                        var_names   = ["age"])

        #Program
        # self.create_file(lambda a: a)


        # Call infer
        trace = infer(ds,Float, "identity.py", draws= 1000, cores=1)
        for ai, oi in zip(trace["age"], trace["output"]):
            self.assertTrue(50 >= ai >= 10)
            self.assertTrue(50 >= oi >= 10)

    def test_discrete_uniform_cutoff(self):
        """
        Ensures that when sampling for discrete uniform, no value exceeds domain
        """
        a = DiscreteUniform(10,50)

    
        # Create dataset and specify program output
        ds = pvds.Dataset(input_specs = [a],
                        var_names   = ["age"])

        #Program
        # self.create_file(lambda a: a)


        # Call infer
        trace = infer(ds,Int, "identity.py", draws= 1000, cores=1)
        for ai, oi in zip(trace["age"], trace["output"]):
            self.assertTrue(50 >= ai >= 10)
            self.assertTrue(50 >= oi >= 10)

    def test_k_samples_gives_k_samples(self):
        """
        Ensures that the number of samples given also corresponds to the number of samples returned
        """
        sample_size = 1000
        
        # Specify distributions
        age  = Normal(mu=55.2, std=3.5)

        # Create dataset and specify program output
        ds = pvds.Dataset(input_specs = [age],
                        var_names   = ["age"])

        #Program

        # Call infer
        trace = infer(ds, Float, "identity.py", draws= sample_size, cores=1, chains=1)

        self.assertEqual(len(trace["age"]), sample_size)
        self.assertEqual(len(trace["output"]), sample_size)


if __name__ == '__main__':
    unittest.main()
