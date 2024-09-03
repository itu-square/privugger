import os
import sys
import inspect
import numpy as np

#Use this on Windows
#currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
#parentdir = os.path.dirname(currentdir)
#sys.path.insert(0, parentdir)


# It appears that this is the way to do it when on Linux
sys.path.append(os.path.join("../.."))

import privugger as pv
import unittest

program_alpha = "privugger/test/alpha.py"
program_addition = "privugger/test/addition.py"
program_multiplication = "privugger/test/multiplication.py"
program_identity = "privugger/test/identity.py"
    

class TestProbabilityGenerators(unittest.TestCase):
    
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
        a = pv.Normal("age", mu=10, std=3.5)
        b = pv.Normal("height", mu=40, std=3.5)

    
        # Create dataset and specify program output
        ds = pv.Dataset(input_specs = [a,b])

        prog = pv.Program("output", dataset=ds, output_type=pv.Float, function=program_addition)
        #Program
        # self.create_file(lambda a,b: a+b)


        # Call infer
        trace = pv.infer(prog, draws= 1000, cores=1)
        os.remove("typed.py")
        for a,b, o in zip(trace.posterior["age"].values, trace.posterior["height"].values, trace.posterior["output"].values):
            for i in range(len(a)): 
                self.assertEqual(a[i]+b[i], o[i])
        # os.remove("temp.py")


    def test_multi_samples_correct_order(self):
        """
        Ensures that when sampling for distribution A and B then output is made of [(a_0,b_0), ..., (a_i, b_i)]
        """
        a = pv.Normal("age", mu=10, std=3.5)
        b = pv.Normal("height", mu=40, std=3.5)

    
        # Create dataset and specify program output
        ds = pv.Dataset(input_specs = [a,b])

        #Program
        # self.create_file(lambda a,b: a*b)

        prog = pv.Program("output", dataset=ds, output_type=pv.Float, function=program_multiplication)
        # Call infer
        trace = pv.infer(prog, draws= 1000, cores=1)
        os.remove("typed.py")
        for a,b, o in zip(trace.posterior["age"], trace.posterior["height"], trace.posterior["output"]):
            for i in range(len(a)):
                self.assertEqual(a[i]*b[i], o[i])


    def test_uniform_cutoff(self):
        """
        Ensures that when sampling for continuous uniform, no value exceeds domain
        """
        a = pv.Uniform("age", 10,50)

    
        # Create dataset and specify program output
        ds = pv.Dataset(input_specs = [a])

        prog = pv.Program("output", dataset=ds, output_type=pv.Float, function=program_identity)
        #Program
        # self.create_file(lambda a: a)


        # Call infer
        trace = pv.infer(prog, draws= 1000, cores=1)
        os.remove("typed.py")
        for ai, oi in zip(trace.posterior["age"], trace.posterior["output"]):
            for i in range(len(ai)):
                self.assertTrue(50 >= ai[i] >= 10)
                self.assertTrue(50 >= oi[i] >= 10)

    def test_discrete_uniform_cutoff(self):
        """
        Ensures that when sampling for discrete uniform, no value exceeds domain
        """
        a = pv.DiscreteUniform("age", 10,50)

    
        # Create dataset and specify program output
        ds = pv.Dataset(input_specs = [a])

        prog = pv.Program("output", dataset=ds, output_type=pv.Int, function=program_identity)
        #Program
        # self.create_file(lambda a: a)


        # Call infer
        trace = pv.infer(prog, draws= 1000, cores=1)
        os.remove("typed.py")
        for ai, oi in zip(trace.posterior["age"], trace.posterior["output"]):
            for i in range(len(ai)):
                self.assertTrue(50 >= ai[i] >= 10)
                self.assertTrue(50 >= oi[i] >= 10)

            
    def test_k_samples_gives_k_samples(self):
        """
        Ensures that the number of samples given also corresponds to the number of samples returned
        """
        sample_size = 1000
        
        # Specify distributions
        age  = pv.Normal("age", mu=55.2, std=3.5)

        # Create dataset and specify program output
        ds = pv.Dataset(input_specs = [age])

        #Program
        prog = pv.Program("output", dataset=ds, output_type=pv.Float, function=program_identity)

        # Call infer
        trace = pv.infer(prog, draws= sample_size, cores=1, chains=1)
        os.remove("typed.py")
        self.assertEqual(len(trace.posterior["age"][0]), sample_size)
        self.assertEqual(len(trace.posterior["output"][0]), sample_size)
        

    def test_constraints_greater_than_works(self):
        """
        Ensures that setting a constraint actually limits the trace
        """

        # Database size
        N    = 10
        # Specify distributions
        age  = pv.Normal("age", mu=55.2, std=3.5, num_elements=N)

        # Create dataset. Refer to "age_alice" as "age1" in the trace and "age" as "age2" in the trace. This is the general naming convention. 
        ds   = pv.Dataset(input_specs = [age])

        # For now output type can be: Int, Float, List(Float), List(Int)
        program = pv.Program("output", dataset=ds, output_type=pv.Float, function=program_alpha)

        # Add observations
        program.add_observation("57>=output>=56")

        # Call infer and specify program output
        trace = pv.infer(program, cores=2, draws=1000)
        os.remove("typed.py")

        self.assertTrue(all(57 > (np.array(trace.posterior["output"] > 56).flatten())))


if __name__ == '__main__':
    unittest.main()
