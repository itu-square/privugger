import os
import sys
import inspect
import numpy as np

#Use this on Windows
#currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
#parentdir = os.path.dirname(currentdir)
#sys.path.insert(0, parentdir)


#It appears that this is the way to do it when on Linux
sys.path.append(os.path.join(".."))

import privugger as pv
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
        a = pv.Normal(mu=10, std=3.5)
        b = pv.Normal(mu=40, std=3.5)

    
        # Create dataset and specify program output
        ds = pv.Dataset(input_specs = [a,b],
                        var_names   = ["age", "height"])

        prog = pv.Program(ds, pv.Float, "addition.py")
        #Program
        # self.create_file(lambda a,b: a+b)


        # Call infer
        trace = pv.infer(prog, draws= 1000, cores=1)
        for a,b, o in zip(trace["age"], trace["height"], trace["output"]):
            self.assertEqual(a+b, o)
        # os.remove("temp.py")

    def test_multi_samples_correct_order(self):
        """
        Ensures that when sampling for distribution A and B then output is made of [(a_0,b_0), ..., (a_i, b_i)]
        """
        a = pv.Normal(mu=10, std=3.5)
        b = pv.Normal(mu=40, std=3.5)

    
        # Create dataset and specify program output
        ds = pv.Dataset(input_specs = [a,b],
                        var_names   = ["age", "height"])

        #Program
        # self.create_file(lambda a,b: a*b)

        prog = pv.Program(ds, pv.Float, "multiplication.py")
        # Call infer
        trace = pv.infer(prog, draws= 1000, cores=1)
        for a,b, o in zip(trace["age"], trace["height"], trace["output"]):
            self.assertEqual(a*b, o)
        # os.remove("temp.py")

    def test_uniform_cutoff(self):
        """
        Ensures that when sampling for continuous uniform, no value exceeds domain
        """
        a = pv.Uniform(10,50)

    
        # Create dataset and specify program output
        ds = pv.Dataset(input_specs = [a],
                        var_names   = ["age"])

        prog = pv.Program(ds, pv.Float, "identity.py")
        #Program
        # self.create_file(lambda a: a)


        # Call infer
        trace = pv.infer(prog, draws= 1000, cores=1)
        for ai, oi in zip(trace["age"], trace["output"]):
            self.assertTrue(50 >= ai >= 10)
            self.assertTrue(50 >= oi >= 10)

    def test_discrete_uniform_cutoff(self):
        """
        Ensures that when sampling for discrete uniform, no value exceeds domain
        """
        a = pv.DiscreteUniform(10,50)

    
        # Create dataset and specify program output
        ds = pv.Dataset(input_specs = [a],
                        var_names   = ["age"])

        prog = pv.Program(ds, pv.Int, "identity.py")
        #Program
        # self.create_file(lambda a: a)


        # Call infer
        trace = pv.infer(prog, draws= 1000, cores=1)
        for ai, oi in zip(trace["age"], trace["output"]):
            self.assertTrue(50 >= ai >= 10)
            self.assertTrue(50 >= oi >= 10)

    def test_k_samples_gives_k_samples(self):
        """
        Ensures that the number of samples given also corresponds to the number of samples returned
        """
        sample_size = 1000
        
        # Specify distributions
        age  = pv.Normal(mu=55.2, std=3.5)

        # Create dataset and specify program output
        ds = pv.Dataset(input_specs = [age],
                        var_names   = ["age"])

        #Program
        prog = pv.Program(ds, pv.Float, "identity.py")

        # Call infer
        trace = pv.infer(prog, draws= sample_size, cores=1, chains=1)

        self.assertEqual(len(trace["age"]), sample_size)
        self.assertEqual(len(trace["output"]), sample_size)

    def test_constraints_greater_than_works(self):
        """
        Ensures that setting a constraint actually limits the trace
        """
        def alpha(age):
            return (age.sum()) / (age.size)
        # Database size
        N    = 10
        # Specify distributions
        age  = pv.Normal(mu=55.2, std=3.5, num_elements=N)

        # Create dataset. Refer to "age_alice" as "age1" in the trace and "age" as "age2" in the trace. This is the general naming convention. 
        ds   = pv.Dataset(input_specs = [age],
                        var_names   = ["age"])

        # For now output type can be: Int, Float, List(Float), List(Int)
        program = pv.Program(dataset=ds, output_type=pv.Float, function=alpha)

        # Add observations
        program.add_observation("57>output>56")

        # Call infer and specify program output
        trace = pv.infer(program, cores=2, draws=1000)

        self.assertTrue(all(57 > (np.array(trace["output"] > 56).flatten())))

if __name__ == '__main__':
    unittest.main()
