from privugger.data_structures.dataset import *
import re
import pymc3 as pm

class Program:

    def __init__(self, name, dataset, output_type, function):
        """
        A class representing the privacy preserving program to be analysed.

        Parameters
        ------------
        dataset: the dataset of type privugger.Dataset containg the values used in the program
        output_type: the output type specified as Int, Float, List(Int), List(Float)
        program: The program to be analysed, either string to location program, lambda method or def function
        """
        if isinstance(dataset, Dataset):
            self.dataset              = dataset
            self.output_type          = output_type
            self.name                 = name
            self.program              = function
            self.observation          = None
            self.execute_observations = lambda a,b: None
        else:
            raise ValueError("The dataset has to be of type privugger.Dataset")

        
    def add_observation(self, constraints, precision=0.01):
        """Adds observation based on a string specifying inequalities of the formed

        `a X var` or `a X var X b` where a,b are of type Int or a,b,c
        are of type Float, `var` is a string corresponding to one of
        the variables in the program or its output, and X \in
        {<,>,=<,>=,==}.
        
        Examples: 
          - 10 > output > 5
          - 52.5 < output
          - output == 42.5 

        Parameters
        ------------
        constraints : String 
            A string representing the constraints to be added

        precision : Float 
            Models the precision with which the observation must
            hold. A value of 0 requires the observation to hold with
            probability 1. The larger the value the lower the
            probability required for the condition to hold.

        """
        cons = "[-+]?([0-9]*\.[0-9]+|[0-9]+)*([>=<]*)([a-zA-Z\s]*)([>=<]*)[-+]?([0-9]*\.[0-9]+|[0-9]+)*"
        constraints = constraints.replace(" ", "")
        vals = re.search(cons, constraints)

        # val1 cons1 name cons2 val2
        # 10 > output >= 2
        val1 = vals.group(1)
        cons1 = vals.group(2)
        name = vals.group(3)
        cons2 = vals.group(4)
        val2 = vals.group(5)

        # DEBUGGING
        # print("val1: " + str(val1) + " cons1: " + str(cons1) + " name: " + str(name) + " cons2: " + str(cons2) + " val2: " + str(val2))

        # Well-formedness checks
        
        ## Check whether we have something of the type `a X var X`
        if val2 == None and cons2 != "":
            raise ValueError("Malformed observation: please review the observation to ensure that the syntax is correct.")

        ## TODO: Check that the type of the a, b in `a X var X b` match the type of var
        
        
        var_names = []
        ########## OLD CODE ############
        # var_names  = self.dataset._collect_distribution_names()
        
        # if name.strip() in var_names:
        #     name = name.strip()
        ########## OLD CODE ############

        partial1 = lambda x: None
        partial2 = lambda x: None
        if name in var_names or "output" in name:
            if val1 != "" and cons1 != "":
                v1 = float(val1) if "." in val1 else int(val1)
                partial1 = self._unwrap_constrain(v1, cons1, precision)
                print(partial1)
                
            if val2 != "" and cons2 != "":
                v2 = float(val2) if "." in val2 else int(val2)
                partial2 = self._unwrap_constrain(v2, cons2, precision, i=1)
        
            def inner(prior, output):
                if name in var_names:
                    idx = var_names.index(name)
                    distribution = prior[idx]
                else:
                    distribution = output
                partial1(distribution)
                partial2(distribution)
            self.execute_observations = inner
            return None # to avoid having a return value
        else:
            raise ValueError("Observation was not known. Make sure that the name is part of the names in privugger.Datastructure")


    def _unwrap_constrain(self, value, cons, precision, i=0):
        if not i % 2:
            cons = cons.replace(">", "<")
        def inner(distribution):
            if cons == ">":
                pm.Normal(f"cons_{i}", distribution>value,  precision, observed=1)
            elif cons == ">=":
                pm.Normal(f"cons_{i}", distribution>=value, precision, observed=1)
            elif cons == "<":
                pm.Normal(f"cons_{i}", distribution<value,  precision, observed=1)
            elif cons == "<=":
                pm.Normal(f"cons_{i}", distribution<=value, precision, observed=1)
            elif cons == "==":
                pm.Normal(f"cons_{i}", distribution,        precision, observed=value)
            else:
                raise ValueError(f"The program does not support {cons} as a constrain")
        return inner
