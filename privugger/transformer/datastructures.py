
#from privugger.transformer.method import infer
import re
import pymc3 as pm


class Dataset:

    def __init__(self, input_specs):

        """
        Dataset represents the specs about the input data 
        
        Parameters
        -------------
        input_spec: specification about the distributions of the input given as a list of distributions
        var_names: names associated with each distribution
        program_output: the output of the program as a standard type
        
        """
        self.input_specs = input_specs


class Program:

    def __init__(self, dataset, output_type, function):
        """
        A class representing the privacy preserving program to be analysed.

        Parameters
        ------------
        dataset: the dataset of type privugger.Dataset containg the values used in the program
        output_type: the output type specified as Int, Float, List(Int), List(Float)
        program: The program to be analysed, either string to location program, lambda method or def function
        """
        if isinstance(dataset, Dataset):
            self.dataset = dataset
            self.output_type = output_type
            self.program = function
            self.observation = None
            self.execute_observations = lambda a,b: None
        else:
            raise ValueError("The dataset has to be of type privugger.Dataset")

    def add_observation(self, constraints):
        """
        Adds observation based on a string so long as the string actually represent legit constraints
        
        e.g: 10 > output > 5

        Parameters
        ------------
        constraints -> String: A string representing the constraints to be added

        """
        cons = "([0-9]*)([>=<]*)([a-zA-Z\s]*)([>=<]*)([0-9]*)"
        vals = re.search(cons, constraints)

        # val1 cons1 name cons2 val2
        # 10 > output >= 2
        val1 = vals.group(1)
        cons1 = vals.group(2)
        name = vals.group(3)
        cons2 = vals.group(4)
        val2 = vals.group(5)

        if name.strip() in self.dataset.var_names:
            name = name.strip()

        partial1 = lambda x : None
        partial2 = lambda x: None
        if name in self.dataset.var_names or "output" in name:
            if val1 != "" and cons1 != "":
                partial1 = self.unwrap_constrain(int(val1), cons1)
            
            if val2 != "" and cons2 != "":
                partial2 = self.unwrap_constrain(int(val2), cons2,i=1)
        
            def inner(prior, output):
                if name in self.dataset.var_names:
                    idx = self.dataset.var_names.index(name)
                    distribution = prior[idx]
                else:
                    distribution = output
                partial1(distribution)
                partial2(distribution)
            self.execute_observations = inner
        else:
            raise ValueError("Observation was not known. Make sure that the name is part of the names in privugger.Datastructure")


    def unwrap_constrain(self, value, cons, i=0):
        if not i % 2:
            cons = cons.replace(">", "<")
        def inner(distribution):
            if cons == ">":
                pm.Normal(f"cons_{i}", distribution>value, 0.01, observed=1)
            elif cons == ">=":
                pm.Normal(f"cons_{i}", distribution>=value, 0.01, observed=1)
            elif cons == "<":
                pm.Normal(f"cons_{i}", distribution<value, 0.01, observed=1)
            elif cons == "<=":
                pm.Normal(f"cons_{i}", distribution<=value, 0.01, observed=1)
            else:
                raise ValueError(f"The program does not support {cons} as a constrain")
        return inner
    
class Float:
    
    def __init__(self, dist= None,  name=None):
        """
        Float represents a floating point value drawn from a spcified distribution
    
        Parameters
        -----------
        dist: distribution 
        name: name associated with distrbution
        
        """
        self.dist=dist
        if(name is None):
            raise ValueError("name must be specified")
        else:
            self.name=name

   
class Int:

    def __init__(self, dist=None,  name=None):
        """
        Int represents an integer value drawn from a spcified distribution

        Parameters
        -----------
        dist: distribution 
        name: name associated with distrbution

        """
        self.dist = dist
        if(name is None):
            raise ValueError("name must be specified")
        else:
            self.name = name


            
