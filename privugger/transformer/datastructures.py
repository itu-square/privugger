
from privugger.transformer.method import infer


def program(pandas):
    pass

class Dataset():

    def __init__(self, input_specs, var_names):

        """
        Dataset represents the specs about the input data 
        
        Parameters
        -------------
        input_spec: specification about the distributions of the input given as a list of distributions
        var_names: names associated with each distribution
        program_output: the output of the program as a standard type
        
        """
        if(len(input_specs) != len(var_names)):
            raise ValueError("There must be the same number of inputs and names")
    
        else:
            self.input_specs           = input_specs
            self.var_names             = var_names



    
class Float():
    
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

   
class Int():

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


            
