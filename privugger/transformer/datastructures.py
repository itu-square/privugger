
from privugger.transformer.method import infer


def program(pandas):
    pass

class Dataset():
    """
    Dataset represents the specs about the input data 
    

    :param input_spec: specification about the distributions of the input given as a list of dictionaries
    :param var_names: names associated with each distribution

    """

    def __init__(self, input_specs, var_names):


        if(len(input_specs) != len(var_names)):
            raise ValueError("all names must be specified")
    
        else:
            self.input_specs           = input_specs
            self.var_names             = var_names

        #infer(program, input_specs)



class Float():
    """
    Float represents a floating point value drawn from a spcified distribution

    :param dist: distribution 
    :param name: name associated with distrbution

    """

    def __init__(self, dist= None,  name=None):
        self.dist=dist
        if(name is None):
            raise ValueError("name must be specified")
        else:
            self.name=name
        #infer(program, dist)

   
class Int():

    """
    Int represents an integer value drawn from a spcified distribution

    :param dist: distribution 
    :param name: name associated with distrbution

    """
    def __init__(self, dist=None,  name=None):
        self.dist = dist
        if(name is None):
            raise ValueError("name must be specified")
        else:
            self.name = name


