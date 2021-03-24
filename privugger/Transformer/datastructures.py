
from type_decoration import load


class Dataset():
    """
    Dataset represents the specs about the input data 
    

    :param input_spec: specification about the distributions of the input given as a list of dictionaries
    :param var_names: names associated with each distribution

    """

    def __init__(self, input_specs=None, output=None, var_names=None):
        self.input_specs = input_specs

        if(len(input_specs) != len(var_names) or var_names is None):
            raise ValueError("all names must be specified")
        self.var_names = var_names
        self.output = output
    



class Float():
    """
    Float represents floating point values drawn from a spcified distribution

    :param dist: distribution 
    :param name: name associated with distrbution

    """

    def __init__(self, dist= None, name=None):
        self.dist=dist

        if(name is None):
            raise ValueError("name must be specified")
        else:
            self.name=name


class Int():

    """
    Int represents a integer values drawn from a spcified distribution

    :param dist: distribution 
    :param name: name associated with distrbution

    """
    def __init__(self, dist=None, name=None):
        self.dist = dist

        if(name is None):
            raise ValueError("name must be specified")
        else:
            self.name = name





