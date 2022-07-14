


__all__ = [
    "Dataset"
]

class Dataset:

    def __init__(self, input_specs):

        """
        Dataset represents the specs about the input data 
        
        Parameters
        -------------
        input_spec: specification about the distributions of the input given as a list of distributions

        program_output: the output of the program as a standard type
        
        """
        self.input_specs = input_specs

    def _collect_distribution_names(self):
        """
        
        Convenience method for collecting names associated with the distributions 
        
        Returns
        -------------
        names: A list containing all the names of the random variables in the dataset
        """
        names = []
        for distribution in self.input_specs:
            names.append(distribution.name)
        return names
        
    
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


            
