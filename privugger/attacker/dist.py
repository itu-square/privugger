import scipy as st
import numpy as np

def normal_domain(domain):
    """
    The domain for the normal distribution
    
    :param domain: A domain specifying upper and lower range for a value
    :returns: A dictionary of mu and standard deviation in accordace to normal distribution and and empty set of constraints
    """
    upper_bound_std = np.sqrt((1/12)*(domain["upper"] - domain["lower"])**2)
    return [
        {
            "name": domain["name"]+"_mu",
            "type": "continuous" if domain["type"] == "float" else "discrete",
            "domain": (domain["lower"], domain["upper"]) if  domain["type"] == "float" else tuple(range(int(domain["lower"]), int(domain["upper"]))),
        },
        {
            "name": domain["name"]+"_std",
            "type": "continuous",
            "domain": (0.1, upper_bound_std),
        }
    ],[
    ]

def uniform_domain(domain, pos):
    """
    We add constraints such that the value will scale correctly. 

    One being the following
    1/12*(x_2)^2 >= 0.1
    and 
    x_2 > x_2
    Where x_2 is the scale. and x_1 is the lower

    :param domain: A domain specifying upper and lower range for a value
    :param pos: Indicator for the x value of the vector
    :returns: A dictionary of lower and upper of a uniform dist and a set of constraints as above
    """
    upper = domain["upper"]
    return [
        {
            "name": domain["name"]+"_lower",
            "type": "continuous" if domain["type"] == "float" else "discrete",
            "domain": (domain["lower"], domain["upper"]) if  domain["type"] == "float" else tuple(range(int(domain["lower"]), int(domain["upper"]))),
        },
        {
            "name": domain["name"]+"_spread",
            "type": "continuous" if domain["type"] == "float" else "discrete",
            "domain": (0, domain["upper"]-domain["lower"]) if domain["type"] == "float" else tuple(range(0, domain["upper"]-domain["lower"])),
        }
    ],[
        {
            'name': domain["name"]+"_constr2", 
            'constraint': f'-x[:,{pos+1}]+np.sqrt(6/5)' # Variance >= 0.1
        },
        {
            'name': domain["name"]+"_constr2", 
            'constraint': f'x[:,{pos+1}]+x[:,{pos}]-{upper}' # lower+scale <= domain[upper]
        },
    ]

def poisson_domain(domain, pos):
    """
    The domain and constraints for the parameters of a Poisson Distribution

    Constraints
    lambda+scale < domain Upper

    :param domain: A domain specifying upper and lower range for a value
    :param pos: An indicator to the vector of values
    :returns: A dictionary of lower and upper of a poisson dist and a set of constraints as above
    """
    upper_bound_std = domain["lower"]+np.sqrt((1/12)*(domain["upper"] - domain["lower"])**2)
    upper = domain["upper"]
    return [
        {
            "name": domain["name"]+"_lambda",
            "type": "discrete",
            "domain": tuple(range(int(domain["lower"]), int(domain["upper"]))),
        },
        {
            "name": domain["name"]+"_loc",
            "type": "discrete",
            "domain": tuple(range(int(domain["lower"]), int(domain["upper"]))),
        }
    ],[
        {
            'name': domain["name"]+"_constr2", 
            'constraint': f'x[:,{pos+1}]+x[:,{pos}]-{upper}' # lower+scale <= domain[upper]
        },
    ]

def half_normal_domain(domain, pos):
    """
    The domain and constraints for the parameters of a Half Normal Distribution

    Constraints
    2*sigma + scale < upper (confidence interval)

    :param domain: A domain specifying upper and lower range for a value
    :param pos: An indicator to the vector of values
    :returns: A dictionary of lower and upper of a half normal dist and a set of constraints as above
    """
    upper_bound_std = np.sqrt((1/12)*(domain["upper"] - domain["lower"])**2)
    upper = domain["upper"]
    return [{
        "name": domain["name"]+"_mu",
        "type": "continuous",
        "domain": (domain["lower"], domain["upper"]),
    },{
        "name": domain["name"]+"_std",
        "type": "continuous",
        "domain": (0.1, upper_bound_std)
    }], [{
            'name': domain["name"]+"_constr1", 
            'constraint': f'2*x[:,{pos+1}]+x[:,{pos}]-{upper}' # Variance >= 0.1
        },
    ]