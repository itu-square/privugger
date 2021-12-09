"Mutual information estimators"

import numpy as np
from sklearn.feature_selection import mutual_info_regression


def mi_sklearn(
        trace,
        var_names=[],
        disc_features=False,
        log2=True,
        n_neigh=20,
        input_inferencedata=True
):
    """
    Binding for estimating mutual information using the `mutual_info_regression` function from `sklearn`.

    Concretely, this function computes mutual information as I(var_names[0]; var_names[1]).      
   

    Parameters
    ----------
    trace : az.InferenceData object
        InferenceData object containing the samples for variables of interest
    var_names: String \times String
        String array with two elements indicating the variables used to compute mutual information.
        Note that the array must contain exactly two elements.
    disc_features : bool
        Boolean indicating whether the domain of the secret is discerete. Default False.
    log2: bool
        Result in log2. Default True. If False, the result is in natural logarithm.
    n_neigh: int
        Number of neighbours used by de estimator. Default 20.

    Returns
    -------
    result : float
        Mutual information in log2 or ln.
    """
    trace1 = trace2 = []
    if input_inferencedata:
        assert len(var_names)==2, "var_names must contain exactly two elements"
        trace1 = trace.posterior[var_names[0]].values.flatten().reshape(-1,1)
        trace2 = trace.posterior[var_names[1]].values.flatten()
    else:
        assert len(trace) == 2, "trace must containt two subtraces for each of each random variables to compute mutual information"
        trace1 = trace[0].reshape(-1,1)
        trace2 = trace[1]

    mi_nat = mutual_info_regression(trace1,
                                    trace2,
                                    discrete_features=disc_features,
                                    n_neighbors=n_neigh)
    result = mi_nat/np.log(2) if log2 else mi_nat
    return result
