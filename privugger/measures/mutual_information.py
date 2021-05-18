"Mutual information estimators"

import numpy as np
from sklearn.feature_selection import mutual_info_regression


def mi_sklearn(
        t_secret,
        t_output,
        disc_features=False,
        log2=True,
        n_neigh=20
):
    """Binding for estimating mutual information using the `mutual_info_regression` function from `sklearn`.
   

    Parameters
    ----------
    t_secret: array of samples
        array of samples of the secret variable
    t_output: array of samples
        array of samples of the output variable
    data: az.InferenceData object
        InferenceData object containing the posterior/prior data.
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
    mi_nat = mutual_info_regression(t_secret.reshape(-1,1),
                                    t_output,
                                    discrete_features=disc_features,
                                    n_neighbors=n_neigh)
    result = mi_nat/np.log(2) if log2 else mi_nat
    return result
