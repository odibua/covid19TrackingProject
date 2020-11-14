# --------------------------
# Standard Python Imports
# --------------------------
from typing import Any, Callable, Dict, List, Tuple, Union

# --------------------------
# Third Party Imports
# --------------------------
import numpy as np
import statsmodels.api as sm

# --------------------------
# covid19Tracking Imports
# --------------------------

def call_multilinear_regression(X: np.ndarray, Y: np.ndarray) -> Any:
    X = sm.add_constant(X)
    model = sm.OLS(Y, X)
    results = model.fit()
    print(results.params)
    return results.summary()