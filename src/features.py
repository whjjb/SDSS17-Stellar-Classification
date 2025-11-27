import numpy as np
import pandas as pd

PHOT_BANDS = ["u","g","r","i","z"]

def add_color_indices(X: pd.DataFrame) -> pd.DataFrame:
    """Add common color indices if all bands exist. Missing bands are ignored gracefully."""
    X = X.copy()
    cols = set(X.columns)
    def add(name, a, b):
        if a in cols and b in cols:
            X[name] = X[a] - X[b]
    add("u_g", "u","g")
    add("g_r", "g","r")
    add("r_i", "r","i")
    add("i_z", "i","z")
    add("u_r", "u","r")
    add("g_i", "g","i")
    return X

def build_feature_matrix(X: pd.DataFrame) -> pd.DataFrame:
    X = add_color_indices(X)
    # keep known numeric columns only (robust against stray objects)
    for c in X.columns:
        if not np.issubdtype(np.array(X[c]).dtype, np.number):
            # Attempt coercion; non-convertible -> NaN
            X[c] = pd.to_numeric(X[c], errors="coerce")
    # Drop columns that are all NaN after coercion
    X = X.dropna(axis=1, how="all")
    # Simple imputation: fill remaining NaNs with column median
    X = X.fillna(X.median(numeric_only=True))
    return X
