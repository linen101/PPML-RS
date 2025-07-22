import numpy as np
import fxpmath as fxpmath
from fxpmath import Fxp

FXP_CONFIG = dict(signed=True, n_word=32, n_frac=16)
def fxp(val): return Fxp(val, **FXP_CONFIG)

def split_matrix_fxp(X_parts, Y_parts):
    Xdist_fxp = (np.empty(len(X_parts), dtype=object))
    ydist_fxp = (np.empty(len(Y_parts), dtype=object))

    for i in range(len(X_parts)):
        Xdist_fxp[i] = fxp(X_parts[i])
        ydist_fxp[i] = fxp(Y_parts[i])
        
    return  (Xdist_fxp, ydist_fxp)  

