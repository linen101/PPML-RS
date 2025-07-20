import numpy as np
import fxpmath as fxpmath
from fxpmath import Fxp

FXP_CONFIG = dict(signed=True, n_word=32, n_frac=16)
def fxp(val): return Fxp(val, **FXP_CONFIG)
