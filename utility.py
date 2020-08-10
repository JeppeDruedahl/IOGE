# -*- coding: utf-8 -*-
"""utility

Compute utility and marginal utility.

"""

import numpy as np
from numba import njit

@njit
def func(c,h,owner,par):
    
    if c <= 0.0: return -np.inf

    if owner:
        eta = par.omega*h
    else:
        eta = h

    tot = c**par.alpha*eta**(1-par.alpha)

    norm = par.alpha*(1.0-par.rho)
    return tot**(1.0-par.rho)/norm

@njit
def marg_func(c,h,owner,par):

    if c <= 0.0: return np.inf

    if owner:
        eta = par.omega*h
    else:
        eta = h

    c_exp = par.alpha*(1.0-par.rho)-1.0
    h_exp = (1.0-par.alpha)*(1.0-par.rho)
    return c**c_exp*eta**h_exp

@njit
def inv_marg_func(q,h,owner,par):   

    if owner:
        eta = par.omega*h
    else:
        eta = h

    RHS = q/eta**((1.0-par.alpha)*(1.0-par.rho))
    c_exp = par.alpha*(1.0-par.rho)-1.0
    return RHS**(1.0/c_exp) 