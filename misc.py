# -*- coding: utf-8 -*-
"""misc

Calculate asset positions and more.

"""

import numpy as np
from numba import njit
from consav.linear_interp import binary_search

@njit
def mx_func(t,iota_lag,i_h_lag,i_p,LTV_lag,a_lag,par):
    """ cash-on-hand without and with home equity """

    LTV = np.nan # for renters, this will not be updated

    # a. cash-on-hand
    m = (1+par.ra)*a_lag + par.Gamma[t]*par.grid_p[i_p]
    
    # b. interest rate and repayments
    if i_h_lag >= 0:
        
        # i. prep
        LTV_mortgage_lag = np.fmin(LTV_lag,par.kappa_h_mortgage)
        LTV_bank_loan_lag = np.fmax(LTV_lag-par.kappa_h_mortgage,0.0)    
        repayments = 0.0

        # ii. stocks
        housing_value_lag = par.ph*par.grid_h[i_h_lag]
        mortgage_lag = LTV_mortgage_lag*housing_value_lag
        bank_loan_lag = LTV_bank_loan_lag*housing_value_lag

        # iii. interest payments
        m -= (par.rm*mortgage_lag + par.rb*bank_loan_lag) # interest
        if iota_lag > 0: m -= par.delta*mortgage_lag # extra IO fee
        
        # iv. repayments
        if iota_lag == 0: repayments += par.gamma_m*mortgage_lag # repayment mortgage
        repayments += par.gamma_b*bank_loan_lag # repayment bank_loand
        m -= repayments

        # v. debt and LTV
        debt = (mortgage_lag + bank_loan_lag) - repayments
        LTV = debt/housing_value_lag

    # b. total ressources
    x = m
    if i_h_lag >= 0:
        x += housing_value_lag - debt

    return m,x,LTV

@njit
def bz_func(i_h,LTV,x,mover,par):
    """ mortgage debt and cash-on-hand for consumption """

    # a. mortgage debt
    housing_value = par.ph*par.grid_h[i_h]
    b = LTV*housing_value
    
    # b. cash-on-hand for consumption
    z = x-(1-LTV)*housing_value
    if LTV > 0: z -= par.tau_f # financing costs
    if mover: z -= par.tau_h # moving-in costs

    return b,z

@njit
def nearest_index(x,xi):
    """ find nearest index """

    Nx = x.size

    if xi <= x[0]: 

        return 0

    elif xi >= x[Nx-1]: 

        return Nx-1

    else:

        i = binary_search(0,x.size,x,xi) # x[i] <= x < x[i+1]

        dist_left = xi-x[i]
        dist_right = x[i+1]-xi
        if dist_left < dist_right:
            return i
        else: 
            return i+1