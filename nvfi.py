# -*- coding: utf-8 -*-
"""nvfi

Apply nested value function tieration.

"""

import numpy as np
from numba import njit, prange

from consav import linear_interp 

import utility
import misc

###########
# generic #
###########

@njit
def logsum_and_choice_probabilities(v,sigma,p):
    """ logsum and choice probabilities """

    imax = np.argmax(v)
    maxv = v[imax]

    expsum = np.sum( np.exp( (v-maxv)/sigma ) )
    logsum = maxv + sigma*np.log(expsum)

    p[:] = np.exp( (v-logsum)/sigma )

    return logsum

################
# intermediary #
################

@njit(parallel=True)
def evaluate_rt(par,sol,t):
    """ evaluate the intermediary problem for renters """

    # fill out these arrays
    rt_inv_v = sol.rt_inv_v[t]
    rt_inv_mu = sol.rt_inv_mu[t]

    # a. states
    for i_p in prange(par.Np):
        for i_ht_lag in range(par.Nht+1):
            for i_beta in range(par.Nbeta):

                # b. allocate
                inv_v = np.zeros(par.Nx)
                inv_mu = np.zeros(par.Nx)

                # c. choices
                for i_ht in range(par.Nht):
                                    
                    # i. implied z
                    ht = par.grid_ht[i_ht]
                    z = par.grid_x-par.rh*par.ph*ht
                    if not i_ht == i_ht_lag: z -= par.tau_ht

                    # ii. interpolate
                    prep = linear_interp.interp_1d_prep(par.Nx)

                    inv_vbar = sol.r_inv_vbar[t,i_beta,i_ht,i_p,:]
                    linear_interp.interp_1d_vec_mon(prep,par.grid_z,inv_vbar,z,inv_v)

                    inv_mubar = sol.r_inv_mubar[t,i_beta,i_ht,i_p,:]
                    linear_interp.interp_1d_vec_mon_rep(prep,par.grid_z,inv_mubar,z,inv_mu)

                    # iii. save
                    for i_x in range(par.Nx):
                        
                        if z[i_x] <= 0.0:

                            rt_inv_v[i_beta,i_ht_lag,i_p,i_x,i_ht] = 0.0
                            rt_inv_mu[i_beta,i_ht_lag,i_p,i_x,i_ht] = 0.0

                        else:

                            rt_inv_v[i_beta,i_ht_lag,i_p,i_x,i_ht] = inv_v[i_x]
                            rt_inv_mu[i_beta,i_ht_lag,i_p,i_x,i_ht] = inv_mu[i_x]

@njit(parallel=True)
def evaluate_ft(par,sol,t):
    """ evaluate the intermediary problem for refinancers """

    # fill out these arrays
    ft_inv_v = sol.ft_inv_v[t]
    ft_inv_mu = sol.ft_inv_mu[t]

    # a. states
    for i_p in prange(par.Np):
        for i_beta in range(par.Nbeta):
            for i_h_lag in range(par.Nh):
                
                # b. alllocate
                inv_v = np.zeros(par.Nx)
                inv_mu = np.zeros(par.Nx)

                # c. choices
                i_h = i_h_lag # forced
                for iota in range(par.Niota):
                    for i_LTV in range(par.NLTV):
                        
                        # i. implied b and z
                        LTV = par.grid_LTV[i_LTV]
                        b = np.empty(par.Nx)
                        z = np.empty(par.Nx)
                        for i_x in range(par.Nx):
                            b[i_x],z[i_x] = misc.bz_func(i_h,LTV,par.grid_x[i_x],False,par)

                        # ii. interpolate
                        prep = linear_interp.interp_1d_prep(par.Nx)
                        
                        inv_vbar = sol.o_inv_vbar[t,i_beta,iota,i_h,i_p,i_LTV,:]
                        linear_interp.interp_1d_vec_mon(prep,par.grid_z,inv_vbar,z,inv_v)
                        
                        inv_mu_bar = sol.o_inv_mubar[t,i_beta,iota,i_h,i_p,i_LTV,:]
                        linear_interp.interp_1d_vec_mon_rep(prep,par.grid_z,inv_mu_bar,z,inv_mu)

                        # iii. save
                        for i_x in range(par.Nx):

                            if z[i_x] <= 0 or b[i_x] > par.kappa_p*par.Gamma[t]*par.grid_p[i_p]:
                        
                                ft_inv_v[i_beta,i_h,i_p,i_x,iota,i_LTV] = 0.0
                                ft_inv_mu[i_beta,i_h,i_p,i_x,iota,i_LTV] = 0.0
                            
                            else:
                                
                                ft_inv_v[i_beta,i_h,i_p,i_x,iota,i_LTV] = inv_v[i_x]
                                ft_inv_mu[i_beta,i_h,i_p,i_x,iota,i_LTV] = inv_mu[i_x]

@njit(parallel=True)
def evaluate_bt(par,sol,t):
    """ evaluate the intermediary problem for buyers """

    # fill out these arrays
    bt_inv_v = sol.bt_inv_v[t]
    bt_inv_mu = sol.bt_inv_mu[t]

    # a. states
    for i_p in prange(par.Np):
        for i_beta in range(par.Nbeta):
        
            # b. allocate
            inv_v = np.zeros(par.Nx)
            inv_mu = np.zeros(par.Nx)

            # c. choices
            for iota in range(par.Niota):
                for i_h in range(par.Nh):
                    for i_LTV in range(par.NLTV):
                            
                        # i. implied b and z
                        LTV = par.grid_LTV[i_LTV]
                        b = np.empty(par.Nx)
                        z = np.empty(par.Nx)
                        for i_x in range(par.Nx):
                            b[i_x],z[i_x] = misc.bz_func(i_h,LTV,par.grid_x[i_x],True,par)

                        # ii. interpolate
                        prep = linear_interp.interp_1d_prep(par.Nx)

                        inv_vbar = sol.o_inv_vbar[t,i_beta,iota,i_h,i_p,i_LTV,:]
                        linear_interp.interp_1d_vec_mon(prep,par.grid_z,inv_vbar,z,inv_v)
                        
                        inv_mu_bar = sol.o_inv_mubar[t,i_beta,iota,i_h,i_p,i_LTV,:]
                        linear_interp.interp_1d_vec_mon_rep(prep,par.grid_z,inv_mu_bar,z,inv_mu)

                        # iii. save
                        for i_x in range(par.Nx):

                            if z[i_x] <= 0 or b[i_x] > par.kappa_p*par.Gamma[t]*par.grid_p[i_p]:

                                bt_inv_v[i_beta,i_p,i_x,iota,i_h,i_LTV] = 0.0
                                bt_inv_mu[i_beta,i_p,i_x,iota,i_h,i_LTV] = 0.0
                            
                            else:

                                bt_inv_v[i_beta,i_p,i_x,iota,i_h,i_LTV] = inv_v[i_x]
                                bt_inv_mu[i_beta,i_p,i_x,iota,i_h,i_LTV] = inv_mu[i_x]

#########
# final #
#########

@njit
def update(par,i,j,inv_v0,inv_v1,inv_mu0,inv_mu1,inv_v,inv_mu,wx,valid,v,p,mu,do_mu,do_interp=True):
    """ update v and p """

    # a. interpolate
    if do_interp:
        inv_v[i:j] = inv_v0 + wx*(inv_v1-inv_v0)
        if do_mu:
            inv_mu[i:j] = inv_mu0 + wx*(inv_mu1-inv_mu0)

    # b. valid
    if do_mu:
        valid[i:j] = (inv_v[i:j] > 0) & (inv_mu[i:j] > 0) 
    else:
        valid[i:j] = inv_v[i:j] > 0

    # c. inverse
    v[i:j][~valid[i:j]] = -np.inf
    v[i:j][valid[i:j]] = -1.0/inv_v[i:j][valid[i:j]]

    if do_mu:
        mu[i:j][~valid[i:j]] = np.nan
        mu[i:j][valid[i:j]] = 1.0/inv_mu[i:j][valid[i:j]]

    # d. logsum
    if np.any(valid[i:j]):
        
        _logsum = logsum_and_choice_probabilities(v[i:j],par.sigma,p[i:j])
        Ev = np.nansum(v[i:j]*p[i:j])

    else:
        
        p[i:j] = np.nan
        Ev = -np.inf

    return Ev

@njit
def find_renter_choice(par,sol,t,i_beta,i_ht_lag,i_p,a_lag,
                       inv_v,inv_mu,v,mu,p,valid,do_mu=True):
    """ find renter choice - used in both solution and simulation """

    v_agg = np.zeros(2)
    p_agg = np.zeros(2)

    # a. x
    iota_lag = -1
    i_h_lag = -1
    LTV_lag = np.nan
    
    _m,x,_LTV = misc.mx_func(t,iota_lag,i_h_lag,i_p,LTV_lag,a_lag,par)

    i_x = linear_interp.binary_search(0,par.Nx,par.grid_x,x)
    wx = (x-par.grid_x[i_x])/(par.grid_x[i_x+1]-par.grid_x[i_x])

    # b. choices

    # 1. renter
    i = 0
    j = i + par.Nrt
    inv_v0 = sol.rt_inv_v[t,i_beta,i_ht_lag,i_p,i_x,:].ravel()
    inv_v1 = sol.rt_inv_v[t,i_beta,i_ht_lag,i_p,i_x+1,:].ravel()
    inv_mu0 = sol.rt_inv_mu[t,i_beta,i_ht_lag,i_p,i_x,:]
    inv_mu1 = sol.rt_inv_mu[t,i_beta,i_ht_lag,i_p,i_x+1,:]

    v_agg[0] = update(par,i,j,inv_v0,inv_v1,inv_mu0,inv_mu1,inv_v,inv_mu,wx,valid,v,p,mu,do_mu)
    i_rt = i
    j_rt = j

    # 2. buyer
    i = j
    j = i + par.Nbt # = par.Ncr
    inv_v0 = sol.bt_inv_v[t,i_beta,i_p,i_x,:,:,:].ravel()
    inv_v1 = sol.bt_inv_v[t,i_beta,i_p,i_x+1,:,:,:].ravel()
    inv_mu0 = sol.bt_inv_mu[t,i_beta,i_p,i_x,:,:,:].ravel()
    inv_mu1 = sol.bt_inv_mu[t,i_beta,i_p,i_x+1,:,:,:].ravel()

    v_agg[1] = update(par,i,j,inv_v0,inv_v1,inv_mu0,inv_mu1,inv_v,inv_mu,wx,valid,v,p,mu,do_mu)
    i_bt = i
    j_bt = j

    # c. aggregate
    if np.any(~np.isinf(v_agg)):

        _logsum = logsum_and_choice_probabilities(v_agg,par.sigma_agg,p_agg)

        p[i_rt:j_rt] *= p_agg[0]
        p[i_bt:j_bt] *= p_agg[1]

        Ev = np.nansum(p*v)

        if do_mu:
            Emu = np.nansum(p*mu)
        else:
            Emu = np.nan

    else:

        p[:] = np.nan
        Ev = np.nan
        Emu = np.nan

    return Ev,Emu

@njit(parallel=True)
def solve_renters(par,sol,t):
    """ solve the full problem for renters """

    # fill out these arrays
    r_v = sol.r_v[t]
    r_mu = sol.r_mu[t]
    r_d = sol.r_d[t]

    # a. parallel state
    for i_a_lag in prange(par.Na):

        # b. allocate (Ncr = number of discrete choices for renters)
        inv_v = np.zeros(par.Ncr)
        inv_mu = np.zeros(par.Ncr)
        v = np.zeros(par.Ncr)
        mu = np.zeros(par.Ncr)
        p = np.zeros(par.Ncr)
        valid = np.zeros(par.Ncr,dtype=np.bool_)

        # c. inner states
        for i_beta in range(par.Nbeta):
            for i_ht_lag in range(par.Nht):
                for i_p in range(par.Np):

                    # i. unpack 
                    a_lag = par.grid_a[i_a_lag]        

                    # ii. evaluate
                    Ev,Emu = find_renter_choice(par,sol,t,i_beta,i_ht_lag,i_p,a_lag,inv_v,inv_mu,v,mu,p,valid)

                    r_v[i_beta,i_ht_lag,i_p,i_a_lag] = Ev
                    r_mu[i_beta,i_ht_lag,i_p,i_a_lag] = Emu

                    # extra: best discrete choice
                    if np.any(valid):
                        i_c = np.argmax(inv_v)
                        r_d[i_beta,i_ht_lag,i_p,i_a_lag] = par.r_d[i_c]
                    else:
                        r_d[i_beta,i_ht_lag,i_p,i_a_lag] = -1

@njit
def find_owner_choice(par,sol,t,i_beta,iota_lag,i_h_lag,i_p,i_LTV_lag,LTV_lag,a_lag,                       
                      inv_v,inv_mu,v,mu,p,valid,do_mu=True):
    """ find renter choice - used in both solution and simulation """                      

    v_agg = np.zeros(4)
    p_agg = np.zeros(4)

    # a. x
    m,x,LTV = misc.mx_func(t,iota_lag,i_h_lag,i_p,LTV_lag,a_lag,par)

    i_x = linear_interp.binary_search(0,par.Nx,par.grid_x,x)
    wx = (x-par.grid_x[i_x])/(par.grid_x[i_x+1]-par.grid_x[i_x])

    # b. choices

    # 1. renter
    i = 0
    j = i + par.Nrt
    inv_v0 = sol.rt_inv_v[t,i_beta,par.Nht,i_p,i_x,:].ravel()
    inv_v1 = sol.rt_inv_v[t,i_beta,par.Nht,i_p,i_x+1,:].ravel()
    inv_mu0 = sol.rt_inv_mu[t,i_beta,par.Nht,i_p,i_x,:]
    inv_mu1 = sol.rt_inv_mu[t,i_beta,par.Nht,i_p,i_x+1,:]

    v_agg[0] = update(par,i,j,inv_v0,inv_v1,inv_mu0,inv_mu1,inv_v,inv_mu,wx,valid,v,p,mu,do_mu)
    i_rt = i
    j_rt = j
    
    # 2. buyer
    i = j
    j = i + par.Nbt

    inv_v0 = sol.bt_inv_v[t,i_beta,i_p,i_x,:,:,:].ravel()
    inv_v1 = sol.bt_inv_v[t,i_beta,i_p,i_x+1,:,:,:].ravel()
    inv_mu0 = sol.bt_inv_mu[t,i_beta,i_p,i_x,:,:,:].ravel()
    inv_mu1 = sol.bt_inv_mu[t,i_beta,i_p,i_x+1,:,:,:].ravel()

    v_agg[1] = update(par,i,j,inv_v0,inv_v1,inv_mu0,inv_mu1,inv_v,inv_mu,wx,valid,v,p,mu,do_mu)
    i_bt = i
    j_bt = j   

    # 3. refinancer
    i = j
    j = i + par.Nft

    inv_v0 = sol.ft_inv_v[t,i_beta,i_h_lag,i_p,i_x,:,:].ravel()
    inv_v1 = sol.ft_inv_v[t,i_beta,i_h_lag,i_p,i_x+1,:,:].ravel()
    inv_mu0 = sol.ft_inv_mu[t,i_beta,i_h_lag,i_p,i_x,:,:].ravel()
    inv_mu1 = sol.ft_inv_mu[t,i_beta,i_h_lag,i_p,i_x+1,:,:].ravel()

    v_agg[2] = update(par,i,j,inv_v0,inv_v1,inv_mu0,inv_mu1,inv_v,inv_mu,wx,valid,v,p,mu,do_mu)
    i_ft = i
    j_ft = j

    # 4. keeper
    i = j
    j = i + par.Nkt

    iota = np.fmax(iota_lag-par.Delta_iota,0)
    inv_vbar_2d = sol.o_inv_vbar[t,i_beta,iota,i_h_lag,i_p,:,:]
    inv_v[i:j] = linear_interp.interp_2d(par.grid_LTV,par.grid_z,inv_vbar_2d,LTV,m)

    if do_mu:
        inv_mubar_2d = sol.o_inv_mubar[t,i_beta,iota,i_h_lag,i_p,:,:]
        inv_mu[i:j] = linear_interp.interp_2d(par.grid_LTV,par.grid_z,inv_mubar_2d,LTV,m)

    v_agg[3] = update(par,i,j,np.zeros(0),np.zeros(0),np.zeros(0),np.zeros(0),inv_v,inv_mu,np.zeros(0),valid,v,p,mu,do_mu,do_interp=False)
    i_kt = i
    j_kt = j

    # c. aggregate
    if np.any(~np.isinf(v_agg)):

        _logsum = logsum_and_choice_probabilities(v_agg[:3],par.sigma_agg,p_agg[:3])

        p[i_rt:j_rt] *= p_agg[0]
        p[i_bt:j_bt] *= p_agg[1]
        p[i_ft:j_ft] *= p_agg[2]

        Ev_adj = np.nansum(p[i_rt:j_ft]*v[i_rt:j_ft])

        if v_agg[3] > Ev_adj:
            p[i_rt:j_ft] = 0.0
            p[i_kt:j_kt] = 1.0
        else:
            p[i_kt:j_kt] = 0.0

        Ev = np.nansum(p*v)

        if do_mu:
            Emu = np.nansum(p*mu)
        else:
            Emu = np.nan

    else:

        p[:] = np.nan
        Ev = np.nan
        Emu = np.nan

    return Ev,Emu

@njit(parallel=True)
def solve_owners(par,sol,t):
    """ solve the full problem for owners """

    # fill out these arrays
    o_v = sol.o_v[t]
    o_mu = sol.o_mu[t]
    o_d = sol.o_d[t]

    # a. parallel state
    for i_a_lag in prange(par.Na):
        
        # b. allocate (Nco = number of discrete choices for owners)
        inv_v = np.zeros(par.Nco)
        inv_mu = np.zeros(par.Nco)
        v = np.zeros(par.Nco)
        mu = np.zeros(par.Nco)
        p = np.zeros(par.Nco)
        valid = np.zeros(par.Nco,dtype=np.bool_)

        # c. inner states
        for i_beta in range(par.Nbeta):
            for iota_lag in range(par.Niota):
                for i_h_lag in range(par.Nh):
                    for i_p in range(par.Np):
                        for i_LTV_lag in range(par.NLTV):
                            
                            # i. unpack
                            a_lag = par.grid_a[i_a_lag]
                            LTV_lag = par.grid_LTV[i_LTV_lag]

                            # ii. evaluate
                            Ev,Emu = find_owner_choice(par,sol,t,i_beta,iota_lag,i_h_lag,i_p,i_LTV_lag,LTV_lag,a_lag,inv_v,inv_mu,v,mu,p,valid)

                            # iii. save
                            o_v[i_beta,iota_lag,i_h_lag,i_p,i_LTV_lag,i_a_lag] = Ev
                            o_mu[i_beta,iota_lag,i_h_lag,i_p,i_LTV_lag,i_a_lag] = Emu

                            # extra: best discrete choice
                            if np.any(valid):

                                i_c = np.argmax(inv_v)
                                o_d[i_beta,iota_lag,i_h_lag,i_p,i_LTV_lag,i_a_lag] = par.o_d[i_c]
                            
                            else:

                                o_d[i_beta,iota_lag,i_h_lag,i_p,i_LTV_lag,i_a_lag] = -1
