
# -*- coding: utf-8 -*-
"""simulate

Simulate

"""

import numpy as np
from numba import njit, prange

import consav
from consav import linear_interp
import misc
from nvfi import find_renter_choice, find_owner_choice

@njit
def find_choices(par,sol,sim,t,i_beta,iota_lag,i_ht_lag,i_h_lag,i_p,LTV_lag,a_lag,pi_c,
                 r_inv_v,r_inv_mu,r_v,r_mu,r_p,r_valid,r_pcs,o_inv_v,o_inv_mu,o_v,o_mu,o_p,o_valid,o_pcs):
    """ find choices in simulation """
    
    m,x,LTV = misc.mx_func(t,iota_lag,i_h_lag,i_p,LTV_lag,a_lag,par)

    # discrete
    if i_h_lag == -1: # renter

        _Ev,_Emu = find_renter_choice(par,sol,t,i_beta,i_ht_lag,i_p,a_lag,
                                      r_inv_v,r_inv_mu,r_v,r_mu,r_p,r_valid,do_mu=False)

        # choice
        Nc = np.sum(r_valid)
        r_pcs[:] = np.cumsum(r_p)
        i_c = consav.misc.choice(pi_c,r_pcs)

        # unpack
        d = par.r_d[i_c]
        i_ht = par.r_i_ht[i_c]
        iota = par.r_iota[i_c]
        i_h = par.r_i_h[i_c]
        i_LTV = par.r_i_LTV[i_c]

    else: # owners

        i_LTV_lag = -1 # we are not on grid
        _Ev,_Emu = find_owner_choice(par,sol,t,i_beta,iota_lag,i_h_lag,i_p,i_LTV_lag,LTV_lag,a_lag,
                                     o_inv_v,o_inv_mu,o_v,o_mu,o_p,o_valid,do_mu=False)

        # choice
        Nc = np.sum(o_valid)
        o_pcs[:] = np.cumsum(o_p)
        i_c = consav.misc.choice(pi_c,o_pcs)

        # unpack
        d = par.o_d[i_c]
        i_ht = par.o_i_ht[i_c]
        iota = par.o_iota[i_c]
        i_h = par.o_i_h[i_c]
        i_LTV = par.o_i_LTV[i_c]

    # continous
    if d == 0: # rt
        
        LTV = np.nan
        
        z = x-par.rh*par.grid_ht[i_ht]
        if i_h_lag == -1 or (not i_ht == i_ht_lag):
            z -= par.tau_ht

        c = linear_interp.interp_1d(par.grid_z,sol.r_cbar[t,i_beta,i_ht,i_p,:],z)

    elif d == 3: # kt

        z = m
        i_ht = -1
        i_h = i_h_lag
        iota = np.fmax(iota_lag-par.Delta_iota,0)                  
        c = linear_interp.interp_2d(par.grid_LTV,par.grid_z,sol.o_cbar[t,i_beta,iota,i_h,i_p,:,:],LTV,z)

    else: # ft or bt

        LTV = par.grid_LTV[i_LTV]

        if d == 2: # ft
            i_h = i_h_lag # overwriting
            mover = False
        else:
            mover = True
        
        _b,z = misc.bz_func(i_h,LTV,x,mover,par)
        c = linear_interp.interp_1d(par.grid_z,sol.o_cbar[t,i_beta,iota,i_h,i_p,i_LTV,:],z)

    a = np.fmax(z-c,0.0)

    return d,Nc,i_ht,iota,i_h,LTV,c,a,x

@njit(parallel=True)
def simulate(par,sol,sim):
    """ simulate model """

    # unpack sim
    sim_d = sim.d
    sim_iota = sim.iota
    sim_i_h = sim.i_h
    sim_i_ht = sim.i_ht
    sim_i_p = sim.i_p
    sim_LTV = sim.LTV
    sim_x = sim.x
    sim_Nc = sim.Nc
    sim_c = sim.c
    sim_a = sim.a

    # simulate
    for i in prange(par.simN):

        i_beta = sim.i_beta[i]

        r_inv_v = np.empty(par.Ncr)
        r_inv_mu = np.empty(par.Ncr)
        r_v = np.empty(par.Ncr)
        r_mu = np.empty(par.Ncr)
        r_p = np.empty(par.Ncr)
        r_valid = np.empty(par.Ncr,dtype=np.bool_)
        r_pcs = np.empty(par.Ncr)

        o_inv_v = np.empty(par.Nco)
        o_inv_mu = np.empty(par.Nco)
        o_v = np.empty(par.Nco)
        o_mu = np.empty(par.Nco)
        o_p = np.empty(par.Nco)
        o_valid = np.empty(par.Nco,dtype=np.bool_)
        o_pcs = np.empty(par.Nco)

        for t in range(par.T):

            # a. states
            if t == 0:
                iota_lag = -1
                i_ht_lag = 0
                i_h_lag = -1
                sim_i_p[i,t] = consav.misc.choice(sim.pi_p[i,t],par.ergodic_cs_p)
                a_lag = sim.a0[i]
                LTV_lag = np.nan
            else:
                iota_lag = sim_iota[i,t-1]
                i_ht_lag = sim_i_ht[i,t-1]
                i_h_lag = sim_i_h[i,t-1]
                LTV_lag = sim_LTV[i,t-1]
                a_lag = sim_a[i,t-1]
                i_p_lag = sim_i_p[i,t-1]
                if t <= par.TR-1:
                    sim_i_p[i,t] = consav.misc.choice(sim.pi_p[i,t],par.trans_cs_p[i_p_lag,:])       
                else:
                    sim_i_p[i,t] = sim_i_p[i,t-1]

            i_p = sim.i_p[i,t]
            
            # b. choices                
            d,Nc,i_ht,iota,i_h,LTV,c,a,x = find_choices(par,sol,sim,t,i_beta,iota_lag,i_ht_lag,i_h_lag,i_p,LTV_lag,a_lag,sim.pi_c[i,t],
                                                        r_inv_v,r_inv_mu,r_v,r_mu,r_p,r_valid,r_pcs,o_inv_v,o_inv_mu,o_v,o_mu,o_p,o_valid,o_pcs)

            # assert d >= 0
            # assert i_ht >= 0 or i_h >= 0
            # assert c >= 0
            # assert a >= 0, a

            # c. save
            sim_x[i,t] = x
            sim_d[i,t] = d
            sim_i_ht[i,t] = i_ht
            sim_iota[i,t] = iota
            sim_i_h[i,t] = i_h
            sim_LTV[i,t] = LTV
            sim_Nc[i,t] = Nc
            sim_c[i,t] = c
            sim_a[i,t] = a

def inspect(model,t,i,i_beta,iota_lag,i_ht_lag,i_h_lag,i_p,LTV_lag,a_lag):

    par = model.par
    sim = model.sim
    sol = model.sol
    
    # a. allocate
    r_inv_v = np.empty(par.Ncr)
    r_inv_mu = np.empty(par.Ncr)
    r_v = np.empty(par.Ncr)
    r_mu = np.empty(par.Ncr)
    r_p = np.empty(par.Ncr)
    r_valid = np.empty(par.Ncr,dtype=np.bool_)
    r_pcs = np.empty(par.Ncr)

    o_inv_v = np.empty(par.Nco)
    o_inv_mu = np.empty(par.Nco)
    o_v = np.empty(par.Nco)
    o_mu = np.empty(par.Nco)
    o_p = np.empty(par.Nco)
    o_valid = np.empty(par.Nco,dtype=np.bool_)
    o_pcs = np.empty(par.Nco)

    # b. find choices
    _d,_Nc,_i_ht,_iota,_i_h,_LTV,c,a,x = find_choices(par,sol,sim,t,i_beta,iota_lag,i_ht_lag,i_h_lag,i_p,LTV_lag,a_lag,sim.pi_c[i,t],
                                                      r_inv_v,r_inv_mu,r_v,r_mu,r_p,r_valid,r_pcs,o_inv_v,o_inv_mu,o_v,o_mu,o_p,o_valid,o_pcs)    

    # c. output
    if i_h_lag == -1:
        return r_v,r_p,c,a,x,r_inv_v
    else:
        return o_v,o_p,c,a,x,o_inv_v