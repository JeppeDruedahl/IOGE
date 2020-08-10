# -*- coding: utf-8 -*-
"""post_decision

Compute w and q.

"""

import numpy as np
from numba import njit, prange

@njit(parallel=True)
def compute_wq_renters(par,sol,t):
    """ compute the post-decision functions w and q """

    # unpack
    r_w = sol.r_w[t]
    r_inv_w = sol.r_inv_w[t]
    r_q = sol.r_q[t]

    retired_next_period = t+1 >= par.TR
    # example: TR = 40, then if t=39 the condition is fulfilled,
    # i.e. it is fulfilled in the 40th period, 0,1,...,38,39

    for i_p in prange(par.Np):
        for i_beta in range(par.Nbeta):
            for i_ht in range(par.Nht):

                # i. initialize w and q
                r_w[i_beta,i_ht,i_p,:] = 0
                r_q[i_beta,i_ht,i_p,:] = 0

                # ii. compute
                if t == par.T-1:
                    
                    bequest = par.grid_a
                    r_w[i_beta,i_ht,i_p,:] = par.nu*(bequest+par.zeta)**(1.0-par.rho)/(1.0-par.rho)
                    r_q[i_beta,i_ht,i_p,:] = par.nu*(bequest+par.zeta)**(-par.rho)

                else:

                    for i_p_plus in range(par.Np):
                        
                        # transition probability and index            
                        trans_p = 1 if retired_next_period else par.trans_p[i_p,i_p_plus]
                        i_p_plus__ = i_p if retired_next_period else i_p_plus            
                        i_p_plus_ = np.int_(i_p_plus__)

                        # post-decision value function
                        w = par.grid_beta[i_beta]*sol.r_v[t+1,i_beta,i_ht,i_p_plus_,:]
                        r_w[i_beta,i_ht,i_p,:] += trans_p*w
                        
                        # post-decision marginal value-of-cash
                        q = par.grid_beta[i_beta]*(1+par.ra)*sol.r_mu[t+1,i_beta,i_ht,i_p_plus_,:]
                        r_q[i_beta,i_ht,i_p,:] += trans_p*q

                        # no risk for retired
                        if retired_next_period: break

                # iii. negative inverse
                r_inv_w[i_beta,i_ht,i_p,:] = -1.0/r_w[i_beta,i_ht,i_p,:]

@njit(parallel=True)
def compute_wq_owners(par,sol,t):
    """ compute the post-decision functions w and q """

    # unpack
    o_w = sol.o_w[t]
    o_inv_w = sol.o_inv_w[t]
    o_q = sol.o_q[t]

    retired_next_period = t+1 >= par.TR 
    # example: TR = 40, then if t=39 the condition is fulfilled,
    # i.e. it is fulfilled in the 40th period, 0,1,...,38,39

    for i_p in prange(par.Np):
        for i_beta in range(par.Nbeta):
            for iota in range(par.Niota):
                for i_h in range(par.Nh):
                    for i_LTV in range(par.NLTV):
                        
                        # i. w and q
                        o_w[i_beta,iota,i_h,i_p,i_LTV,:] = 0.0
                        o_q[i_beta,iota,i_h,i_p,i_LTV,:] = 0.0

                        # ii. compute
                        if t == par.T-1:
                            
                            bequest = par.grid_a + (1.0-par.grid_LTV[i_LTV])*par.ph*par.grid_h[i_h]
                            o_w[i_beta,iota,i_h,i_p,i_LTV,:] = par.nu*(bequest+par.zeta)**(1.0-par.rho)/(1.0-par.rho)
                            o_q[i_beta,iota,i_h,i_p,i_LTV,:] = par.nu*(bequest+par.zeta)**(-par.rho)
                        
                        else:

                            for i_p_plus in range(par.Np):
                                
                                # transition probability and index
                                trans_p = 1 if retired_next_period else par.trans_p[i_p,i_p_plus]
                                i_p_plus__ = i_p if retired_next_period else i_p_plus
                                i_p_plus_ = np.int_(i_p_plus__)

                                # post-decision value function
                                w = par.grid_beta[i_beta]*sol.o_v[t+1,i_beta,iota,i_h,i_p_plus_,i_LTV,:]
                                o_w[i_beta,iota,i_h,i_p,i_LTV,:] += trans_p*w

                                # post-decision marginal value-of-cash
                                q = par.grid_beta[i_beta]*(1+par.ra)*sol.o_mu[t+1,i_beta,iota,i_h,i_p_plus_,i_LTV,:]
                                o_q[i_beta,iota,i_h,i_p,i_LTV,:] += trans_p*q

                                # no risk for retired
                                if retired_next_period: break

                        # iii. negative inverse
                        o_inv_w[i_beta,iota,i_h,i_p,i_LTV,:] = -1.0/o_w[i_beta,iota,i_h,i_p,i_LTV,:]