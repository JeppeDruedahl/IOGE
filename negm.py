# -*- coding: utf-8 -*-
"""negm

Apply the nested endogenous grid method.

"""

import numpy as np
from numba import njit, prange

# consav
from consav import linear_interp # for linear interpolation
from consav import upperenvelope

# local modules
import utility

negm_upperenvelope = upperenvelope.create(utility.func,use_inv_w=True)

@njit(parallel=True)
def solve_renters(par,sol,t):
    """solve the renter problem with the endogenous grid method"""

    # unpack
    r_inv_vbar = sol.r_inv_vbar[t]
    r_cbar = sol.r_cbar[t]
    r_inv_mubar = sol.r_inv_mubar[t]

    # parallel loop
    for i_p in prange(par.Np):
        
        # a. temporary containers
        q_c = np.zeros(par.Na)
        q_z = np.zeros(par.Na)
        v_ast_vec = np.zeros(par.Nz-1)
        
        # b. outer states
        owner = False
        for i_beta in range(par.Nbeta):
            for i_ht in range(par.Nht):
                
                ht = par.grid_ht[i_ht] # rental housing

                # i. use euler equation
                for i_a in range(par.Na):
                    q_c[i_a] = utility.inv_marg_func(sol.r_q[t,i_beta,i_ht,i_p,i_a],ht,owner,par)
                    q_z[i_a] = par.grid_a[i_a] + q_c[i_a]

                # ii. upperenvelope
                r_cbar[i_beta,i_ht,i_p,0] = 0.0
                negm_upperenvelope(par.grid_a,q_z,q_c,sol.r_inv_w[t,i_beta,i_ht,i_p,:],
                                par.grid_z[1:],r_cbar[i_beta,i_ht,i_p,1:],v_ast_vec,ht,owner,par) 

                # iii. value
                r_inv_vbar[i_beta,i_ht,i_p,0] = 0.0
                r_inv_vbar[i_beta,i_ht,i_p,1:] = -1.0/v_ast_vec

                # iv. marginal utlity
                r_inv_mubar[i_beta,i_ht,i_p,0] = 0.0
                for i_z in range(1,par.Nz):
                    c = r_cbar[i_beta,i_ht,i_p,i_z]
                    r_inv_mubar[i_beta,i_ht,i_p,i_z] = 1.0/utility.marg_func(c,ht,owner,par)

@njit(parallel=True)
def solve_owners(par,sol,t):
    """solve the owner problem with the endogenous grid method"""

    # unpack
    o_inv_vbar = sol.o_inv_vbar[t]
    o_cbar = sol.o_cbar[t]
    o_inv_mubar = sol.o_inv_mubar[t]

    # parallel loop
    for i_p in prange(par.Np):
        
        # a. temporary containers
        q_c = np.zeros(par.Na)
        q_z = np.zeros(par.Na)
        v_ast_vec = np.zeros(par.Nz-1)

        # c. outer states
        owner = True
        for i_beta in range(par.Nbeta):
            for iota in range(par.Niota):
                for i_h in range(par.Nh):
                    for i_LTV in range(par.NLTV):
                    
                        h = par.grid_h[i_h] # owned housing

                        # i. use euler equation
                        for i_a in range(par.Na):
                            q_c[i_a] = utility.inv_marg_func(sol.o_q[t,i_beta,iota,i_h,i_p,i_LTV,i_a],h,owner,par)
                            q_z[i_a] = par.grid_a[i_a] + q_c[i_a]
            
                        # ii. upperenvelope
                        o_cbar[i_beta,iota,i_h,i_p,i_LTV,0] = 0.0
                        negm_upperenvelope(par.grid_a,q_z,q_c,sol.o_inv_w[t,i_beta,iota,i_h,i_p,i_LTV,:],
                                        par.grid_z[1:],o_cbar[i_beta,iota,i_h,i_p,i_LTV,1:],v_ast_vec,h,owner,par)        

                        # iii. value
                        o_inv_vbar[i_beta,iota,i_h,i_p,i_LTV,0] = 0.0
                        o_inv_vbar[i_beta,iota,i_h,i_p,i_LTV,1:] = -1.0/v_ast_vec

                        # iv. marginal_utility
                        o_inv_mubar[i_beta,iota,i_h,i_p,i_LTV,0] = 0.0
                        for i_z in range(1,par.Nz):
                            c = o_cbar[i_beta,iota,i_h,i_p,i_LTV,i_z]
                            o_inv_mubar[i_beta,iota,i_h,i_p,i_LTV,i_z] = 1.0/utility.marg_func(c,h,owner,par)