# -*- coding: utf-8 -*-
"""HousingModel

Main model class

"""

import time
import numpy as np
import pandas as pd

# consav package
from consav import ModelClass
from consav.misc import elapsed, nonlinspace, markov_rouwenhorst

# local
import post_decision
import negm
import nvfi
import simulate

def solve_model(model,t_min=0,do_print=True):
    """ solve the model """

    par = model.par
    sol = model.sol

    t0_outer = time.time()

    # a. re-set up grids
    t0 = time.time()
    model.create_grids()
    
    if do_print: print(f'setup grids in {elapsed(t0)}')
    
    # c. time loop
    for t in reversed(range(t_min,par.T)):
        
        t0 = time.time()

        # i. post-decions
        t0_pd = time.time()
        post_decision.compute_wq_renters(par,sol,t)
        post_decision.compute_wq_owners(par,sol,t)
        t_pd = elapsed(t0_pd)

        # ii. negm
        t0_negm = time.time()
        negm.solve_renters(par,sol,t)       
        negm.solve_owners(par,sol,t)       
        t_negm = elapsed(t0_negm)

        # iii. evaluate values of each discrete choice
        t0_evaluate = time.time()
        nvfi.evaluate_rt(par,sol,t)
        nvfi.evaluate_ft(par,sol,t)
        nvfi.evaluate_bt(par,sol,t)
        t_evaluate = elapsed(t0_evaluate)

        # iv. final nvfi
        t0_nvfi = time.time()
        nvfi.solve_renters(par,sol,t)
        t_nvfi_r = elapsed(t0_nvfi)
        
        t0_nvfi = time.time()
        nvfi.solve_owners(par,sol,t)
        t_nvfi_o = elapsed(t0_nvfi)

        if do_print: 
            msg = f't = {t:2d} solved in {elapsed(t0)}'
            msg += f'[pd: {t_pd}, negm: {t_negm}, evaluate: {t_evaluate}, nvfi_r: {t_nvfi_r}, nvfi_o: {t_nvfi_o}]'
            print(msg)

    if do_print: print(f'model solved in {elapsed(t0_outer)}')

def simulate_model(model,do_print=True,seed=1986):
    """ simulate the model """

    if not seed is None: np.random.seed(seed)

    par = model.par
    sol = model.sol
    sim = model.sim

    t0_outer = time.time()

    # a. draw random numbers
    sim.i_beta[:] = np.random.choice(par.Nbeta,size=par.simN) # preferences
    sim.a0[:] = np.random.gamma(par.a0_shape,par.a0_scale,size=par.simN) # initial assets

    sim.pi_p[:] = np.random.uniform(size=(par.simN,par.T)) # income process
    sim.pi_c[:] = np.random.uniform(size=(par.simN,par.T)) # discrete choice

    # b. simulate
    simulate.simulate(par,sol,sim)

    if do_print: print(f'model simulated in {elapsed(t0_outer)}')    

# class
class HousingModelClass(ModelClass):
    
    def setup(self):
        """ set baseline parameters in .par """

        par = self.par

        # specify list over parameters, which are allowed not to be floats
        self.not_float_list = ['T','TR','age_min','t_min','Delta_iota',
                               'Nbeta','Na','Niota','Nh','Nht','Np','NLTV','Nm','Nx','Nz',
                               'Nrt','Nbt','Nft','Nkt','Ncr','Nco','do_sim','simN']

        # a. economic parameters

        # life-cycle    
        par.T = 55 # life-span from age_min
        par.TR = 37 # working-life-span from age_min
        par.age_min = 25 # only used in figures
        par.t_min = 0 # used when solving
        
        # income
        par.rho_p = 0.99 # persistence of income shocks 
        par.sigma_p = 0.30 # std. of income shocks

        par.G = np.ones(par.T) # age-specific growth factors of income
        par.G[:20] = 1.066
        par.G[20:par.TR] = 1.015
        par.G[par.TR:] = 0.96
        par.retirement_drop = 1.00 # drop in income at retirement

        # assets and housing
        par.ra = 0.035 # return on liquid assets
        par.rm = 0.040 # mortgage interest rate
        par.rb = 0.070 # bank loan interest rate

        par.ph = 1.000 # housing price
        par.rh = 0.045 # rental price

        par.delta = 0.0075 # mortgage interest only spread
        par.gamma_m = 0.050 # mortgage repayment rate
        par.gamma_b = 0.100 # bank loan repayment rate
        
        par.tau_f = 0.100 # loan refinancing cost
        par.tau_h = 0.200 # moving-in cost for owners
        par.tau_ht = 0.010 # moving-in cost for renters
        
        par.kappa_p = 4.00 # loan-to-income ratio
        par.kappa_h = 0.95 # loan-to-value ratio
        par.kappa_h_mortgage = 0.80 # loan-to-value ratio (mortgage)

        par.grid_h = np.array([2.0,4.0,6.0,8.0,10.0,15.0,20.0,25.0,30.0,35.0],dtype=np.float_) # housing choices 
        par.grid_ht = par.grid_h.copy()

        par.Niota = 2 # maximum interest only period
        par.Delta_iota = 0 # = 0 permanent interest only possible, else = 1

        # preferences
        par.beta_mean = 0.96
        par.beta_low = 0.85 
        par.beta_high = 0.99

        par.rho = 2.0 # CRRA parameter
        par.nu = 20.0 # bequest utility multiplicative scaling
        par.zeta = 8.0 # bequest utility additive scaling

        par.alpha = 0.70 # non-durable weight
        par.omega = 1.20 # homeowner bonus

        par.sigma = 0.025 # smoothing
        par.sigma_agg = 0.050 # smoothing

        # b. computational parameters
        par.Nbeta = 3 # grid for beta
        par.Np = 7 # grid for p
        
        par.NLTV = 20 # grid for LTV
        par.LTV_phi = 1.0 # 1 -> equally spaced, > 1 more points closer to kappa_p

        par.Na = 100 # grid for a
        par.a_min = 0.0
        par.a_max = 50.0
        par.a_phi = 1.25 # 1 -> equally spaced, > 1 more points closer to min

        par.Nx = 200 # grid for x
        par.x_min = 0.0
        par.x_max = 80.0
        par.x_phi = 1.25 # 1 -> equally spaced, > 1 more points closer to min

        par.Nz = 200 # grid for z
        par.z_min = 0.0
        par.z_max = 50.0
        par.z_phi = 1.25 # 1 -> equally spaced, > 1 more points closer to min

        # c. simulation parameters
        par.do_sim = True
        par.a0_shape = 0.1
        par.a0_scale = 5.0
        par.simN = 100_000

    def create_grids(self):
        """ create grids """

        par = self.par

        assert par.Delta_iota in [0,1]

        # a. states
        if par.Nbeta == 1:
            par.grid_beta = np.array([par.beta_mean])
        else:
            par.grid_beta = np.array([par.beta_low,par.beta_mean,par.beta_high])

        assert par.Nbeta == par.grid_beta.size

        par.grid_LTV = np.flip(par.kappa_h-nonlinspace(0.0,par.kappa_h,par.NLTV,par.LTV_phi))
        par.grid_a = nonlinspace(par.a_min,par.a_max,par.Na,par.a_phi)
        par.grid_z = nonlinspace(0,par.z_max,par.Nz,par.z_phi)
        par.grid_x = nonlinspace(0,par.x_max,par.Nx,par.x_phi)

        # inferred size of housing grids
        par.Nh = par.grid_h.size # owners
        par.Nht = par.grid_ht.size # renters
        
        # infered number of discrete choices
        par.Nrt = par.Nht # number of choices for renters
        par.Nkt = 1 # number of choices for keepers
        par.Nft = par.Niota*par.NLTV # number of choices for refinancers
        par.Nbt = par.Niota*par.Nh*par.NLTV # number of choices for buyers

        par.Ncr = par.Nht + par.Nbt # number of choices for lagged renters
        par.Nco = par.Nht + par.Nbt + par.Nft + par.Nkt # number of choices for lagged owners

        # renters
        par.r_i_ht = -1*np.ones(par.Ncr,dtype=np.int_)
        par.r_iota = -1*np.ones(par.Ncr,dtype=np.int_)
        par.r_i_h = -1*np.ones(par.Ncr,dtype=np.int_)
        par.r_i_LTV = -1*np.ones(par.Ncr,dtype=np.int_)
        par.r_d = -1*np.ones(par.Ncr,dtype=np.int_)

        # owners
        par.o_i_ht = -1*np.ones(par.Nco,dtype=np.int_)
        par.o_iota = -1*np.ones(par.Nco,dtype=np.int_)
        par.o_i_h = -1*np.ones(par.Nco,dtype=np.int_)
        par.o_i_LTV = -1*np.ones(par.Nco,dtype=np.int_)
        par.o_d = -1*np.ones(par.Nco,dtype=np.int_)

        # rt
        i = 0
        for i_ht in range(par.Nht):

            par.r_i_ht[i] = i_ht
            par.r_d[i] = 0

            par.o_i_ht[i] = i_ht
            par.o_d[i] = 0

            i += 1

        # bt
        for iota in range(par.Niota):
            for i_h in range(par.Nh):
                for i_LTV in range(par.NLTV):

                    par.r_iota[i] = iota
                    par.r_i_h[i] = i_h
                    par.r_i_LTV[i] = i_LTV
                    par.r_d[i] = 1

                    par.o_iota[i] = iota
                    par.o_i_h[i] = i_h
                    par.o_i_LTV[i] = i_LTV
                    par.o_d[i] = 1

                    i += 1
        # ft
        for iota in range(par.Niota):
            for i_LTV in range(par.NLTV):

                par.o_iota[i] = iota
                par.o_i_LTV[i] = i_LTV
                par.o_d[i] = 2

                i += 1

        # kt
        par.o_d[i] = 3
        
        # b. income
        out_ = markov_rouwenhorst(par.rho_p,par.sigma_p,par.Np)
        par.grid_p, par.trans_p, par.ergodic_p, par.trans_cs_p, par.ergodic_cs_p = out_
        
        par.Gamma = np.empty(par.T)
        for t in range(par.T):
            if t == 0: par.Gamma[t] = 1
            else: par.Gamma[t] = par.G[t]*par.Gamma[t-1]
            if t == par.TR: par.Gamma[t] *= par.retirement_drop
            
    def allocate(self):
        """ create grids and allocate memory for .par, .sol and .sim  """

        par = self.par
        sol = self.sol
        sim = self.sim

        # a. parameters
        self.create_grids()

        # b. solution
        
        # post-decison
        post_r_shape = (par.T,par.Nbeta,par.Nht,par.Np,par.Na)
        sol.r_q = np.nan*np.ones(post_r_shape)
        sol.r_w = np.nan*np.ones(post_r_shape)
        sol.r_inv_w = np.nan*np.ones(post_r_shape)

        post_o_shape = (par.T,par.Nbeta,par.Niota,par.Nh,par.Np,par.NLTV,par.Na)
        sol.o_q = np.nan*np.ones(post_o_shape)
        sol.o_w = np.nan*np.ones(post_o_shape)
        sol.o_inv_w = np.nan*np.ones(post_o_shape)

        # consumption
        negm_r_shape = (par.T,par.Nbeta,par.Nht,par.Np,par.Nz)
        sol.r_inv_vbar = np.nan*np.ones(negm_r_shape)
        sol.r_inv_mubar = np.nan*np.ones(negm_r_shape)
        sol.r_cbar = np.nan*np.ones(negm_r_shape)

        negm_o_shape = (par.T,par.Nbeta,par.Niota,par.Nh,par.Np,par.NLTV,par.Nz)
        sol.o_inv_vbar = np.nan*np.ones(negm_o_shape)
        sol.o_inv_mubar = np.nan*np.ones(negm_o_shape)
        sol.o_cbar = np.nan*np.ones(negm_o_shape)

        # intermediary
        rt_shape = (par.T,par.Nbeta,par.Nht+1,par.Np,par.Nx,par.Nht)
        sol.rt_inv_v = np.nan*np.ones(rt_shape)
        sol.rt_inv_mu = np.nan*np.ones(rt_shape)

        bt_shape = (par.T,par.Nbeta,par.Np,par.Nx,par.Niota,par.Nh,par.NLTV)
        sol.bt_inv_v = np.nan*np.ones(bt_shape)
        sol.bt_inv_mu = np.nan*np.ones(bt_shape)

        ft_shape = (par.T,par.Nbeta,par.Nh,par.Np,par.Nx,par.Niota,par.NLTV)
        sol.ft_inv_v = np.nan*np.ones(ft_shape)
        sol.ft_inv_mu = np.nan*np.ones(ft_shape)

        # final
        final_r_shape = (par.T,par.Nbeta,par.Nht,par.Np,par.Na)
        sol.r_v = np.nan*np.ones(final_r_shape)        
        sol.r_mu = np.nan*np.ones(final_r_shape)
        sol.r_d = np.nan*np.ones(final_r_shape,dtype=np.int_)

        final_o_shape = (par.T,par.Nbeta,par.Niota,par.Nh,par.Np,par.NLTV,par.Na)
        sol.o_v = np.nan*np.ones(final_o_shape)
        sol.o_mu = np.nan*np.ones(final_o_shape)
        sol.o_d = np.nan*np.ones(final_o_shape,dtype=np.int_)

        # b. simulation
        sim_shape = (par.simN,par.T)
        sim.d = -1*np.ones(sim_shape,dtype=np.int_)
        sim.iota = -1*np.ones(sim_shape,dtype=np.int_)
        sim.i_h = -1*np.ones(sim_shape,dtype=np.int_)
        sim.i_ht = -1*np.ones(sim_shape,dtype=np.int_)
        sim.i_p = -1*np.ones(sim_shape,dtype=np.int_)
        sim.LTV = np.nan*np.ones(sim_shape)
        sim.a = np.nan*np.ones(sim_shape)
        sim.c = np.nan*np.ones(sim_shape)
        sim.Nc = np.nan*np.ones(sim_shape)
        sim.x = np.nan*np.ones(sim_shape)

        sim.i_beta = -1*np.ones(par.simN,dtype=np.int_)
        sim.a0 = np.nan*np.ones(par.simN)

        sim.pi_p = np.nan*np.ones(sim_shape)
        sim.pi_c = np.nan*np.ones(sim_shape)

    def load_data(self):

        self.owner = dict()
        self.renter = dict()
        self.full = dict()

        stats = ['count', 'mean', 'p10', 'p25', 'p50', 'p75', 'p90']
        for stat in stats: 
            self.owner[stat]  = pd.read_csv(f'moments/{stat}_by_age_owner.txt', index_col='fam_age')
            self.renter[stat] = pd.read_csv(f'moments/{stat}_by_age_renter.txt', index_col='fam_age')
            self.full[stat]   = pd.read_csv(f'moments/{stat}_by_age_all.txt', index_col='fam_age')

    solve = solve_model
    simulate = simulate_model

    def test(self):
        """ used for testing parallel possibilities """

        solve_model(self,t_min=0,do_print=True)
        self.simulate()