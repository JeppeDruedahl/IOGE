# -*- coding: utf-8 -*-
"""figs

Plot results.

"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
plt.style.use('seaborn-whitegrid')
colors = [x['color'] for x in plt.style.library['seaborn']['axes.prop_cycle']]

##########
# owners #
##########

def post_decision_owner(model,varnames,t,i_beta,iota,i_h,i_p):

    # a. unpack
    par = model.par
    sol = model.sol

    varnames = varnames if type(varnames) is list else [varnames]
    Nvarnames = len(varnames)

    # b. figure
    fig = plt.figure(figsize=(6*Nvarnames,6))

    # c. plot
    LTV,m = np.meshgrid(par.grid_LTV,par.grid_a,indexing='ij')
    for i,varname in enumerate(varnames):

        ax = fig.add_subplot(1,Nvarnames,1+i,projection='3d')
        varname_full = f'o_{varname}'

        y = getattr(sol,varname_full)[t,i_beta,iota,i_h,i_p,:,:]
        ax.plot_surface(LTV,m,y,cmap=cm.viridis,edgecolor='none')
        ax.set_title(f'{varname_full}[{t},{i_beta},{iota},{i_h},{i_p}]')

        # d. details
        ax.set_xlabel('$\lambda_t$')
        ax.set_xlim([par.grid_LTV[0],par.grid_LTV[-1]])
        ax.set_ylabel('$a_t$')
        ax.set_ylim([par.grid_a[0],par.grid_a[-1]])
        ax.invert_xaxis()

    plt.show()

def negm_owner(model,varnames,t,i_beta,iota,i_h,i_p):

    # a. unpack
    par = model.par
    sol = model.sol

    varnames = varnames if type(varnames) is list else [varnames]
    Nvarnames = len(varnames)
    
    # b. figure
    fig = plt.figure(figsize=(6*Nvarnames,6))

    # c. plot
    LTV,z = np.meshgrid(par.grid_LTV,par.grid_z,indexing='ij')
    for i,varname in enumerate(varnames):

        ax = fig.add_subplot(1,Nvarnames,1+i,projection='3d')
        varname_full = f'o_{varname}'

        y = getattr(sol,varname_full)[t,i_beta,iota,i_h,i_p,:,:]
        ax.plot_surface(LTV,z,y,cmap=cm.viridis,edgecolor='none')
        ax.set_title(f'{varname_full}[{t},{i_beta},{iota},{i_h},{i_p}]')

        # d. details
        ax.set_xlabel(r'$\lambda_{t}$')
        ax.set_xlim([par.grid_LTV[0],par.grid_LTV[-1]])
        ax.set_ylabel('$z_t$')
        ax.set_ylim([par.grid_z[0],par.grid_z[-1]])
        ax.invert_xaxis()

    plt.show()

def ft(model,t,i_beta,i_p,iota,i_LTV):

    # a. unpack
    par = model.par
    sol = model.sol

    # b. figure
    fig = plt.figure(figsize=(6,4))
    ax = fig.add_subplot(1,1,1)

    # c. plot
    for i_h,h in enumerate(par.grid_h):
        ax.plot(par.grid_x,sol.ft_inv_v[t,i_beta,i_h,i_p,:,iota,i_LTV],label=f'$h = {h:.2f}$')

    # d. details
    ax.set_title(f'ft_inv_v[{t},{i_beta},{i_h},{i_p},:,{iota},{i_LTV}]')
    ax.set_xlabel('$x_t$')
    ax.legend(frameon=True)

    plt.show()

def bt(model,t,i_beta,i_p,iota,i_h,i_LTV):

    # a. unpack
    par = model.par
    sol = model.sol

    # b. figure
    fig = plt.figure(figsize=(6,4))
    ax = fig.add_subplot(1,1,1)

    # c. plot
    ax.plot(par.grid_x,sol.bt_inv_v[t,i_beta,i_p,:,iota,i_h,i_LTV])

    # d. details
    ax.set_title(f'bt_inv_v[{t},{i_p},:,{i_beta},{iota},{i_h},{i_LTV}]')
    ax.set_xlabel('$x_t$')

    plt.show()

def owner(model,varnames,t,i_beta,iota_lag,i_h_lag,i_p_lag):

    # a. unpack
    par = model.par
    sol = model.sol

    varnames = varnames if type(varnames) is list else [varnames]
    Nvarnames = len(varnames)

    # b. figure
    fig = plt.figure(figsize=(6*Nvarnames,6))

    # c. plot
    LTV,a_lag = np.meshgrid(par.grid_LTV,par.grid_a,indexing='ij')
    for i,varname in enumerate(varnames):

        if varname == 'd':
            ax = fig.add_subplot(1,Nvarnames,1+i)
        else:
            ax = fig.add_subplot(1,Nvarnames,1+i,projection='3d')
        varname_full = f'o_{varname}'

        y = getattr(sol,varname_full)[t,i_beta,iota_lag,i_h_lag,i_p_lag,:,:]
        if varname == 'd':
            for i,label in [(-1,'not feasible'),(0,'renter'),(1,'buyer'),(2,'refinancer'),(3,'keeper')]:
                if np.any(y == i):
                    ax.scatter(LTV[y == i],a_lag[y == i],s=4,label=label)
            ax.legend(frameon=True)
        else:
            ax.plot_surface(LTV,a_lag,y,cmap=cm.viridis,edgecolor='none')

        ax.set_title(f'{varname}[{t},{i_beta},{iota_lag},{i_h_lag},{i_p_lag}]')

        # d. details
        ax.set_xlabel('$\lambda_{t-1}$')
        ax.set_xlim([par.grid_LTV[0],par.grid_LTV[-1]])
        ax.set_ylabel('$a_{t-1}$')
        ax.set_ylim([par.grid_a[0],par.grid_a[-1]])

        if not varname == 'd':
            ax.invert_xaxis()

    plt.show()

###########
# renters #
###########

def post_decision_renter(model,varnames,t,i_beta,i_ht,i_p):

    # a. unpack
    par = model.par
    sol = model.sol

    varnames = varnames if type(varnames) is list else [varnames]
    Nvarnames = len(varnames)

    # b. figure
    fig = plt.figure(figsize=(6*Nvarnames,4))

    # c. plot
    for i,varname in enumerate(varnames):

        ax = fig.add_subplot(1,Nvarnames,1+i)
        varname_full = f'r_{varname}'

        y = getattr(sol,varname_full)[t,i_beta,i_ht,i_p,:]
        ax.plot(par.grid_a,y)
        ax.set_title(f'{varname_full}[{t},{i_beta},{i_ht},{i_p}]')

        # d. details
        ax.set_xlabel('$a_t$')
        ax.set_xlim([par.grid_a[0],par.grid_a[-1]])

    plt.show()

def negm_renter(model,varnames,t,i_beta,i_ht,i_p):

    # a. unpack
    par = model.par
    sol = model.sol

    varnames = varnames if type(varnames) is list else [varnames]
    Nvarnames = len(varnames)
    
    # b. figure
    fig = plt.figure(figsize=(6*Nvarnames,4))

    # c. plot
    for i,varname in enumerate(varnames):

        ax = fig.add_subplot(1,Nvarnames,1+i)
        varname_full = f'r_{varname}'

        y = getattr(sol,varname_full)[t,i_beta,i_ht,i_p,:]
        ax.plot(par.grid_z,y)
        ax.set_title(f'{varname_full}[{t},{i_beta},{i_ht},{i_p}]')

        # d. details
        ax.set_xlabel(r'$z_t$')
        ax.set_xlim([par.grid_z[0],par.grid_z[-1]])

    plt.show()

def rt(model,t,i_beta,i_ht_lag,i_p):

    # a. unpack
    par = model.par
    sol = model.sol

    # b. figure
    fig = plt.figure(figsize=(6,4))
    ax = fig.add_subplot(1,1,1)

    # c. plot
    for i_ht,ht in enumerate(par.grid_ht):
        ax.plot(par.grid_x,sol.rt_inv_v[t,i_beta,i_ht_lag,i_p,:,i_ht],label=fr'$\tilde{{h}} = {ht}$')

    # d. details
    ax.set_title(f'rt_inv_v[{t},{i_beta},{i_ht_lag},{i_p},:,:]')

    ax.legend(frameon=True)
    ax.set_xlabel('$x_t$')

    plt.show()

def renter(model,varnames,t,i_beta,i_ht_lag,i_p):

    # a. unpack
    par = model.par
    sol = model.sol

    varnames = varnames if type(varnames) is list else [varnames]
    Nvarnames = len(varnames)

    # b. figure
    fig = plt.figure(figsize=(6*Nvarnames,6))

    # c. plot
    for i,varname in enumerate(varnames):

        ax = fig.add_subplot(1,Nvarnames,1+i)
        varname_full = f'r_{varname}'

        y = getattr(sol,varname_full)[t,i_beta,i_ht_lag,i_p,:]
        if varname == 'd':
            for i,label in [(-1,'not feassible'),(0,'renter'),(1,'buyer'),(2,'refinancer'),(3,'keeper')]:
                a_lag = par.grid_a[y == i]
                if a_lag.size > 0:
                    ax.scatter(a_lag,i*np.ones(a_lag.size),s=4,label=label)      
            ax.legend(frameon=True)      
        else:
            ax.plot(par.grid_a,y)
        ax.set_title(f'{varname_full}[{t},{i_beta}{i_ht_lag}{i_p}]')

        # d. details
        ax.set_xlabel('$a_{t-1}$')
        ax.set_xlim([par.grid_a[0],par.grid_a[-1]])

    plt.show()

##############
# life-cycle #
##############

def calibration(model,prefix=None):

    prefix_ = '' if prefix is None else prefix + '_'

    # a. unpack
    par = model.par
    sim = model.sim
    ages = par.age_min + np.arange(par.T)

    # data
    income_data = model.full['mean']['inc'].values 
    income_p25_data = model.full['p25']['inc'].values 
    income_p50_data = model.full['p50']['inc'].values 
    income_p75_data = model.full['p75']['inc'].values 
    norm = income_data[0]

    print(f'data is normalized with mean income at age {ages[0]} = {norm:,.1f}')
    ages_data = ages[:income_data.size]

    net_worth_data = model.full['mean']['net_wealth'].values
    net_worth_p25_data = model.full['p25']['net_wealth'].values
    net_worth_p75_data = model.full['p75']['net_wealth'].values

    housing_value_data = model.owner['mean']['property_value'].values
    housing_value_p25_data = model.owner['p25']['property_value'].values
    housing_value_p75_data = model.owner['p75']['property_value'].values
    
    owners_count_data = model.owner['count'].inc.values
    renters_count_data = model.renter['count'].inc.values
    owner_share_data = owners_count_data/(owners_count_data+renters_count_data)    
    io_share_data = model.owner['mean']['io_share'].values

    net_worth_owners_data = model.owner['mean']['net_wealth'].values
    debt_data = model.owner['mean']['mortgage_loans'].values

    # model 
    income = par.Gamma[np.newaxis,:]*par.grid_p[sim.i_p]
    owner_share = np.mean(sim.i_h >= 0,axis=0)
    home_equity_all = np.zeros(sim.a.shape)
    I = sim.i_h >= 0
    home_equity_all[I] += (1.0-sim.LTV[I])*par.ph*par.grid_h[sim.i_h[I]]
    net_worth = np.mean(sim.a + home_equity_all,axis=0)
    net_worth_p25 = np.percentile(sim.a + home_equity_all,25,axis=0)
    net_worth_p75 = np.percentile(sim.a + home_equity_all,75,axis=0)
    income_p50 = np.percentile(income,50,axis=0)

    # owners vs. renters
    io_share = np.nan*np.ones(par.T)
    housing_value = np.nan*np.ones(par.T)
    housing_value_p75 = np.nan*np.ones(par.T)
    housing_value_p25 = np.nan*np.ones(par.T)
    home_equity = np.nan*np.ones(par.T)
    debt = np.nan*np.ones(par.T)
    net_worth_owners = np.nan*np.ones(par.T)

    for t in range(par.T):
    
        I = sim.i_h[:,t] >= 0
        iota = sim.iota[I,t]
        i_h = sim.i_h[I,t]
        LTV = sim.LTV[I,t]
        a_owners = sim.a[I,t]

        if np.any(I):

            io_share[t] = np.mean(iota > 0)
            housing_value[t] = np.mean(par.ph*par.grid_h[i_h])
            housing_value_p25[t] = np.percentile(par.ph*par.grid_h[i_h],25)
            housing_value_p75[t] = np.percentile(par.ph*par.grid_h[i_h],75)
            home_equity[t] = np.mean((1-LTV)*par.ph*par.grid_h[i_h])
            debt[t] = housing_value[t]-home_equity[t]
            net_worth_owners[t] = home_equity[t] + np.mean(a_owners)
        
    # income
    fig = plt.figure(figsize=(6,4))
    ax = fig.add_subplot(1,1,1)
    ax.set_title('income')
    fac = 1.02**np.arange(income_data.size)
    ax.plot(ages,np.percentile(income,25,axis=0),ls='--',color=colors[0])
    ax.plot(ages,np.percentile(income,75,axis=0),ls='--',color=colors[0])
    ax.plot(ages,np.mean(income,axis=0),color=colors[0],label='model')
    ax.plot(ages_data,fac*income_p25_data/norm,ls='--',color='black')
    ax.plot(ages_data,fac*income_p75_data/norm,ls='--',color='black')
    ax.plot(ages_data,fac*income_data/norm,color='black',label='data')
    ax.legend(frameon=True)
    plt.show()
    ax.set_title('')
    plt.tight_layout()
    fig.savefig(f'figs/{prefix_}income.pdf')

    # owner share 
    fig = plt.figure(figsize=(6,4))
    ax = fig.add_subplot(1,1,1)
    ax.set_title('owner share')
    ax.plot(ages,owner_share,color=colors[0],label='model')
    ax.plot(ages_data,owner_share_data,color='black',label='data')
    ax.legend(frameon=True)
    plt.show()
    ax.set_title('')
    plt.tight_layout()
    fig.savefig(f'figs/{prefix_}owner_share.pdf')

    # housing value
    fig = plt.figure(figsize=(6,4))
    ax = fig.add_subplot(1,1,1)
    ax.set_title('property value (owners) (relative to median income)')
    ax.plot(ages,housing_value_p25/income_p50,ls='--',color=colors[0])
    ax.plot(ages,housing_value_p75/income_p50,ls='--',color=colors[0])
    ax.plot(ages,housing_value/income_p50,color=colors[0],label='model')
    ax.plot(ages_data,housing_value_data/income_p50_data,color='black',label='data')
    ax.plot(ages_data,housing_value_p25_data/income_p50_data,ls='--',color='black')
    ax.plot(ages_data,housing_value_p75_data/income_p50_data,ls='--',color='black')
    ax.legend(frameon=True)
    plt.show()
    ax.set_title('')
    plt.tight_layout()
    fig.savefig(f'figs/{prefix_}housing_value.pdf')

    # # debt
    fig = plt.figure(figsize=(6,4))
    ax = fig.add_subplot(1,1,1)
    ax.set_title('debt (owners) (relative to median income)')
    ax.plot(ages,debt/income_p50,color=colors[0],label='model')
    ax.plot(ages_data,debt_data/income_p50_data,color='black',label='data')
    ax.legend(frameon=True)
    plt.show()
    ax.set_title('')
    plt.tight_layout()
    fig.savefig(f'figs/{prefix_}debt.pdf')

    # networth
    fig = plt.figure(figsize=(6,4))
    ax = fig.add_subplot(1,1,1)
    ax.set_title('net worth (relative to median income)')
    ax.plot(ages,net_worth_p25/income_p50,ls='--',color=colors[0])
    ax.plot(ages,net_worth_p75/income_p50,ls='--',color=colors[0])
    ax.plot(ages,net_worth/np.percentile(income,50,axis=0),color=colors[0],label='model')
    ax.plot(ages_data,net_worth_data/income_p50_data,color='black',label='data')
    ax.plot(ages_data,net_worth_p25_data/income_p50_data,ls='--',color='black')
    ax.plot(ages_data,net_worth_p75_data/income_p50_data,ls='--',color='black')
    ax.legend(frameon=True)
    plt.show()
    ax.set_title('')
    plt.tight_layout()
    fig.savefig(f'figs/{prefix_}networth.pdf')

    # io share 
    fig = plt.figure(figsize=(6,4))
    ax = fig.add_subplot(1,1,1)
    ax.set_title('io share')
    ax.plot(ages,io_share,color=colors[0],label='model')
    ax.plot(ages_data,io_share_data,color='black',label='data')
    ax.legend(frameon=True)
    plt.show()
    ax.set_title('')
    plt.tight_layout()
    fig.savefig(f'figs/{prefix_}io_share.pdf')
    
##############
# life-cycle #
##############

def life_cycle(model,varname,latex=None):

    # a. unpack
    par = model.par
    sim = model.sim

    # b. figure
    fig = plt.figure(figsize=(6,4))
    ax = fig.add_subplot(1,1,1)

    if varname == 'housing_value':
        var = np.zeros((model.par.simN,model.par.T))
        I = model.sim.i_h > 0
        var[I] = model.par.ph*model.par.grid_h[model.sim.i_h[I]]
    else:
        var = getattr(sim,varname)
        
    # c. plot
    y = np.nanmean(var,axis=0)
    y_p10 = np.nanpercentile(var,10,axis=0)
    y_p90 = np.nanpercentile(var,90,axis=0)
    ages = par.age_min + np.arange(par.T)
    ax.axvline(par.age_min+par.TR-1,ls='--',color='black',alpha=0.5)
    ax.plot(ages,y,label='mean',color=colors[0])
    ax.plot(ages,y_p10,ls='--',label='10th percentile',color=colors[0])
    ax.plot(ages,y_p90,ls='--',label='90th percentile',color=colors[0])
    ax.set_title(f'{varname}')

    # d. details
    xlabel = varname if latex is None else latex
    ax.set_xlabel(xlabel)

    ax.legend(frameon=True)
    plt.show()

def life_cycle_income(model):

    # a. unpack
    par = model.par
    sim = model.sim

    # b. figure
    fig = plt.figure(figsize=(6,4))
    ax = fig.add_subplot(1,1,1)

    # c. plot
    ages = par.age_min + np.arange(par.T)    

    for i_p,p in enumerate(par.grid_p):
        shares = np.mean(sim.i_p == i_p,axis=0)
        ax.plot(ages,shares,label=f'p = {p:.2f}')
    
    # d. details
    ax.set_xlabel('shares')
    ax.legend(frameon=True)

    plt.show()

def life_cycle_housing(model,owners=False,renters=False):

    # a. unpack
    par = model.par
    sim = model.sim

    # b. figure
    fig = plt.figure(figsize=(6,4))
    ax = fig.add_subplot(1,1,1)

    # c. plot
    ages = par.age_min + np.arange(par.T)    

    if owners:

        for i_h,h in enumerate(par.grid_h):
            shares = np.mean(sim.i_h == i_h,axis=0)
            ax.plot(ages,shares,label=f'h = {h:.2f}')
        
        ax.set_title('owners')

    elif renters:

        for i_ht,ht in enumerate(par.grid_ht):
            shares = np.mean(sim.i_ht == i_ht,axis=0)
            ax.plot(ages,shares,label=fr'$\tilde{{h}} = {ht:.2f}$')

        ax.set_title('renters')

    else:

        owners = np.mean(sim.i_h >= 0,axis=0)
        renters = 1-owners
        ax.plot(ages,owners,label='owners')
        ax.plot(ages,renters,label='renters')

    # d. details
    ax.set_xlabel('shares')
    ax.legend(frameon=True)

    plt.show()

def life_cycle_housing_income_dist(model,age):

    # a. unpack
    par = model.par
    sim = model.sim
    t = par.age_min - age

    # b. figure
    fig = plt.figure(figsize=(6,4))
    ax = fig.add_subplot(1,1,1)

    # c. plot
    owners = sim.i_h[:,t] >= 0
    renters = ~owners
    income = par.Gamma[t]*par.grid_p[sim.i_p[:,t]]

    ax.hist(income[owners],bins=par.Np,alpha=0.50,label='owners',density=True)
    ax.hist(income[renters],bins=par.Np,alpha=0.50,label='renters',density=True)

    # d. details
    ax.set_xlabel('income')
    ax.legend(frameon=True)

    plt.show()

def life_cycle_io(model):

    # a. unpack
    par = model.par
    sim = model.sim

    # b. figure
    fig = plt.figure(figsize=(6,4))
    ax = fig.add_subplot(1,1,1)

    # c. plot
    ages = par.age_min + np.arange(par.T)    

    totalsum = np.sum(sim.i_h >= 0,axis=0)
    I = totalsum > 0
    for iota in range(par.Niota):
        iotasum = np.sum(sim.iota == iota,axis=0)
        share = np.nan*np.ones(par.T)
        share[I] = iotasum[I]/totalsum[I]
        ax.plot(ages,share,label=fr'$\iota = {iota}$')
    
    # d. details
    ax.set_xlabel('shares')
    ax.legend(frameon=True)

    plt.show()

def life_cycle_d_renter(model):

    # a. unpack
    par = model.par
    sim = model.sim

    # b. figure
    fig = plt.figure(figsize=(6,4))
    ax = fig.add_subplot(1,1,1)

    # c. plot
    ages = par.age_min + np.arange(par.T)    

    totalsum = np.sum(sim.i_ht[:,:-1] >= 0,axis=0)
    I = totalsum > 0

    dsum = np.sum( (sim.d[:,1:] == 0) & (sim.i_ht[:,:-1] >= 0) ,axis=0)
    share = np.nan*np.ones(par.T-1)
    share[I] = dsum[I]/totalsum[I]
    ax.plot(ages[1:],share,label=fr'stay renter')
    
    dsum = np.sum( (sim.d[:,1:] == 1) & (sim.i_ht[:,:-1] >= 0) ,axis=0)
    share = np.nan*np.ones(par.T-1)
    share[I] = dsum[I]/totalsum[I]    
    ax.plot(ages[1:],share,label=fr'become houseowner')
    
    # d. details
    ax.set_xlabel('shares')
    ax.legend(frameon=True)

    plt.show()

def life_cycle_d_owner(model):

    # a. unpack
    par = model.par
    sim = model.sim

    # b. figure
    fig = plt.figure(figsize=(6,4))
    ax = fig.add_subplot(1,1,1)

    # c. plot
    ages = par.age_min + np.arange(par.T)    

    totalsum = np.sum(sim.i_h[:,:-1] >= 0,axis=0)
    I = totalsum > 0

    dsum = np.sum( (sim.d[:,1:] == 0) & (sim.i_h[:,:-1] >= 0) ,axis=0)
    share = np.nan*np.ones(par.T-1)
    share[I] = dsum[I]/totalsum[I]
    ax.plot(ages[1:],share,label=fr'become renter')
    
    dsum = np.sum( (sim.d[:,1:] == 1) & (sim.i_h[:,:-1] >= 0) ,axis=0)
    share = np.nan*np.ones(par.T-1)
    share[I] = dsum[I]/totalsum[I]    
    ax.plot(ages[1:],share,label=fr'move to new house')
    
    dsum = np.sum( (sim.d[:,1:] == 2) & (sim.i_h[:,:-1] >= 0) ,axis=0)
    share = np.nan*np.ones(par.T-1)
    share[I] = dsum[I]/totalsum[I]    
    ax.plot(ages[1:],share,label=fr'refinance mortgage')

    dsum = np.sum( (sim.d[:,1:] == 3) & (sim.i_h[:,:-1] >= 0) ,axis=0)
    share = np.nan*np.ones(par.T-1)
    share[I] = dsum[I]/totalsum[I]    
    ax.plot(ages[1:],share,color='black',label=fr'stay with same house and mortgage')

    # d. details
    ax.set_xlabel('shares')
    ax.legend(frameon=True)

    plt.show()

###########
# compare #
###########

def life_cycle_compare(models,varname,latex=None,prefix=None):

    fig = plt.figure(figsize=(6,4))
    ax = fig.add_subplot(1,1,1)

    for model,label in zip(models,['baseline','No interest only']):

        par = model.par
        sim = model.sim

        if varname == 'housing_value':
            var = np.zeros((model.par.simN,model.par.T))
            I = model.sim.i_h > 0
            var[I] = model.par.ph*model.par.grid_h[model.sim.i_h[I]]
        elif varname == 'homeowner_share':
            var = model.sim.i_h > 0
        else:
            var = getattr(sim,varname)
        
        y = np.mean(var,axis=0)
        ages = par.age_min + np.arange(par.T)
        ax.plot(ages,y,label=label)

    ax.set_title(f'{varname}')
    xlabel = varname if latex is None else latex
    ax.set_xlabel(xlabel)

    ax.legend(frameon=True)
    plt.show()

    plt.tight_layout()
    prefix_ = '' if prefix is None else prefix + '_'
    ax.set_title('')
    fig.savefig(f'figs/{prefix_}{varname}_compare.pdf')