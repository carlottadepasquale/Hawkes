from __future__ import division
from __future__ import print_function

import sys,time,datetime,copy,subprocess,itertools,pickle,warnings,numbers

import numpy as np
import scipy as sp
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib as mpl

from .indexed_ndarray import indexed_ndarray

##################################
## MCMC
##################################
"""
def MCMC_DetermieStepSize(F_LG,para_ini,Data,cdt,stg,n_core,prior=[],opt=[]):

    step_size_list = np.array([0.06,0.08,0.1,0.12,0.15,0.2,0.25,0.3,0.4,0.5])
    m = len(step_size_list)

    p = Pool(n_core)
    rslt = []
    for i in range(m):
        stg_tmp = deepcopy(stg)
        stg_tmp['step_size'] = step_size_list[i]
        rslt.append( p.apply_async(MCMC,args=[F_LG,para_ini,Data,cdt,stg_tmp,200,prior,['print']]) )
    p.close()
    p.join()
    rslt = [ rslt[i].get() for i in range(m) ]

    step_size    = [ rslt[i][2] for i in range(m) ]
    r_accept     = [ rslt[i][3] for i in range(m) ]
    elapsed_time = [ rslt[i][4] for i in range(m) ]
    dtl = pd.DataFrame(np.vstack([step_size,r_accept,elapsed_time]).T,columns=['step_size','r_accept','elapsed_time'])

    opt_step_size = dtl.iloc[ np.argmin(np.fabs(dtl['r_accept'].values.copy()-0.5)) ]['step_size']

    return [opt_step_size,dtl]

def MCMC_prl(F_LG,para_ini,Data,cdt,stg,n_sample,n_core,prior=[],opt=[]):

    print( "MCMC n_sample:%d n_core:%d" % (n_sample,n_core) )

    ##determine step size
    [opt_step_size,dtl1] = MCMC_DetermieStepSize(F_LG,para_ini,Data,cdt,stg,n_core,prior,opt)
    stg['step_size'] = opt_step_size

    print( "estimated processing time %.2f minutes" % (dtl1['elapsed_time'].mean()*n_sample/200.0/60.0) )

    p = Pool(n_core)
    rslt = [ p.apply_async(MCMC,args=[F_LG,para_ini,Data,cdt,stg,n_sample,prior,opt])   for i in range(n_core)   ]
    p.close()
    p.join()
    rslt = [ rslt[i].get() for i in range(n_core) ]

    para_mcmc     = pd.concat([rslt[i][0].iloc[0::10] for i in range(n_core) ],ignore_index=True)
    L_mcmc        = np.array([ rslt[i][1][0::10]      for i in range(n_core) ]).flatten()
    step_size     = np.array([ rslt[i][2]             for i in range(n_core) ])
    r_accept      = np.array([ rslt[i][3]             for i in range(n_core) ])
    elapsed_time  = np.array([ rslt[i][4]             for i in range(n_core) ])
    dtl2 = pd.DataFrame(np.vstack([step_size,r_accept,elapsed_time]).T,columns=['step_size','r_accept','elapsed_time'])

    dtl_mcmc = {'step_size':opt_step_size,'dtl1':dtl1,'dtl2':dtl2}

    return [para_mcmc,L_mcmc,dtl_mcmc]


def MCMC(model,n_sample,step_coef,prior=[],opt=[]):

    #random number seed
    seed = datetime.datetime.now().microsecond *datetime.datetime.now().microsecond % 4294967295
    np.random.seed(seed)

    #parameter
    ## parameter setting
    param = model.para.inherit()
    para1 = model.para.copy()
    m = len(para1)
    m_exp = len(para1["para_exp"])
    m_ord = len(para1["para_ord"])

    ##step
    step_MCMC = model.ste.copy()
    step_MCMC["para_exp"] = np.minimum( np.log( 1.0 + step_MCMC["para_exp"]/para1["para_exp"] ) ,0.4)
    step_MCMC.values *= step_coef

    ## prior setting
    if prior:
        ##prior format transform
        prior =  [ {"name":prior_i[0],"index": prior_i[1], "type":prior_i[2], "mu":prior_i[3], "sigma": prior_i[4]} for prior_i in prior ]
        prior = [ prior_i for prior_i in prior if prior_i["type"] != "f" ]

    #prepare
    para_mcmc = []
    L_mcmc    = []

    #initial value
    [L1,_] = Penalized_LG(model,para1,prior,only_L=True)
    para_mcmc.append(para1.values)
    L_mcmc.append(L1)

    i = 1
    j = 0
    k = 0
    t_start = time.time()

    while 1:

        para2 = para1.inherit().initialize_values(1)
        para2["para_ord"] = para1["para_ord"] + np.random.randn(m_ord)*step_MCMC["para_ord"]
        para2["para_exp"] = para1["para_exp"] * np.exp( np.random.randn(m_exp)*step_MCMC["para_exp"] )
        [L2,_] = Penalized_LG(model,para2,prior,only_L=True)

        if L1<L2 or np.random.rand() < np.exp(L2-L1): #accept

            j += 1
            k += 1
            para1 = para2
            L1 = L2

        para_mcmc.append(para1.values)
        L_mcmc.append(L1)

        if 'print' in opt and np.mod(i,1000) == 0:
            print(i)

        #adjust the step width
        if np.mod(i,500) == 0:

            if k<250:
                step_MCMC.values *= 0.95
            else:
                step_MCMC.values *= 1.05

            k = 0

        i += 1

        if i == n_sample:
            break


    r_accept = 1.0*j/n_sample
    elapsed_time = time.time() - t_start

    para_mcmc = param.inherit().set_values(np.vstack(para_mcmc).transpose())
    L_mcmc = np.array(L_mcmc)

    return [para_mcmc,L_mcmc,r_accept,elapsed_time]
"""
##################################
## Quasi Newton
##################################
def Quasi_Newton(model,prior=[],merge=[],opt=[]):

    ## parameter setting
    para_list  = model.stg["para_list"]
    para_length = [ model.stg["para_length"][key] for key in para_list ]
    param = indexed_ndarray().set_hash(para_list,para_length)
    param.add_hash( [ pr for pr in para_list if     model.stg["para_exp"][pr] ], "para_exp")
    param.add_hash( [ pr for pr in para_list if not model.stg["para_exp"][pr] ], "para_ord")

    para = param.inherit().set_values_from_dict(model.stg["para_ini"]) if "para_ini" not in opt else param.inherit().set_values_from_dict(model.para)
    step_Q = param.inherit().set_values_from_dict(model.stg["para_step_Q"]).values
    m = len(para)

    ## prior setting
    if prior:
        ##fix check
        para_fix_index = [ (prior_i["name"],prior_i["index"]) for prior_i in prior if prior_i["type"] == "f" ]
        para_fix_value = [ prior_i["mu"] for prior_i in prior if prior_i["type"] == "f" ]
        param.add_hash(para_fix_index,"fix")
        para["fix"] = para_fix_value
        prior = [ prior_i for prior_i in prior if prior_i["type"] != "f" ]
    else:
        param.add_hash([],"fix")

    ## merge setting
    if merge:
        d = len(merge)
        index_merge = pd.Series(0,index=pd.MultiIndex.from_tuples(param.index_tuple),dtype="i8")

        for i in range(d):
            para[merge[i]] = para[merge[i]].mean()
            index_merge.loc[merge[i]] = i+1

        M_merge_z = np.eye(m)[ index_merge == 0 ]
        M_merge_nz = np.vstack(  [ np.eye(m)[index_merge == i+1].sum(axis=0) for i in range(d) ] )
        M_merge = np.vstack([M_merge_z,M_merge_nz])
        M_merge_T = np.transpose(M_merge)
        m_reduced = M_merge.shape[0]

    else:
        M_merge = 1
        M_merge_T = 1
        m_reduced = m

    # calculate Likelihood and Gradient at the initial state
    [L1,G1] = Penalized_LG(model,para,prior)
    G1["para_exp"] *= para["para_exp"]
    G1 = np.dot(M_merge,G1.values)

    # main
    H = np.eye(m_reduced)
    i_loop = 0

    while 1:

        if 'print' in opt:
            print(i_loop)
            print(para)
            #print(G1)
            print( "L = %.3f, norm(G) = %e\n" % (L1,np.linalg.norm(G1)) )
            #sys.exit()

        #break rule
        if np.linalg.norm(G1) < 1e-5 :
            break

        #calculate direction
        s = H.dot(G1);
        s_data = np.dot(M_merge_T,s)
        gamma = 1/np.max([np.max(np.abs(s_data)/step_Q),1])
        s_data = s_data * gamma
        s = s * gamma

        #move to new point
        s_extended = param.inherit().set_values(s_data)
        para["para_ord"] += s_extended["para_ord"]
        para["para_exp"] *= np.exp( s_extended["para_exp"] )

        #calculate Likelihood and Gradient at the new point
        [L2,G2] = Penalized_LG(model,para,prior)
        G2["para_exp"] *= para["para_exp"]
        G2 = np.dot(M_merge,G2.values)

        #update hessian matrix
        y = (G1-G2).reshape(-1,1)
        s = s.reshape(-1,1)

        if  y.T.dot(s) > 0:
            H = H + (y.T.dot(s)+y.T.dot(H).dot(y))*(s*s.T)/(y.T.dot(s))**2 - (H.dot(y)*s.T+(s*y.T).dot(H))/(y.T.dot(s))
        else:
            H = np.eye(m_reduced)

        #update Gradients
        L1 = L2
        G1 = G2

        i_loop += 1

    ###OPTION: Estimation Error
    if 'ste' in opt:
        ste = EstimationError(model,para,prior)
    else:
        ste = []

    ###OPTION: Check map solution
    if 'check' in opt:
            Check_QN(model,para,prior)

    return [para,L1,ste,np.linalg.norm(G1),i_loop]

def Check_QN(model,para,prior):

    ste = EstimationError_approx(model,para,prior)
    ste["fix"] = 0
    a = np.linspace(-1,1,21)

    for index in para.index_tuple:

        plt.figure()
        plt.title(index)

        for i in range(len(a)):
            para_tmp = para.copy()
            para_tmp[index] += a[i] * ste[index]
            L = Penalized_LG(model,para_tmp,prior)[0]
            plt.plot(para_tmp[index],L,"ko")

            if i==10:
                plt.plot(para_tmp[index],L,"ro")

#################################
## Basic funnctions
#################################

def G_NUMERICAL(model,para):

    para_list = model.stg["para_list"]
    para_length = model.stg["para_length"]
    step_diff = para.inherit().set_values_from_dict(model.stg["para_step_diff"])
    step_diff["para_exp"] *= para["para_exp"]
    G = para.inherit().initialize_values(1)

    for index in para.index_tuple:

        step = step_diff[index]

        """
        para_tmp = para.copy(); para_tmp[index] -= step;  L1 = model.LG(para_tmp)[0]
        para_tmp = para.copy(); para_tmp[index] += step;  L2 = model.LG(para_tmp)[0]
        G[index]= (L2-L1)/2/step
        """

        para_tmp = para.copy(); para_tmp[index] -= 2*step;  L1 = model.LG(para_tmp)[0]
        para_tmp = para.copy(); para_tmp[index] -= 1*step;  L2 = model.LG(para_tmp)[0]
        para_tmp = para.copy(); para_tmp[index] += 1*step;  L3 = model.LG(para_tmp)[0]
        para_tmp = para.copy(); para_tmp[index] += 2*step;  L4 = model.LG(para_tmp)[0]
        G[index]= (L1-8*L2+8*L3-L4)/12/step

    return G

def Hessian(model,para,prior):

    para_list = model.stg["para_list"]
    para_length = model.stg["para_length"]
    step_diff = para.inherit().set_values_from_dict(model.stg["para_step_diff"])
    step_diff["para_exp"] *= para["para_exp"]
    H = para.inherit().initialize_values(len(para))

    for index in para.index_tuple:

        step = step_diff[index]

        para_tmp = para.copy(); para_tmp[index] -= step;  G1 = Penalized_LG(model,para_tmp,prior)[1].values
        para_tmp = para.copy(); para_tmp[index] += step;  G2 = Penalized_LG(model,para_tmp,prior)[1].values
        H[index] = (G2-G1)/2/step

    H["fix"] = 0
    H.values[H.hash["fix"],H.hash["fix"]] = -1e+20

    return H

def EstimationError(model,para,prior):
    H = Hessian(model,para,prior)
    ste = para.inherit().set_values(np.sqrt(np.diag(np.linalg.inv(-H.values))))
    return ste

def EstimationError_approx(model,para,prior):
    H = Hessian(model,para,prior)
    ste = para.inherit().set_values(1.0/np.sqrt(np.diag(-H.values)))
    return ste


def Penalized_LG(model,para,prior,only_L=False):

    [L,G] = model.LG(para,only_L)

    if isinstance(G,str):
        G = G_NUMERICAL(model,para)

    ## fix
    if not only_L:
        G["fix"] = 0

    ## prior
    if prior:

        for prior_i in prior:
            para_key = prior_i["name"]
            para_index = prior_i["index"]
            prior_type = prior_i["type"]
            mu = prior_i["mu"]
            sigma = prior_i["sigma"]
            x = para[(para_key,para_index)]

            if prior_type == 'n': #prior: normal distribution
                L  += - np.log(2*np.pi*sigma**2)/2 - (x-mu)**2/2/sigma**2
                if not only_L:
                    G[(para_key,para_index)] += - (x-mu)/sigma**2
            elif prior_type ==  'ln': #prior: log-normal distribution
                L  += - np.log(2*np.pi*sigma**2)/2 - np.log(x) - (np.log(x)-mu)**2/2/sigma**2
                if not only_L:
                    G[(para_key,para_index)] += - 1/x - (np.log(x)-mu)/sigma**2/x
            elif prior_type == "b": #prior: barrier function
                L  += - mu/x
                if not only_L:
                    G[(para_key,para_index)] += mu/x**2
            elif prior_type == "b2": #prior: barrier function
                L  += mu *np.log10(np.e)*np.log(x)
                if not only_L:
                    G[(para_key,para_index)] += mu * np.log10(np.e)/x

    return [L,G]

#################################
## para_stg
#################################
def merge_stg(para_stgs):

    stg = {}
    stg['para_list']      = []
    stg['para_length']    = {}
    stg['para_exp']       = {}
    stg['para_ini']       = {}
    stg['para_step_Q']    = {}
    stg['para_step_diff'] = {}

    for para_stg in para_stgs:
        stg['para_list'].extend(para_stg['list'])
        stg['para_length'].update(para_stg['length'])
        stg['para_exp'].update(para_stg['exp'])
        stg['para_ini'].update(para_stg['ini'])
        stg['para_step_Q'].update(para_stg['step_Q'])
        stg['para_step_diff'].update(para_stg['step_diff'])

    return stg
