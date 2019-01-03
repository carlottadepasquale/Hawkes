from __future__ import division
from __future__ import print_function

import sys,time,datetime,copy,subprocess,itertools,pickle,warnings,numbers

import numpy as np
import scipy as sp
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib as mpl

from scipy.special import gamma,digamma

from .StatTool import Quasi_Newton,indexed_ndarray
from .BasisFunction import loglinear_COS_intensity,plinear_intensity

try:
    import pyximport; pyximport.install(setup_args={'include_dirs': np.get_include()})
    from .Hawkes_C import LG_kernel_SUM_exp_i_cython
    from .Hawkes_C import LG_kernel_SUM_pow_cython
    cython_import = True
    #print("cython")
except:
    cython_import = False
    #print("python")


##########################################################################################################
## class
##########################################################################################################
class base_class:

    ### initialize
    def set_kernel(self,type,num_exp=1):

        if type == 'exp':
            self.kernel = kernel_exp(self,num_exp=num_exp)
        elif type == 'pow':
            self.kernel = kernel_pow(self)

        return self

    def set_baseline(self,type,num_basis=4,l_prop=None,l_custom=None):

        if type == 'const':
            self.baseline = baseline_const(self)
        elif type == 'loglinear':
            self.baseline = baseline_loglinear(self,num_basis)
        elif type == 'plinear':
            self.baseline = baseline_plinear(self,num_basis)
        elif type == 'prop':
            self.baseline = baseline_prop(self,l_prop)
        elif type == 'custom':
            self.baseline = baseline_custom(self,l_custom)

        return self

    def set_itv(self,itv):
        self.itv = itv
        return self

    def set_parameter(self,parameter):
        self.parameter = parameter
        self.para = indexed_ndarray().create_from_dict(parameter)
        return self

    ### l
    def l_baseline(self,x):
        para = self.para
        return self.baseline.l(para,x)

    def tl(self):
        T = self.T
        itv = self.itv
        para = self.para
        l_kernel_sequential = self.kernel.l_sequential(para)
        [t,l_kernel] = tl_kernel(l_kernel_sequential,T,itv)
        l_baseline = self.baseline.l(para,t)
        return [t,l_kernel+l_baseline,l_baseline]
    
    ### branching ratio
    def br(self):
        return self.kernel.branching_ratio(self.para)
    
    ### plot
    def plot_l(self):
        T = self.T
        [t,l,l_baseline] = self.tl()
        plot_l(T,t,l,l_baseline)

    def plot_N(self):
        T = self.T
        itv = self.itv
        plot_N(T,itv)

class simulator(base_class):

    def simulate(self):
        itv = self.itv;
        para = self.para
        l_kernel_sequential = self.kernel.l_sequential(para)
        l_baseline = self.l_baseline
        T = simulate(l_kernel_sequential,l_baseline,itv)
        self.T = T
        return T

class estimator(base_class):

    def fit(self,T,prior=[],opt=[],merge=[]):
        itv = self.itv
        T = np.array(T); T = T[(itv[0]<T)&(T<itv[1])].copy();
        self.T = T
        kernel = self.kernel.type; baseline = self.baseline.type;

        #baseline
        if baseline == 'const':
            para_length_baseline =    {'mu':1 }
            para_exp_baseline =       {'mu':True }
            para_ini_baseline =       {'mu':0.5*len(T)/(itv[1]-itv[0]) }
            para_step_Q_baseline =    {'mu':0.2 }
            para_step_diff_baseline = {'mu':0.01 }

        elif baseline == 'loglinear':
            num_basis = self.baseline.num_basis;
            para_length_baseline =    {'mu':num_basis }
            para_exp_baseline =       {'mu':False }
            para_ini_baseline =       {'mu':np.log( 0.5*len(T)/(itv[1]-itv[0]) ) * np.ones(num_basis) }
            para_step_Q_baseline =    {'mu':0.2 * np.ones(num_basis) }
            para_step_diff_baseline = {'mu':0.01 * np.ones(num_basis) }
            self.baseline.loglinear = loglinear_COS_intensity(itv,num_basis).set_x(T)

        elif baseline == 'plinear':
            num_basis = self.baseline.num_basis;
            para_length_baseline =    {'mu':num_basis }
            para_exp_baseline =       {'mu':True }
            para_ini_baseline =       {'mu':0.5*len(T)/(itv[1]-itv[0])*np.ones(num_basis) }
            para_step_Q_baseline =    {'mu':0.2 * np.ones(num_basis) }
            para_step_diff_baseline = {'mu':0.01 * np.ones(num_basis) }
            self.baseline.plinear = plinear_intensity(itv,num_basis).set_x(T)

        elif baseline == 'prop':
            baseline_prop = self.baseline.l_prop
            para_length_baseline =    {'mu':1 }
            para_exp_baseline =       {'mu':True }
            para_ini_baseline =       {'mu':0.5*len(T) }
            para_step_Q_baseline =    {'mu':0.2 }
            para_step_diff_baseline = {'mu':0.01 }
            self.baseline.l_prop_T = baseline_prop(T)

        #kernel
        if kernel == 'exp':
            num_exp = self.kernel.num_exp
            para_list_kernel = ['alpha','beta']
            para_length_kernel =    {'alpha':num_exp, 'beta':num_exp                                                                   }
            para_exp_kernel =       {'alpha':True, 'beta':True                                                                         }
            para_ini_kernel =       {'alpha':(np.arange(num_exp)+1.0)*0.5/np.sum(np.arange(num_exp)+1.0), 'beta':np.ones(num_exp)      }
            para_step_Q_kernel =    {'alpha':np.ones(num_exp)*0.2,                                        'beta':np.ones(num_exp)*0.2  }
            para_step_diff_kernel = {'alpha':np.ones(num_exp)*0.01,                                       'beta':np.ones(num_exp)*0.01 }

        elif kernel == 'pow':
            para_list_kernel = ['k','p','c']
            para_length_kernel =    {'k': 1,    'p':1,    'c':1    }
            para_exp_kernel =       {'k': True, 'p':True, 'c':True }
            para_ini_kernel =       {'k': 0.25, 'p':1.5,  'c':1.0  }
            para_step_Q_kernel =    {'k': 0.2,  'p':0.2,  'c':0.2  }
            para_step_diff_kernel = {'k': 0.01, 'p':0.01, 'c':0.01 }

        stg = {}
        stg['para_list'] =  ['mu'] + para_list_kernel
        stg['para_length']    = dict(para_length_baseline,    **para_length_kernel)
        stg['para_exp']       = dict(para_exp_baseline,       **para_exp_kernel)
        stg['para_ini']       = dict(para_ini_baseline,       **para_ini_kernel)
        stg['para_step_Q']    = dict(para_step_Q_baseline,    **para_step_Q_kernel)
        stg['para_step_diff'] = dict(para_step_diff_baseline, **para_step_diff_kernel)
        self.stg = stg

        [para,L,ste,G_norm,i_loop] = Quasi_Newton(self,prior,merge,opt)

        self.para = para
        self.parameter = para.to_dict()
        self.L = L
        self.AIC = -2.0*(L-len(para))
        self.br = self.kernel.branching_ratio(para)
        self.ste = ste
        self.i_loop = i_loop

        return self

    def LG(self,para,only_L=False):
        return LG_HAWKES(self,para)

    def t_trans(self):
        para = self.para
        T = self.T
        itv = self.itv
        l_baseline = lambda t: self.baseline.l(para,t)
        kernel_int = lambda itv: self.kernel.int(para,itv[0],itv[1])
        return t_trans(l_baseline,kernel_int,T,itv)

    def predict(self,en_f,num_seq=1):
        T = self.T
        itv = self.itv;
        para = self.para
        l_kernel_sequential = self.kernel.l_sequential(para)
        l_baseline = self.l_baseline
        l_kernel_sequential.initialize([T,itv[1]])
        T_pred = []
        for i in range(num_seq):
            l_kernel_sequential.reset()
            T_pred.append( simulate(l_kernel_sequential,l_baseline,[itv[1],en_f]) )
        self.en_f = en_f
        self.T_pred = T_pred
        return T_pred
    
    def plot_N_pred(self):
        T = self.T
        T_pred = self.T_pred
        itv = self.itv
        en_f = self.en_f
        plot_N_pred(T,T_pred,itv,en_f)
        

##########################################################################################################
## wrapper to routines
##########################################################################################################
def LG_HAWKES(model,para):

    n = len(model.T)

    [l_baseline,dl_baseline]     = model.baseline.LG_SUM(para)
    [Int_baseline,dInt_baseline] = model.baseline.LG_INT(para)
    [l_kernel,dl_kernel]         = model.kernel.LG_SUM(para)
    [Int_kernel,dInt_kernel]     = model.kernel.LG_INT(para)

    l = l_baseline + l_kernel
    Int = Int_baseline + Int_kernel

    dl = para.inherit().initialize_values(n).insert_values_from_dict(dl_baseline).insert_values_from_dict(dl_kernel).values
    dInt = para.inherit().initialize_values(1).insert_values_from_dict(dInt_baseline).insert_values_from_dict(dInt_kernel).values

    L = np.sum(np.log(l)) - Int
    G = (dl/l).sum(axis=1) - dInt
    G = para.inherit().set_values(G)

    return [L,G]

###########################
class baseline_const():

    def __init__(self,model):
        self.type = 'const'
        self.model = model

    def LG_SUM(self,para):
        T = self.model.T
        return LG_baseline_const_SUM(T,para)

    def LG_INT(self,para):
        [st,en] = self.model.itv
        return LG_baseline_const_INT(para,st,en)

    def l(self,para,t):
        return l_baseline_const(para,t)

###########################
class baseline_loglinear():

    def __init__(self,model,num_basis):
        self.type = 'loglinear'
        self.num_basis = num_basis
        self.model = model

    def LG_SUM(self,para):
        loglinear = self.loglinear
        return LG_baseline_loglinear_SUM(para,loglinear)

    def LG_INT(self,para):
        loglinear = self.loglinear
        return LG_baseline_loglinear_INT(para,loglinear)

    def l(self,para,t):
        loglinear = self.loglinear
        return l_baseline_loglinear(para,loglinear,t)

###########################
class baseline_plinear():

    def __init__(self,model,num_basis):
        self.type = 'plinear'
        self.num_basis = num_basis
        self.model = model

    def LG_SUM(self,para):
        plinear = self.plinear
        return LG_baseline_plinear_SUM(para,plinear)

    def LG_INT(self,para):
        plinear = self.plinear
        return LG_baseline_plinear_INT(para,plinear)

    def l(self,para,t):
        plinear = self.plinear
        return l_baseline_plinear(para,plinear,t)

###########################
class baseline_prop():

    def __init__(self,model,l_prop):
        self.type = 'prop'
        self.l_prop = l_prop
        self.model = model

    def LG_SUM(self,para):
        l_prop_T = self.l_prop_T
        return LG_baseline_prop_SUM(para,l_prop_T)

    def LG_INT(self,para):
        return LG_baseline_prop_INT(para)

    def l(self,para,t):
        l_prop = self.l_prop
        return l_baseline_prop(para,l_prop,t)

###########################
class baseline_custom():

    def __init__(self,model,l_custom):
        self.type = 'custom'
        self.l_custom = l_custom
        self.model = model

    def l(self,para,t):
        return self.l_custom(t)

###########################
class kernel_exp():

    def __init__(self,model,num_exp=1):
        self.type = 'exp'
        self.num_exp = num_exp
        self.model = model

    def LG_SUM(self,para):
        T = self.model.T
        num_exp = self.num_exp
        return LG_kernel_SUM_exp(T,para,num_exp)

    def LG_INT(self,para):
        T = self.model.T
        [_,en] = self.model.itv
        num_exp = self.num_exp
        return LG_kernal_INT_exp(T,para,num_exp,en)

    def l_sequential(self,para):
        num_exp = self.num_exp
        return l_kernel_sequential_exp(para,num_exp)

    def form(self,para,x):
        num_exp = self.num_exp
        return kernel_form_exp(para,num_exp,x)

    def int(self,para,x1,x2):
        num_exp = self.num_exp
        return kernel_int_exp(para,num_exp,x1,x2)

    def branching_ratio(self,para):
        return branching_ratio_exp(para)

###########################
class kernel_pow():

    def __init__(self,model):
        self.type = 'pow'
        self.model = model

    def LG_SUM(self,para):
        T = self.model.T
        return LG_kernel_SUM_pow(T,para)

    def LG_INT(self,para):
        T = self.model.T
        [_,en] = self.model.itv
        return LG_kernal_INT_pow(T,para,en)

    def l_sequential(self,para):
        return l_kernel_sequential_pow(para)

    def form(self,para,x):
        return kernel_form_pow(para,x)

    def int(self,para,x1,x2):
        return kernel_int_pow(para,x1,x2)

    def branching_ratio(self,para):
        return branching_ratio_pow(para)

##########################################################################################################
## Base routines
##########################################################################################################

############################## baseline rate: const
def LG_baseline_const_SUM(T,para):
    mu = para['mu']
    n = len(T)
    l = mu*np.ones(n)
    dl = {'mu':np.ones(n)}
    return [l,dl]

def LG_baseline_const_INT(para,st,en):
    mu = para['mu']
    Int = mu*(en-st)
    dInt = {'mu':en-st}
    return [Int,dInt]

def l_baseline_const(para,x):
    mu = para['mu']
    return mu if isinstance(x,numbers.Number) else mu*np.ones_like(x)

############################## baseline rate: basis function (cosine bump)
def LG_baseline_loglinear_SUM(para,loglinear):
    loglinear.set_coef(para['mu'])
    [l,dl] = loglinear.get_y_dy()
    dl = {'mu':np.transpose(dl)}
    return [l,dl]

def LG_baseline_loglinear_INT(para,loglinear):
    loglinear.set_coef(para['mu'])
    [Int,dInt] = loglinear.get_int_dint()
    dInt = {'mu':dInt}
    return [Int,dInt]

def l_baseline_loglinear(para,loglinear,t):
    return loglinear.set_coef(para['mu']).get_y_at(t)

############################## baseline rate: basis function (piecewise linear)
def LG_baseline_plinear_SUM(para,plinear):
    plinear.set_coef(para['mu'])
    [l,dl] = plinear.get_y_dy()
    dl = {'mu':np.transpose(dl)}
    return [l,dl]

def LG_baseline_plinear_INT(para,plinear):
    plinear.set_coef(para['mu'])
    [Int,dInt] = plinear.get_int_dint()
    dInt = {'mu':dInt}
    return [Int,dInt]

def l_baseline_plinear(para,plinear,t):
    return plinear.set_coef(para['mu']).get_y_at(t)

############################## baseline rate: scaled model
def LG_baseline_prop_SUM(para,baseline_prop_T):
    coef = para['mu']
    l = baseline_prop_T * coef
    dl = {'mu':baseline_prop_T}
    return [l,dl]

def LG_baseline_prop_INT(para):
    coef = para['mu']
    Int = coef
    dInt = {'mu':1.0}
    return [Int,dInt]

def l_baseline_prop(para,baseline_prop,t):
    return para['mu'] * baseline_prop(t)


############################## kernel Exp
def LG_kernel_SUM_exp(T,para,num_exp):

    alpha = [para['alpha']] if num_exp == 1 else para['alpha']
    beta  = [para['beta']]  if num_exp == 1 else para['beta']

    n = len(T)
    l = np.zeros(n)
    dl = {}

    for i in range(num_exp):
        if cython_import:
            [l_i,dl_a_i,dl_b_i] = LG_kernel_SUM_exp_i_cython(T,alpha[i],beta[i])
        else:
            [l_i,dl_i_a,dl_i_b] = LG_kernel_SUM_exp_i_python(T,alpha[i],beta[i])

        l = l + l_i
        dl[('alpha',i)] = dl_a_i
        dl[('beta',i)] = dl_b_i

    return [l,dl]

def LG_kernel_SUM_exp_i_python(T,alpha,beta):

    n = len(T)
    l    = np.zeros(n)
    dl_a = np.zeros(n)
    dl_b = np.zeros(n)

    '''
    for i in np.arange(1,n):
        l[i]    = ( alpha*beta*np.exp(-beta*(T[i]-T[0:i]))                                                        ).sum()
        dl_a[i] = (       beta*np.exp(-beta*(T[i]-T[0:i]))                                                        ).sum()
        dl_b[i] = ( alpha*     np.exp(-beta*(T[i]-T[0:i])) - alpha*beta*(T[i]-T[0:i])*np.exp(-beta*(T[i]-T[0:i])) ).sum()

    '''

    dT = T[1:]-T[:-1]
    x = 0; x_a = 0; x_b = 0;

    for i in np.arange(n-1):
        r = np.exp(-beta*dT[i])
        x   = ( x + alpha*beta ) * r
        x_a = ( x_a + beta ) * r
        x_b = ( x_b + alpha ) * r - x*dT[i]
        l[i+1] = x; dl_a[i+1] = x_a; dl_b[i+1] = x_b;

    return [l,dl_a,dl_b]

def LG_kernal_INT_exp(T,para,num_exp,en):

    alpha = [para['alpha']] if num_exp == 1 else para['alpha']
    beta  = [para['beta']]  if num_exp == 1 else para['beta']

    Int = 0
    dInt = {}

    for i in range(num_exp):
        [Int_i,dInt_i_a,dInt_i_b] = LG_kernel_INT_exp_i(T,alpha[i],beta[i],en)
        Int += Int_i
        dInt[('alpha',i)] = dInt_i_a
        dInt[('beta',i)] = dInt_i_b

    return [Int,dInt]

def LG_kernel_INT_exp_i(T,alpha,beta,en):
    tau = en-T
    Int = ( alpha*( 1 - np.exp(-beta*tau) ) ).sum()
    dInt_a = Int/alpha
    dInt_b = ( alpha*tau* np.exp(-beta*tau) ) .sum()
    return [Int,dInt_a,dInt_b]

def kernel_form_exp(para,num_exp,x):
    alpha = [para['alpha']] if num_exp == 1 else para['alpha']
    beta  = [para['beta']]  if num_exp == 1 else para['beta']
    l = np.zeros_like(x)
    for i in range(num_exp):
        l += alpha[i]*beta[i]*np.exp(-beta[i]*x)
    return l

def kernel_int_exp(para,num_exp,x1,x2):
    alpha = [para['alpha']] if num_exp == 1 else para['alpha']
    beta  = [para['beta']]  if num_exp == 1 else para['beta']
    Int = np.zeros_like(x1)
    x1 = x1*np.heaviside(x1,0); x2 = x2*np.heaviside(x2,0);
    for i in range(num_exp):
        Int += alpha[i] * ( np.exp( -beta[i]*x1 ) - np.exp( -beta[i]*x2 ) )
    return Int

def branching_ratio_exp(para):
    alpha = para['alpha']
    br = alpha if isinstance(alpha,numbers.Number) else alpha.sum()
    return br

class l_kernel_sequential_exp:

    def __init__(self,para,num_exp):
        self.para = para
        self.g = 0 if num_exp == 1 else np.zeros_like(para['alpha'])
        self.l = 0

    def step_forward(self,step):
        beta = self.para['beta']; g = self.g;
        g = g*np.exp(-beta*step)
        l = g.sum()
        self.g  = g
        self.l = l

    def event(self):
        alpha = self.para['alpha']; beta = self.para['beta']; g = self.g;
        g += alpha*beta
        l = g.sum()
        self.g  = g
        self.l = l

    def initialize(self,history):
        initialize_l_sequential(self,history)

    def reset(self):
        self.g = self.g0 if isinstance(self.g0,numbers.Number) else self.g0.copy()
        self.l = self.l0

############################## kernel Pow
def LG_kernel_SUM_pow(T,para):
    k = para['k']; p = para['p']; c = para['c'];
    if cython_import:
        [l,dl_k,dl_p,dl_c] = LG_kernel_SUM_pow_cython(T,k,p,c)
    else:
        [l,dl_k,dl_p,dl_c] = LG_kernel_SUM_pow_python(T,k,p,c)
    dl = {'k':dl_k, 'p':dl_p, 'c':dl_c}
    return [l,dl]

def LG_kernel_SUM_pow_python(T,k,p,c):

    n = len(T)

    '''
    l    = np.zeros(n)
    dl_k = np.zeros(n)
    dl_p = np.zeros(n)
    dl_c = np.zeros(n)

    for i in np.arange(1,n):
        l[i]    = ( k*(T[i]-T[0:i]+c)**(-p)                             ).sum()
        dl_k[i] = (   (T[i]-T[0:i]+c)**(-p)                             ).sum()
        dl_p[i] = ( -k*np.log(T[i]-T[0:i]+c)*(T[i]-T[0:i]+c)**(-p)      ).sum()
        dl_c[i] = ( -k*p*(T[i]-T[0:i]+c)**(-p-1)                        ).sum()
    '''


    num_div = 16
    delta = 1.0/num_div
    s = np.linspace(-9,9,num_div*18+1)
    log_phi = s-np.exp(-s)
    log_dphi = log_phi + np.log(1+np.exp(-s))
    phi = np.exp(log_phi)   # phi = np.exp(s-np.exp(-s))
    dphi = np.exp(log_dphi) # dphi = phi*(1+np.exp(-s))

    H   = delta * k * np.exp( log_dphi + (p-1)*log_phi - c*phi ) / gamma(p)
    H_p = delta * k * np.exp( log_dphi + (p-1)*log_phi - c*phi ) / gamma(p) * (log_phi-digamma(p))
    H_c = delta * k * np.exp( log_dphi +     p*log_phi - c*phi ) / gamma(p) * (-1)

    g = np.zeros_like(s)
    l = np.zeros(n)
    dl_k = np.zeros(n)
    dl_p = np.zeros(n)
    dl_c = np.zeros(n)

    for i in range(n-1):
        g = (g+1)*np.exp( - phi*(T[i+1]-T[i]) )
        l[i+1] = g.dot(H)
        dl_k[i+1] = g.dot(H)/k
        dl_p[i+1] = g.dot(H_p)
        dl_c[i+1] = g.dot(H_c)


    return [l,dl_k,dl_p,dl_c]

def LG_kernal_INT_pow(T,para,en):
    k = para['k']; p = para['p']; c = para['c'];
    Int_array = k/(-p+1)*( (en-T+c)**(-p+1) - (c)**(-p+1) )
    Int = Int_array.sum()
    dInt = {}
    dInt['k'] = Int/k
    dInt['p'] = Int/(-p+1) + ( k/(-p+1)*( -np.log(en-T+c)*(en-T+c)**(-p+1) + np.log(c)*(c)**(-p+1) ) ).sum()
    dInt['c'] = ( k *( (en-T+c)**(-p) - (c)**(-p) ) ).sum()

    return [Int,dInt]

def kernel_form_pow(para,x):
    k = para['k']; p = para['p']; c = para['c'];
    return k*(x+c)**(-p)

def kernel_int_pow(para,x1,x2):
    k = para['k']; p = para['p']; c = para['c'];
    x1 = x1*np.heaviside(x1,0); x2 = x2*np.heaviside(x2,0);
    Int = k / (-p+1) * ( (x2+c)**(-p+1) - (x1+c)**(-p+1) )
    return Int

def branching_ratio_pow(para):
    k = para['k']; p = para['p']; c = para['c'];
    br = -k/(-p+1)*c**(-p+1) if p>1 else np.inf
    return br

class l_kernel_sequential_pow:

    def __init__(self,para):
        self.para = para

        k = para['k']; p = para['p']; c = para['c'];
        num_div = 16
        delta = 1.0/num_div
        s = np.linspace(-9,9,num_div*18+1)
        log_phi = s-np.exp(-s)
        log_dphi = log_phi + np.log(1+np.exp(-s))
        phi = np.exp(log_phi)   # phi = np.exp(s-np.exp(-s))
        dphi = np.exp(log_dphi) # dphi = phi*(1+np.exp(-s))
        H   = delta * k * np.exp( log_dphi + (p-1)*log_phi - c*phi ) / gamma(p)
        g = np.zeros_like(s)
        self.g = g
        self.phi = phi
        self.H = H
        self.l = 0

    def step_forward(self,step):
        g = self.g; phi = self.phi; H = self.H;
        g = g*np.exp(-phi*step)
        l = g.dot(H)
        self.g = g
        self.l = l

    def event(self):
        g = self.g; H = self.H;
        g += 1.0
        l = g.dot(H)
        self.g = g
        self.l = l

    def initialize(self,history):
        initialize_l_sequential(self,history)

    def reset(self):
        self.g = self.g0
        self.l = self.l0

###########################################################################################
###########################################################################################
## general routine
###########################################################################################
###########################################################################################
def t_trans(l_baseline,kernel_int,T,itv):
    [st,en] = itv
    n = len(T)
    T_ext = np.hstack([st,T,en])
    Int_ext = np.zeros(n+1)

    for i in range(n+1):
        Int_ext[i] = kernel_int([T_ext[i]-T,T_ext[i+1]-T]).sum() + (l_baseline(T_ext[i])+l_baseline(T_ext[i+1]))*(T_ext[i+1]-T_ext[i])/2.0

    Int_ext_cumsum = Int_ext.cumsum()

    return [Int_ext[:n],[0,Int_ext_cumsum[-1]]]

def tl_kernel(l_kernel_sequential,T,itv):

    [st,en] = itv
    t = np.hstack([ np.linspace(t[0],t[1],30) for t in np.vstack([np.hstack([st,T]),np.hstack([T,en])]).transpose() ])
    mark = np.zeros_like(t,dtype='i8')
    mark[-31::-30] = 1
    l = np.zeros_like(t)

    for i in range(t.shape[0]-1):

        if mark[i] == 1:
            l_kernel_sequential.event()

        l_kernel_sequential.step_forward(t[i+1]-t[i])
        l[i+1] = l_kernel_sequential.l

    return [t,l]

def initialize_l_sequential(l_kernel_sequential,history):
    T_ext = np.hstack(history) # [T,en] = history
    n = len(T_ext)-1
    for i in range(n):
        l_kernel_sequential.event()
        l_kernel_sequential.step_forward(T_ext[i+1]-T_ext[i])
    l_kernel_sequential.g0 = l_kernel_sequential.g
    l_kernel_sequential.l0 = l_kernel_sequential.l

def simulate(l_kernel_sequential,l_baseline,itv):

    N_MAX = 1000001
    T = np.empty(N_MAX,dtype='f8')
    [st,en] = itv
    x = st;
    l0 = l_baseline(st)+l_kernel_sequential.l;
    i = 0;

    while 1:

        step = np.random.exponential()/l0
        x += step
        l_kernel_sequential.step_forward(step)
        l1 = l_baseline(x) + l_kernel_sequential.l

        if (x>en) or (i==N_MAX):
            break

        if np.random.rand() < l1/l0: ## Fire
            T[i] = x
            i += 1
            l_kernel_sequential.event()

        l0 = l_baseline(x) + l_kernel_sequential.l

    T = T[:i]

    return T

###########################################################################################
###########################################################################################
## graph routine
###########################################################################################
###########################################################################################
def plot_l(T,x,l,l_baseline):

    plt.figure(figsize=(5,5), dpi=100)
    mpl.rc('font', size=12, family='Arial')
    mpl.rc('axes',titlesize=12)
    mpl.rc('pdf',fonttype=42)

    l_max = l.max()

    plt.plot(x,l,'k-',lw=1)
    plt.plot(x,l_baseline,'k:',lw=1)
    plt.plot(np.hstack([ [t,t,np.NaN] for t in T]),np.tile([l_max*1.1,l_max*1.15,np.NaN],len(T)),'k-',linewidth=0.5)
    #plt.plot(T,l_max*1.1*np.ones_like(T),'bo',markersize=1)
    plt.xlim([x[0],x[-1]])
    plt.ylim([0,l_max*1.15])
    plt.xlabel('time')
    plt.ylabel(r'$\lambda(t|H_t)$')
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

def plot_N(T,itv):

    plt.figure(figsize=(5,5), dpi=100)
    mpl.rc('font', size=12, family='Arial')
    mpl.rc('axes',titlesize=12)
    mpl.rc('pdf',fonttype=42)

    [st,en] = itv
    n = len(T)
    x = np.hstack([st,np.repeat(T,2),en])
    y = np.repeat(np.arange(n+1),2)

    plt.plot(x,y,'k-')
    plt.plot(np.hstack([ [t,t,np.NaN] for t in T]),np.tile([n*1.1,n*1.15,np.NaN],len(T)),'k-',linewidth=0.5)
    plt.xlim(itv)
    plt.ylim([0,1.15*n])
    plt.xlabel('time')
    plt.ylabel(r'$N(0,t)$')
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

def plot_N_pred(T,T_pred,itv,en_f):

    plt.figure(figsize=(5,5), dpi=100)
    mpl.rc('font', size=12, family='Arial')
    mpl.rc('axes',titlesize=12)
    mpl.rc('pdf',fonttype=42)

    [st,en] = itv
    n = len(T)
    x = np.hstack([st,np.repeat(T,2),en])
    y = np.repeat(np.arange(n+1),2)
    n_pred_max = np.max([ len(T_i) for T_i in T_pred ])
    
    plt.plot(x,y,'k-')
    plt.plot([en,en],[0,n+n_pred_max],'k--')

    for i in range(len(T_pred)):
        n_pred = len(T_pred[i])
        x = np.hstack([en,np.repeat(T_pred[i],2),en_f])
        y = np.repeat(np.arange(n_pred+1),2) + n
        plt.plot(x,y,'-',color=[0.7,0.7,1.0],lw=0.5)

    plt.xlim([st,en_f])
    plt.ylim([0,n+n_pred_max])
    plt.xlabel('time')
    plt.ylabel(r'$N(0,t)$')
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
