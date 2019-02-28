from __future__ import division
from __future__ import print_function

import sys,time,datetime,copy,subprocess,itertools,pickle,warnings,numbers

import numpy as np
import scipy as sp
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib as mpl
import matplotlib.gridspec as gridspec

from scipy.special import gamma,digamma

from .StatTool import Quasi_Newton,indexed_ndarray,merge_stg
from .BasisFunction import loglinear_COS,plinear

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
    def set_kernel(self,type,**kwargs):
        kernel_class = {'exp':kernel_exp, 'pow':kernel_pow}
        self.kernel = kernel_class[type](self,**kwargs)
        return self

    def set_baseline(self,type,**kwargs):
        baseline_class = {'const':baseline_const,'loglinear':baseline_loglinear,'plinear':baseline_plinear,'prop':baseline_prop,'custom':baseline_custom}
        self.baseline = baseline_class[type](self,**kwargs)
        return self

    def set_parameter(self,parameter):
        self.parameter = parameter
        self.para = indexed_ndarray().create_from_dict(parameter)
        return self

    ### l
    def tl(self):
        T = self.Data['T']
        itv = self.itv
        l_kernel_sequential = self.kernel.l_sequential()
        [t,l_kernel] = tl_kernel(l_kernel_sequential,T,itv)
        l_baseline = self.baseline.l(t)
        return [t,l_kernel+l_baseline,l_baseline]

    def t_trans(self):
        T = self.Data['T']
        itv = self.itv
        l_baseline = self.baseline.l
        kernel_int = self.kernel.int
        [T_trans,itv_trans] = t_trans(l_baseline,kernel_int,T,itv)
        self.T_trans = T_trans
        self.itv_trans = itv_trans
        return [T_trans,itv_trans]

    ### branching ratio
    def branching_ratio(self):
        return self.kernel.branching_ratio()

    ### plot
    def plot_l(self):
        T = self.Data['T']
        [t,l,l_baseline] = self.tl()
        plot_l(T,t,l,l_baseline)

    def plot_N(self):
        T = self.Data['T']
        itv = self.itv
        plot_N(T,itv)

    def plot_KS(self):
        self.t_trans()
        T_trans = self.T_trans
        itv_trans = self.itv_trans
        plot_KS(T_trans,itv_trans)

class simulator(base_class):

    def simulate(self,itv):
        self.itv = itv
        l_kernel_sequential = self.kernel.l_sequential()
        l_baseline = self.baseline.l
        T = simulate(l_kernel_sequential,l_baseline,itv)
        self.Data = {'T':T}
        return T

class estimator(base_class):

    def fit(self,T,itv,prior=[],opt=[],merge=[]):
        self.itv = itv
        T = np.array(T); T = T[(itv[0]<T)&(T<itv[1])].copy();
        self.Data = {'T':T}

        stg_b = self.baseline.prep_fit()
        stg_k = self.kernel.prep_fit()
        stg = merge_stg([stg_b,stg_k])
        self.stg = stg

        [para,L,ste,G_norm,i_loop] = Quasi_Newton(self,prior,merge,opt)

        self.para = para
        self.parameter = para.to_dict()
        self.L = L
        self.AIC = -2.0*(L-len(para))
        self.br = self.kernel.branching_ratio()
        self.ste = ste
        self.i_loop = i_loop

        return self

    def LG(self,para,only_L=False):
        self.para = para
        return LG_SelfExcitingPointProcess(self)

    def predict(self,en_f,num_seq=1):
        T = self.Data['T']
        itv = self.itv;
        l_kernel_sequential = self.kernel.l_sequential()
        l_baseline = self.baseline.l
        l_kernel_sequential.initialize([T,itv[1]])
        T_pred = []
        for i in range(num_seq):
            l_kernel_sequential.reset()
            T_pred.append( simulate(l_kernel_sequential,l_baseline,[itv[1],en_f]) )
        self.en_f = en_f
        self.T_pred = T_pred
        return T_pred

    def plot_N_pred(self):
        T = self.Data['T']
        T_pred = self.T_pred
        itv = self.itv
        en_f = self.en_f
        plot_N_pred(T,T_pred,itv,en_f)


##########################################################################################################
## LG for general point process
##########################################################################################################
def LG_SelfExcitingPointProcess(model):

    n = len(model.Data['T'])
    para = model.para

    [l_baseline,dl_baseline]     = model.baseline.LG_SUM()
    [Int_baseline,dInt_baseline] = model.baseline.LG_INT()
    [l_kernel,dl_kernel]         = model.kernel.LG_SUM()
    [Int_kernel,dInt_kernel]     = model.kernel.LG_INT()

    l = l_baseline + l_kernel
    Int = Int_baseline + Int_kernel

    dl   = para.inherit().initialize_values(n).insert_values_from_dict(dl_baseline  ).insert_values_from_dict(dl_kernel  ).values
    dInt = para.inherit().initialize_values(1).insert_values_from_dict(dInt_baseline).insert_values_from_dict(dInt_kernel).values

    L = np.sum(np.log(l)) - Int
    G = (dl/l).sum(axis=1) - dInt
    G = para.inherit().set_values(G)

    return [L,G]

##########################################################################################################
## baseline class
##########################################################################################################
class baseline_const:

    def __init__(self,model):
        self.type = 'const'
        self.model = model

    def prep_fit(self):
        itv = self.model.itv
        T = self.model.Data['T']
        n = len(T)

        list =      ['mu']
        length =    {'mu':1 }
        exp =       {'mu':True }
        ini =       {'mu':0.5*len(T)/(itv[1]-itv[0]) }
        step_Q =    {'mu':0.2 }
        step_diff = {'mu':0.01 }

        return {"list":list,'length':length,'exp':exp,'ini':ini,'step_Q':step_Q,'step_diff':step_diff}

    def LG_SUM(self):
        para = self.model.para
        Data = self.model.Data
        n = len(Data['T'])
        mu = para['mu']
        l = mu*np.ones(n)
        dl = {'mu':np.ones(n)}
        return [l,dl]

    def LG_INT(self):
        para = self.model.para
        mu = para['mu']
        [st,en] = self.model.itv
        Int = mu*(en-st)
        dInt = {'mu':en-st}
        return [Int,dInt]

    def l(self,t):
        para = self.model.para
        mu = para['mu']
        return mu if isinstance(t,numbers.Number) else mu*np.ones_like(t)

###########################
class baseline_loglinear:

    def __init__(self,model,num_basis):
        self.type = 'loglinear'
        self.num_basis = num_basis
        self.model = model

    def prep_fit(self):
        num_basis = self.num_basis;
        itv = self.model.itv
        T = self.model.Data['T']
        n = len(T)

        list =      ['mu']
        length =    {'mu':num_basis }
        exp =       {'mu':False }
        ini =       {'mu':np.log( 0.5*n/(itv[1]-itv[0]) ) * np.ones(num_basis) }
        step_Q =    {'mu':0.2 * np.ones(num_basis) }
        step_diff = {'mu':0.01 * np.ones(num_basis) }

        self.loglinear = loglinear_COS(itv,num_basis).set_x(T)

        return {"list":list,'length':length,'exp':exp,'ini':ini,'step_Q':step_Q,'step_diff':step_diff}

    def LG_SUM(self):
        para = self.model.para
        loglinear = self.loglinear
        coef = para['mu']
        loglinear.set_coef(coef)
        l = loglinear.get_y()
        dl = loglinear.get_dy()
        dl = {'mu':np.transpose(dl)}
        return [l,dl]

    def LG_INT(self):
        para = self.model.para
        loglinear = self.loglinear
        coef = para['mu']
        loglinear.set_coef(coef)
        Int = loglinear.get_int()
        dInt = loglinear.get_dint()
        dInt = {'mu':dInt}
        return [Int,dInt]

    def l(self,t):
        para = self.model.para
        loglinear = self.loglinear
        coef = para['mu']
        return loglinear.set_coef(coef).get_y_at(t)

###########################
class baseline_plinear:

    def __init__(self,model,num_basis):
        self.type = 'plinear'
        self.num_basis = num_basis
        self.model = model

    def prep_fit(self):
        itv = self.model.itv
        T = self.model.Data['T']
        n = len(T)
        num_basis = self.num_basis

        list   =    ['mu']
        length =    {'mu':num_basis }
        exp =       {'mu':True }
        ini =       {'mu':0.5*n/(itv[1]-itv[0])*np.ones(num_basis) }
        step_Q =    {'mu':0.2 * np.ones(num_basis) }
        step_diff = {'mu':0.01 * np.ones(num_basis) }

        self.plinear = plinear(itv,num_basis).set_x(T)

        return {"list":list,'length':length,'exp':exp,'ini':ini,'step_Q':step_Q,'step_diff':step_diff}

    def LG_SUM(self):
        para = self.model.para
        plinear = self.plinear
        coef = para['mu']
        plinear.set_coef(coef)
        l = plinear.get_y()
        dl = plinear.get_dy()
        dl = {'mu':np.transpose(dl)}
        return [l,dl]

    def LG_INT(self):
        para = self.model.para
        plinear = self.plinear
        coef = para['mu']
        plinear.set_coef(coef)
        Int = plinear.get_int()
        dInt = plinear.get_dint()
        dInt = {'mu':dInt}
        return [Int,dInt]

    def l(self,t):
        para = self.model.para
        plinear = self.plinear
        coef = para['mu']
        return plinear.set_coef(coef).get_y_at(t)

###########################
class baseline_prop:

    def __init__(self,model,l_prop):
        self.type = 'prop'
        self.l_prop = l_prop
        self.model = model

    def prep_fit(self):
        T = self.model.Data['T']
        n = len(T)

        list =      ['mu']
        length =    {'mu':1 }
        exp =       {'mu':True }
        ini =       {'mu':0.5*n }
        step_Q =    {'mu':0.2 }
        step_diff = {'mu':0.01 }

        self.l_prop_T = self.l_prop(T)

        return {"list":list,'length':length,'exp':exp,'ini':ini,'step_Q':step_Q,'step_diff':step_diff}

    def LG_SUM(self):
        para = self.model.para
        coef = para['coef']
        l_prop_T = self.l_prop_T
        l = baseline_prop_T * coef
        dl = {'mu':baseline_prop_T}
        return [l,dl]

    def LG_INT(self):
        para = self.model.para
        coef = para['coef']
        Int = coef
        dInt = {'mu':1.0}
        return [Int,dInt]

    def l(self,t):
        para = self.model.para
        coef = para['coef']
        l_prop = self.l_prop
        return coef * baseline_prop(t)

###########################
class baseline_custom:

    def __init__(self,model,l_custom):
        self.type = 'custom'
        self.l_custom = l_custom
        self.model = model

    def l(self,t):
        return self.l_custom(t)

##########################################################################################################
## kernel class
##########################################################################################################
class kernel_exp:

    def __init__(self,model,num_exp=1):
        self.type = 'exp'
        self.num_exp = num_exp
        self.model = model
        self.para_list = list( itertools.product(['alpha','beta'],range(num_exp)))

    def prep_fit(self):
        num_exp = self.num_exp
        list =      ['alpha','beta']
        length =    {'alpha':num_exp,                                                     'beta':num_exp               }
        exp =       {'alpha':True,                                                        'beta':True                  }
        ini =       {'alpha':(np.arange(num_exp)+1.0)*0.5/np.sum(np.arange(num_exp)+1.0), 'beta':np.ones(num_exp)      }
        step_Q =    {'alpha':np.ones(num_exp)*0.2,                                        'beta':np.ones(num_exp)*0.2  }
        step_diff = {'alpha':np.ones(num_exp)*0.01,                                       'beta':np.ones(num_exp)*0.01 }
        return {"list":list,'length':length,'exp':exp,'ini':ini,'step_Q':step_Q,'step_diff':step_diff}

    def LG_SUM(self):
        Data = self.model.Data

        if cython_import:
            para = self.model.para
            alpha = np.array(para['alpha']).reshape(-1)
            beta  = np.array(para['beta']).reshape(-1)
            num_exp = self.num_exp
            T = Data['T']
            n = len(T)
            l  = 0
            dl = {}
            for i in range(num_exp):
                [l_i,dl_a_i,dl_b_i] = LG_kernel_SUM_exp_i_cython(T, alpha[i], beta[i])
                l = l + l_i
                dl[('alpha',i)] = dl_a_i
                dl[('beta',i)] = dl_b_i
        else:
            l_kernel_sequential = self.l_sequential()
            [l,dl] = lg_kernel_sum_sequential(l_kernel_sequential,Data)

        """
        num_exp = self.num_exp
        T = Data['T']
        n = len(T)
        l  = np.zeros(n)
        dl = { key:np.zeros(n) for key in itertools.product(['alpha','beta'],range(num_exp)) }

        for i in np.arange(1,n):
            l[i] = self.func(T[i]-T[:i]).sum()
            dl_i = self.d_func(T[i]-T[:i])
            for j in range(num_exp):
                dl[('alpha',j)][i] = dl_i[('alpha',j)].sum()
                dl[('beta',j)][i] = dl_i[('beta',j)].sum()
        """

        return [l,dl]

    def LG_INT(self):
        Data = self.model.Data
        T = Data['T']
        [_,en] = self.model.itv
        num_exp = self.num_exp
        Int  = self.int(0,en-T).sum()
        dInt = { key:grad.sum() for key,grad in self.d_int(np.zeros_like(T),en-T).items() }
        return [Int,dInt]

    def func(self,x):
        para = self.model.para
        alpha = np.array(para['alpha']).reshape(-1)
        beta  = np.array(para['beta']).reshape(-1)
        num_exp = self.num_exp
        l = 0
        for i in range(num_exp):
            l = l + alpha[i] * beta[i] * np.exp( -beta[i] * x )
        return l

    def d_func(self,x):
        para = self.model.para
        alpha = np.array(para['alpha']).reshape(-1)
        beta  = np.array(para['beta']).reshape(-1)
        num_exp = self.num_exp
        dl = {}
        for i in range(num_exp):
            dl[('alpha',i)] = np.exp( -beta[i] * x ) * beta[i]
            dl[('beta',i) ] = np.exp( -beta[i] * x ) * ( alpha[i] - alpha[i] * beta[i] * x )
        return dl

    def int(self,x1,x2):
        para = self.model.para
        alpha = np.array(para['alpha']).reshape(-1)
        beta  = np.array(para['beta']).reshape(-1)
        num_exp = self.num_exp
        Int = 0
        for i in range(num_exp):
            Int = Int + alpha[i] * ( np.exp( -beta[i] * x1 ) - np.exp( -beta[i] * x2 ) )
        return Int

    def d_int(self,x1,x2):
        para = self.model.para
        alpha = np.array(para['alpha']).reshape(-1)
        beta  = np.array(para['beta']).reshape(-1)
        num_exp = self.num_exp
        dInt = {}
        for i in range(num_exp):
            dInt[('alpha',i)] = np.exp( -beta[i] * x1 ) - np.exp( -beta[i] * x2 )
            dInt[('beta',i) ] = alpha[i] * ( - x1 * np.exp( -beta[i] * x1 ) + x2 * np.exp( -beta[i] * x2 ) )
        return dInt

    def branching_ratio(self):
        para = self.model.para
        br = np.array(para['alpha']).sum()
        return br

    def l_sequential(self):
        para = self.model.para
        num_exp = self.num_exp
        return l_kernel_sequential_exp(para,num_exp)

class l_kernel_sequential_exp:

    def __init__(self,para,num_exp):
        self.alpha = np.array(para['alpha']).reshape(-1)
        self.beta = np.array(para['beta']).reshape(-1)
        self.num_exp = num_exp
        self.g   = np.zeros(num_exp)
        self.g_b = np.zeros(num_exp)
        self.l = 0
        self.dl = { key:0 for key in itertools.product(['alpha','beta'],range(num_exp)) }

    def step_forward(self,step):
        alpha = self.alpha; beta = self.beta; num_exp = self.num_exp;
        g = self.g; g_b = self.g_b
        r = np.exp(-beta*step)

        g   = g*r
        g_b = g_b*r - g*step
        l = g.sum()
        dl = { ('alpha',i):g[i]/alpha[i] for i in range(num_exp) }
        dl.update( { ('beta',i): g_b[i]  for i in range(num_exp) } )
        self.g = g
        self.g_b = g_b
        self.l = l
        self.dl = dl


    def event(self):
        alpha = self.alpha; beta = self.beta;
        g = self.g; g_b = self.g_b
        g = g + alpha*beta
        g_b = g_b + alpha
        l = g.sum()
        self.g  = g
        self.g_b = g_b
        self.l = l

    def initialize(self,history):
        initialize_l_sequential(self,history)

    def reset(self):
        self.g = self.g0
        self.l = self.l0

###########################
class kernel_pow():

    def __init__(self,model):
        self.type = 'pow'
        self.model = model
        self.para_list = ['k','p','c']

    def prep_fit(self):
        list = ['k','p','c']
        length =    {'k': 1,    'p':1,    'c':1    }
        exp =       {'k': True, 'p':True, 'c':True }
        ini =       {'k': 0.25, 'p':1.5,  'c':1.0  }
        step_Q =    {'k': 0.2,  'p':0.2,  'c':0.2  }
        step_diff = {'k': 0.01, 'p':0.01, 'c':0.01 }
        return {"list":list,'length':length,'exp':exp,'ini':ini,'step_Q':step_Q,'step_diff':step_diff}

    def LG_SUM(self):
        Data = self.model.Data
        l_kernel_sequential = self.l_sequential()
        [l,dl] = lg_kernel_sum_sequential(l_kernel_sequential,Data)

        """
        T = Data['T']
        n = len(T)
        l  = np.zeros(n)
        dl = {'k':np.zeros(n), 'p':np.zeros(n), 'c':np.zeros(n)}

        for i in np.arange(1,n):
            l[i] = self.func(T[i]-T[:i]).sum()
            dl_i = self.d_func(T[i]-T[:i])
            dl['k'][i] = dl_i['k'].sum()
            dl['p'][i] = dl_i['p'].sum()
            dl['c'][i] = dl_i['c'].sum()
        """

        return [l,dl]

    def LG_INT(self):
        Data = self.model.Data
        T = Data['T']
        [_,en] = self.model.itv
        Int  = self.int(0,en-T).sum()
        dInt = { key:grad.sum() for key,grad in self.d_int(0,en-T).items() }
        return [Int,dInt]

    def func(self,x):
        para = self.model.para
        k = para['k']; p = para['p']; c = para['c'];
        return k * (x+c)**(-p)

    def d_func(self,x):
        para = self.model.para
        k = para['k']; p = para['p']; c = para['c'];
        dl = {}
        dl['k']     =      (x+c)**(-p)
        dl['p']     = -k * (x+c)**(-p) * np.log(x+c)
        dl['c']     = -k * (x+c)**(-p-1) * p
        return dl

    def int(self,x1,x2):
        para = self.model.para
        k = para['k']; p = para['p']; c = para['c'];
        Int = k / (-p+1) * ( (x2+c)**(-p+1) - (x1+c)**(-p+1) )
        return Int

    def d_int(self,x1,x2):
        para = self.model.para
        k = para['k']; p = para['p']; c = para['c'];
        f1 = k / (-p+1) * (x1+c)**(-p+1)
        f2 = k / (-p+1) * (x2+c)**(-p+1)
        dInt = {}
        dInt['k']     = (f2-f1)/k
        dInt['p']     = (f2-f1)/(-p+1) + ( - f2*np.log(x2+c) + f1*np.log(x1+c) )
        dInt['c']     = ( f2/(x2+c) - f1/(x1+c) ) * (-p+1)
        return dInt

    def branching_ratio(self):
        para = self.model.para
        k = para['k']; p = para['p']; c = para['c'];
        br = -k/(-p+1)*c**(-p+1) if p>1 else np.inf
        return br

    def l_sequential(self):
        para = self.model.para
        return l_kernel_sequential_pow(para)

class l_kernel_sequential_pow:

    def __init__(self,para):
        k = para['k']; p = para['p']; c = para['c'];
        num_div = 16
        delta = 1.0/num_div
        s = np.linspace(-9,9,num_div*18+1)
        log_phi = s-np.exp(-s)
        log_dphi = log_phi + np.log(1+np.exp(-s))
        phi = np.exp(log_phi)   # phi = np.exp(s-np.exp(-s))

        H   = delta * k * np.exp( log_dphi + (p-1)*log_phi - c*phi ) / gamma(p)
        H_k = delta *     np.exp( log_dphi + (p-1)*log_phi - c*phi ) / gamma(p)
        H_p = delta * k * np.exp( log_dphi + (p-1)*log_phi - c*phi ) / gamma(p) * (log_phi-digamma(p))
        H_c = delta * k * np.exp( log_dphi +     p*log_phi - c*phi ) / gamma(p) * (-1)

        g = np.zeros_like(s)
        self.g = g
        self.l = 0
        self.dl = {'k':0, 'p':0, 'c':0}
        self.phi = phi
        self.H = H
        self.H_k = H_k
        self.H_p = H_p
        self.H_c = H_c

    def step_forward(self,step):
        g = self.g; phi = self.phi; H = self.H; H_k = self.H_k; H_p = self.H_p; H_c = self.H_c;
        g = g*np.exp(-phi*step)
        l = g.dot(H)
        dl = {'k':g.dot(H_k),'p':g.dot(H_p),'c':g.dot(H_c)}
        self.g = g
        self.l = l
        self.dl = dl

    def event(self):
        g = self.g; H = self.H;
        g = g+1.0
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
def lg_kernel_sum_sequential(l_kernel_sequential,Data):
    T = Data['T'];
    n = len(T)
    l  = np.zeros(n)
    dl = l_kernel_sequential.dl
    dl = { key:np.zeros(n) for key in dl }

    for i in range(n-1):
        l_kernel_sequential.event()
        l_kernel_sequential.step_forward(T[i+1]-T[i])
        l[i+1] = l_kernel_sequential.l
        dl_i = l_kernel_sequential.dl
        for key in dl_i:
            dl[key][i+1] = dl_i[key]

    return [l,dl]


def t_trans(l_baseline,kernel_int,T,itv):
    [st,en] = itv
    n = len(T)
    T_ext = np.hstack([st,T,en])
    Int_ext = np.zeros(n+1)

    for i in range(n+1):
        Int_ext[i] += (l_baseline(T_ext[i])+l_baseline(T_ext[i+1]))*(T_ext[i+1]-T_ext[i])/2.0

    for i in range(n):
        Int_ext[i+1] += kernel_int(T_ext[i+1]-T[:i+1],T_ext[i+2]-T[:i+1]).sum()

    Int_ext_cumsum = Int_ext.cumsum()

    return [Int_ext_cumsum[:n],[0,Int_ext_cumsum[-1]]]

def tl_kernel(l_kernel_sequential,T,itv):

    [st,en] = itv
    t = np.hstack([ np.linspace(t[0],t[1],30) for t in np.vstack([np.hstack([st,T]),np.hstack([T,en])]).transpose() ])
    mark = np.zeros_like(t,dtype='i8')
    mark[29:-30:30] = 1
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
def plot_N(T,itv):

    gs = gridspec.GridSpec(100,1)

    plt.figure(figsize=(4,5), dpi=100)
    mpl.rc('font', size=12, family='Arial')
    mpl.rc('axes',titlesize=12)
    mpl.rc('pdf',fonttype=42)

    [st,en] = itv
    n = len(T)
    x = np.hstack([st,np.repeat(T,2),en])
    y = np.repeat(np.arange(n+1),2)

    plt.subplot(gs[0:10,0])
    plt.plot(np.hstack([ [t,t,np.NaN] for t in T]),np.array( [0,1,np.NaN] * n ),'k-',linewidth=0.5)
    plt.xticks([])
    plt.xlim(itv)
    plt.ylim([0,1])
    plt.yticks([])
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)

    plt.subplot(gs[15:100,0])
    plt.plot(x,y,'k-',clip_on=False)
    plt.xlim(itv)
    plt.ylim([0,n])
    plt.xlabel('time')
    plt.ylabel(r'$N(0,t)$')
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

def plot_l(T,x,l,l_baseline):

    gs = gridspec.GridSpec(100,1)

    plt.figure(figsize=(4,5), dpi=100)
    mpl.rc('font', size=12, family='Arial')
    mpl.rc('axes',titlesize=12)
    mpl.rc('pdf',fonttype=42)

    l_max = l.max()
    n = len(T)

    plt.subplot(gs[0:10,0])
    plt.plot(np.hstack([ [t,t,np.NaN] for t in T]),np.array( [0,1,np.NaN] * n ),'k-',linewidth=0.5)
    plt.xticks([])
    plt.xlim([x[0],x[-1]])
    plt.ylim([0,1])
    plt.yticks([])
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)

    plt.subplot(gs[15:100,0])
    plt.plot(x,l,'k-',lw=1)
    plt.plot(x,l_baseline,'k:',lw=1)
    plt.xlim([x[0],x[-1]])
    plt.ylim([0,l_max])
    plt.xlabel('time')
    plt.ylabel(r'$\lambda(t|H_t)$')
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

def plot_N_pred(T,T_pred,itv,en_f):

    gs = gridspec.GridSpec(100,1)

    plt.figure(figsize=(4,5), dpi=100)
    mpl.rc('font', size=12, family='Arial')
    mpl.rc('axes',titlesize=12)
    mpl.rc('pdf',fonttype=42)

    [st,en] = itv
    n = len(T)
    x = np.hstack([st,np.repeat(T,2),en])
    y = np.repeat(np.arange(n+1),2)
    n_pred_max = np.max([ len(T_i) for T_i in T_pred ])

    plt.subplot(gs[0:10,0])
    plt.plot(np.hstack([ [t,t,np.NaN] for t in T]),np.array( [0,1,np.NaN] * n ),'k-',linewidth=0.5)
    plt.xticks([])
    plt.xlim([itv[0],en_f])
    plt.ylim([0,1])
    plt.yticks([])
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)

    plt.subplot(gs[15:,0])
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

def plot_KS(T_trans,itv_trans):
    from scipy.stats import kstest

    plt.figure(figsize=(4,4), dpi=100)
    mpl.rc('font', size=12, family='Arial')
    mpl.rc('axes',titlesize=12)
    mpl.rc('pdf',fonttype=42)

    n = len(T_trans)
    [st,en] = itv_trans
    x = np.hstack([st,np.repeat(T_trans,2),en])
    y = np.repeat(np.arange(n+1),2)/n
    w = 1.36/np.sqrt(n)
    [_,pvalue] = kstest(T_trans/itv_trans[1],'uniform')

    plt.plot(x,y,"k-",label='Data')
    plt.fill_between([0,n*w,n*(1-w),n],[0,0,1-2*w,1-w],[w,2*w,1,1],color="#dddddd",label='95% interval')
    plt.xlim([0,n])
    plt.ylim([0,1])
    plt.ylabel("cumulative distribution function")
    plt.xlabel("transfunced time")
    plt.title("p-value = %.3f" % pvalue)
    plt.legend(loc="upper left")
