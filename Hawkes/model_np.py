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

from .tools import Quasi_Newton,indexed_ndarray



class estimator_np:

    def __init__(self,support,num_bin):

        self.type = 'nonpara'
        self.support = support
        self.num_bin = num_bin
        self.para_list = [ ('lambda',i) for i in range(num_bin) ]

    def fit(self,T,itv,prior=[],opt=[],merge=[]):

        self.itv = itv
        T = np.array(T); T = T[(itv[0]<T)&(T<itv[1])].copy();
        self.Data = {'T':T}

        support = self.support
        num_bin = self.num_bin

        list =      ['mu','lambda']
        length =    {'mu':1,'lambda':num_bin }
        exp =       {'mu':True,'lambda':True }
        ini =       {'mu':0.5*len(T)/(itv[1]-itv[0]),'lambda':np.ones(num_bin)*0.5/support  }
        step_Q =    {'mu':0.2,'lambda':np.ones(num_bin)*0.2  }
        step_diff = {'mu':0.01,'lambda':np.ones(num_bin)*0.01 }
        self.stg = {"para_list":list,'para_length':length,'para_exp':exp,'para_ini':ini,'para_step_Q':step_Q,'para_step_diff':step_diff}

        self.set_data(self.Data,self.itv)

        [para,L,ste,G_norm,i_loop] = Quasi_Newton(self,prior,merge,opt)

        self.para = para
        self.parameter = para.to_dict()
        self.L = L
        self.AIC = -2.0*(L-len(para))
        self.br = 0
        self.ste = ste
        self.i_loop = i_loop

        return self

    def set_data(self,Data,itv):

        self.T = Data['T']
        self.itv = itv

        T = Data['T']
        n = len(T)
        st,en = itv
        support = self.support
        num_bin = self.num_bin

        bins = np.linspace(0,support,num_bin+1)
        size_bin = support/num_bin

        #dTs = [ T[i]-T[:i] for i in range(n) ]
        #dTs = [ dT[dT<support] for dT in dTs ]
        j = 0
        dTs = []
        for i in range(n):
            while T[j] <= T[i]-support:
                j += 1
            dTs.append(T[i]-T[j:i])

        dTs_index = [ np.digitize(dT,bins,right=False) - 1 for dT in dTs ]

        #T_st = st - T; T_st[T_st<0] = 0
        T_en = en - T
        #T_st_index = np.digitize(T_st,np.hstack([bins,np.inf]),right=False) - 1
        T_en_index = np.digitize(T_en,np.hstack([bins,np.inf]),right=False) - 1
        #T_st_from_left_edge = T_st - T_st_index*size_bin
        T_en_from_left_edge = T_en - T_en_index*size_bin

        index_cum = np.split(np.tri(num_bin+1).flatten(),num_bin+1)[:-1]
        index_cum.insert(0,np.zeros(num_bin+1))
        index_p = np.split(np.identity(num_bin+1).flatten(),num_bin+1)

        dl = np.vstack([ np.bincount(index,minlength=num_bin).astype('f8') for index in dTs_index ]).transpose()
        dl = { key:x for key,x in zip(self.para_list,dl) }

        dInt = np.vstack([ ( index_cum[index]*size_bin + index_p[index]*t )[:-1] for index,t in zip(T_en_index,T_en_from_left_edge) ]).sum(axis=0)
        dInt = { key:x for key,x in zip(self.para_list,dInt) }

        self.n = n
        self.bins = bins
        self.size_bin = size_bin
        self.dTs = dTs
        self.dTs_index = dTs_index
        #self.T_st_index = T_st_index
        self.T_en_index = T_en_index
        #self.T_st_from_left_edge = T_st_from_left_edge
        self.T_en_from_left_edge = T_en_from_left_edge

        self.dl = dl
        self.dInt = dInt

        #### for L_itv
        j = 0
        dTs1_itv = [np.array([])]
        dTs2_itv = [np.array([])]
        for i in range(1,n):
            while T[j] <= T[i-1]-support:
                j += 1
            dTs1_itv.append(T[i-1]-T[j:i])
            dTs2_itv.append(T[i]  -T[j:i])

        dTs1_index = [ np.digitize(dT,bins,right=False) - 1 for dT in dTs1_itv ]
        dTs2_index = [ np.digitize(dT,bins,right=False) - 1 for dT in dTs2_itv ]
        dTs1_from_left_edge = [ dT-index*size_bin for dT,index in zip(dTs1_itv,dTs1_index) ]
        dTs2_from_left_edge = [ dT-index*size_bin for dT,index in zip(dTs2_itv,dTs2_index) ]

        self.dTs1_index = dTs1_index
        self.dTs2_index = dTs2_index
        self.dTs1_from_left_edge = dTs1_from_left_edge
        self.dTs2_from_left_edge = dTs2_from_left_edge

        return self

    def LG(self,para,only_L=False):
        self.para = para
        mu = para['mu']
        itv = self.itv
        n = self.n

        l_bin = np.hstack([self.para['lambda'],0])
        l_bin_int = np.hstack([0,(l_bin[:-1].cumsum() * self.size_bin )])

        ###kernel
        l = np.array([ l_bin[index].sum() for index in self.dTs_index ])
        dl = self.dl

        Int = np.sum(  l_bin_int[self.T_en_index] + l_bin[self.T_en_index]*(self.T_en_from_left_edge) )
        dInt = self.dInt
        dInt = { key: dInt[key] - 1.0/( np.log(self.para[key]) - np.log(1e-5) )/self.para[key]  for key in dInt }

        ###baseline
        l = l + mu
        dl['mu'] = np.ones(n)
        Int = Int + mu*(itv[1]-itv[0])
        dInt['mu'] = (itv[1]-itv[0])

        dl   = para.inherit().initialize_values(n).insert_values_from_dict(dl).values
        dInt = para.inherit().initialize_values(1).insert_values_from_dict(dInt).values

        L = np.sum(np.log(l)) - Int
        G = (dl/l).sum(axis=1) - dInt
        G = para.inherit().set_values(G)

        return [L,G]

    def L_index(self):
        mu = self.para['mu']
        T = self.T

        l_bin = np.hstack([self.para['lambda'],0])
        l_bin_int = np.hstack([0,(l_bin[:-1].cumsum() * self.size_bin )])
        l = np.array([ l_bin[index].sum() + mu for index in self.dTs_index ])
        Int1 = [ np.sum(  l_bin_int[index] + l_bin[index]*dT ) for (dT,index) in zip(self.dTs1_from_left_edge,self.dTs1_index) ]
        Int2 = [ np.sum(  l_bin_int[index] + l_bin[index]*dT ) for (dT,index) in zip(self.dTs2_from_left_edge,self.dTs2_index) ]
        Int = np.array(Int2) - np.array(Int1) + mu*(T-np.hstack([0,T[:-1]]))
        return np.log(l) - Int

    def plot_kernel(self):
        bins = np.linspace(0,self.support,self.num_bin+1)
        x = np.vstack([bins[:-1],bins[1:]]).transpose().flatten()
        y = np.repeat(self.para['lambda'],2)
        plt.plot(x,y,'k-')

    def set_para(self,para):
        self.para = para
        return self

class estimator_np_MultiSeq():

    def __init__(self,support,num_bin):
        self.type = 'nonpara'
        self.support = support
        self.num_bin = num_bin
        self.para_list = [ ('lambda',i) for i in range(num_bin) ]

    def fit(self,Data,itv,num_seq,prior=[],opt=[],merge=[]):

        self.num_seq = num_seq
        self.itv = itv
        self.Data = [ Data[i][ (itv[i][0]<Data[i]) & (Data[i]<itv[i][1]) ].copy() for i in range(num_seq) ]

        support = self.support
        num_bin = self.num_bin

        n_total = np.sum([ len(self.Data[i])   for i in range(num_seq) ])
        l_total = np.sum([ itv[i][1]-itv[i][0] for i in range(num_seq) ])
        mu_init = 0.5*n_total/l_total

        list =      ['mu','lambda']
        length =    {'mu':1,'lambda':num_bin }
        exp =       {'mu':True,'lambda':True }
        ini =       {'mu':mu_init,'lambda':np.ones(num_bin)*0.5/support  }
        step_Q =    {'mu':0.2,'lambda':np.ones(num_bin)*0.2  }
        step_diff = {'mu':0.01,'lambda':np.ones(num_bin)*0.01 }
        self.stg = {"para_list":list,'para_length':length,'para_exp':exp,'para_ini':ini,'para_step_Q':step_Q,'para_step_diff':step_diff}

        self.estimators = []
        for i in range(num_seq):
            estimator_tmp = estimator_np(support,num_bin).set_data({'T':self.Data[i]},self.itv[i])
            self.estimators.append(estimator_tmp)

        [para,L,ste,G_norm,i_loop] = Quasi_Newton(self,prior,merge,opt)

        self.para = para
        self.parameter = para.to_dict()
        self.L = L
        self.AIC = -2.0*(L-len(para))
        self.ste = ste
        self.i_loop = i_loop

        model = estimator_np(self.support,self.num_bin).set_para(para)
        model.L = L

        return model

    def LG(self,para,only_L=False):

        self.para = para

        for i in range(self.num_seq):
            [L_tmp,G_tmp]=self.estimators[i].LG(para)

            if i==0:
                L = L_tmp
                G = G_tmp
            else:
                L = L + L_tmp
                G.values = G.values + G_tmp.values

        return [L,G]
