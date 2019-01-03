from __future__ import division
from __future__ import print_function

import sys,time,datetime,copy,subprocess,itertools,pickle,warnings,numbers

import numpy as np
import scipy as sp
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib as mpl

######################################################################################
class loglinear_COS_intensity:

    def __init__(self,itv,m):
        self.itv = itv
        self.m = m

    def set_x(self,x):
        itv = self.itv; m = self.m
        self.M_BF = Matrix_BasisFunction_COS(itv,m,x)
        x_ext = np.hstack([itv[0],x,itv[1]])
        dx_ext = x_ext[1:] - x_ext[:-1]
        self.M_BF_ext = Matrix_BasisFunction_COS(itv,m,x_ext)
        self.weight_int = ( np.hstack([dx_ext,0]) + np.hstack([0,dx_ext]) )/2
        return self

    def set_coef(self,coef):
        self.coef = coef
        return self

    def get_y(self):
        M_BF = self.M_BF; coef = self.coef
        return np.exp( M_BF.dot(coef) )

    def get_y_at(self,x):
        itv = self.itv; m = self.m; coef = self.coef;
        M_BF = Matrix_BasisFunction_COS(itv,m,x)
        return np.exp( M_BF.dot(coef) )

    def get_y_dy(self):
        M_BF = self.M_BF; coef = self.coef
        l = np.exp( M_BF.dot(coef) )
        dl = l.reshape(-1,1) * M_BF
        return [l,dl]

    def get_int_dint(self):
        M_BF_ext = self.M_BF_ext; coef = self.coef; weight_int = self.weight_int;
        l_ext = np.exp(M_BF_ext.dot(coef))
        Int = weight_int.dot( l_ext )
        dInt = weight_int.dot( l_ext.reshape(-1,1) * M_BF_ext )
        return [Int,dInt]

def cos_bump(x):
    y = np.zeros_like(x)
    index = (-2<x) & (x<2)
    y[index] = ( np.cos( np.pi*x[index]/2 ) + 1 )/4
    return y

def Matrix_BasisFunction_COS(itv,m,x):
    [st,en] = itv
    w = (en-st)/(m-3)
    return np.vstack([ cos_bump( (x-st-(i-1)*w)/w ) for i in range(m) ]).transpose()

######################################################################################
class plinear_intensity:

    def __init__(self,itv,m):
        self.itv = itv
        self.m = m

    def set_x(self,x):
        itv = self.itv; m = self.m
        self.M_BF = Matrix_BasisFunction_hat(itv,m,x)
        x_ext = np.hstack([itv[0],x,itv[1]])
        dx_ext = x_ext[1:] - x_ext[:-1]
        self.M_BF_ext = Matrix_BasisFunction_hat(itv,m,x_ext)
        self.weight_int = ( np.hstack([dx_ext,0]) + np.hstack([0,dx_ext]) )/2
        return self

    def set_coef(self,coef):
        self.coef = coef
        return self

    def get_y(self):
        M_BF = self.M_BF; coef = self.coef
        return M_BF.dot(coef)

    def get_y_at(self,x):
        itv = self.itv; m = self.m; coef = self.coef;
        M_BF = Matrix_BasisFunction_hat(itv,m,x)
        return M_BF.dot(coef)

    def get_y_dy(self):
        M_BF = self.M_BF; coef = self.coef
        l = M_BF.dot(coef)
        dl = M_BF
        return [l,dl]

    def get_int_dint(self):
        M_BF_ext = self.M_BF_ext; coef = self.coef; weight_int = self.weight_int;
        l_ext = M_BF_ext.dot(coef)
        Int = weight_int.dot( l_ext )
        dInt = weight_int.dot( M_BF_ext )
        return [Int,dInt]

def hat(x):
    y = np.interp(x,[-1.0,0.0,1.0],[0.0,1.0,0.0])
    return y

def Matrix_BasisFunction_hat(itv,m,x):
    [st,en] = itv
    w = (en-st)/(m-1)
    return np.vstack([ hat( (x-st-i*w)/w ) for i in range(m) ]).transpose()
