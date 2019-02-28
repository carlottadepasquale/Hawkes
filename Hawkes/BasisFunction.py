from __future__ import division
from __future__ import print_function

import sys,time,datetime,copy,subprocess,itertools,pickle,warnings,numbers

import numpy as np
import scipy as sp
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib as mpl

from scipy.spatial import Voronoi

###################################################################################### core class
class BasisFunctionExpansion_1D:

    def __init__(self,itv,num_basis):
        self.itv = itv
        self.num_basis = num_basis

    def set_coef(self,coef):
        self.coef = coef
        return self

    def Matrix_BasisFunction(self,x):
        pass

    def set_x(self,x):
        [st,en] = self.itv
        self.M_BF = self.Matrix_BasisFunction(x)
        bin_edge = np.hstack([st,(x[:-1]+x[1:])/2,en])
        self.weight = bin_edge[1:] - bin_edge[:-1]
        return self

    def get_y(self):
        pass

    def get_dy(self):
        pass

    def get_y_at(self,x):
        pass

    def get_int(self):
        weight = self.weight
        y = self.get_y()
        Int = weight.dot(y)
        return Int

    def get_dint(self):
        weight = self.weight
        dy = self.get_dy()
        dInt = weight.dot(dy)
        return dInt

###################################################################################### Base class
class linear_1D(BasisFunctionExpansion_1D):

    def get_y(self):
        M_BF = self.M_BF; coef = self.coef
        y = M_BF.dot(coef)
        return y

    def get_dy(self):
        M_BF = self.M_BF;
        return M_BF

    def get_y_at(self,x):
        coef = self.coef
        M_BF = self.Matrix_BasisFunction(x)
        return M_BF.dot(coef)

class loglinear_1D(BasisFunctionExpansion_1D):

    def get_y(self):
        M_BF = self.M_BF; coef = self.coef
        y = np.exp( M_BF.dot(coef) )
        self.y = y
        return y

    def get_dy(self):
        M_BF = self.M_BF; y = self.y
        dy = y.reshape(-1,1) * M_BF
        return dy

    def get_y_at(self,x):
        coef = self.coef
        M_BF = self.Matrix_BasisFunction(x)
        return np.exp( M_BF.dot(coef) )

###################################################################################### cosine bump function
class loglinear_COS(loglinear_1D):

    def bump(self,x):
        y = np.zeros_like(x)
        index = (-2<x) & (x<2)
        y[index] = ( np.cos( np.pi*x[index]/2 ) + 1 )/4
        return y

    def Matrix_BasisFunction(self,x):
        [st,en] = self.itv; m = self.num_basis;
        w = (en-st)/(m-3)
        return np.vstack([ self.bump( (x-st-(i-1)*w)/w ) for i in range(m) ]).transpose()

###################################################################################### piecewise linear function
class plinear(linear_1D):

    def bump(self,x):
        y = np.interp(x,[-1.0,0.0,1.0],[0.0,1.0,0.0])
        return y

    def Matrix_BasisFunction(self,x):
        [st,en] = self.itv; m = self.num_basis;
        w = (en-st)/(m-1)
        return np.vstack([ self.bump( (x-st-i*w)/w ) for i in range(m) ]).transpose()
