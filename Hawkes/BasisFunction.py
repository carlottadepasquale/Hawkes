from __future__ import division
from __future__ import print_function

import sys,time,datetime,copy,subprocess,itertools,pickle,warnings,numbers

import numpy as np
import scipy as sp
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib as mpl

from scipy.spatial import Voronoi

###################################################################################### Base class
class linear_model:

    def set_coef(self,coef):
        self.coef = coef
        return self

    def set_coef(self,coef):
        self.coef = coef
        return self

    def get_l(self):
        M_BF = self.M_BF; coef = self.coef
        return M_BF.dot(coef)

    def get_l_at(self,x):
        coef = self.coef
        M_BF = self.Matrix_BasisFunction(x)
        return M_BF.dot(coef)

    def get_l_dl(self):
        M_BF = self.M_BF; coef = self.coef
        l = M_BF.dot(coef)
        dl = M_BF
        return [l,dl]

    def get_int_dint(self):
        M_BF = self.M_BF; coef = self.coef; weight = self.weight;
        l = M_BF.dot(coef)
        Int = weight.dot( l )
        dInt = weight.dot( M_BF )
        return [Int,dInt]

class loglinear_model:

    def set_coef(self,coef):
        self.coef = coef
        return self

    def get_l(self):
        M_BF = self.M_BF; coef = self.coef
        return np.exp( M_BF.dot(coef) )

    def get_l_at(self,*xy):
        coef = self.coef
        M_BF = self.Matrix_BasisFunction(*xy)
        return np.exp( M_BF.dot(coef) )

    def get_l_dl(self):
        M_BF = self.M_BF; coef = self.coef
        l = np.exp( M_BF.dot(coef) )
        dl = l.reshape(-1,1) * M_BF
        return [l,dl]

    def get_int_dint(self):
        M_BF = self.M_BF; coef = self.coef; weight = self.weight;
        l = np.exp(M_BF.dot(coef))
        Int = weight.dot( l )
        dInt = weight.dot( l.reshape(-1,1) * M_BF )
        return [Int,dInt]

###################################################################################### cosine bump function
class loglinear_COS(loglinear_model):

    def __init__(self,itv,m):
        self.itv = itv
        self.m = m

    def bump(self,x):
        y = np.zeros_like(x)
        index = (-2<x) & (x<2)
        y[index] = ( np.cos( np.pi*x[index]/2 ) + 1 )/4
        return y

    def Matrix_BasisFunction(self,x):
        [st,en] = self.itv; m = self.m;
        w = (en-st)/(m-3)
        return np.vstack([ self.bump( (x-st-(i-1)*w)/w ) for i in range(m) ]).transpose()

    def set_x(self,x):
        [st,en] = self.itv
        self.M_BF = self.Matrix_BasisFunction(x)
        bin_edge = np.hstack([st,(x[:-1]+x[1:])/2,en])
        self.weight = bin_edge[1:] - bin_edge[:-1]
        return self

class loglinear_COS_2d(loglinear_model):

    def __init__(self,itv_x,m_x,itv_y,m_y):
        self.itv_x = itv_x
        self.itv_y = itv_y
        self.m_x = m_x
        self.m_y = m_y

    def bump(self,x,y):
        index = (-2<x) & (x<2) & (-2<y) & (y<2)
        z = np.zeros_like(x)
        z[index] = ( np.cos( np.pi*x[index]/2 ) + 1 ) * ( np.cos( np.pi*y[index]/2 ) + 1 ) / 16
        return z

    def Matrix_BasisFunction(self,x,y):
        [st_x,en_x] = self.itv_x; [st_y,en_y] = self.itv_y; m_x = self.m_x; m_y = self.m_y;
        w_x = (en_x-st_x)/(m_x-3)
        w_y = (en_y-st_y)/(m_y-3)
        WM = []
        for i in range(m_x):
            for j in range(m_y):
                WM.append( self.bump( (x-st_x-(i-1)*w_x)/w_x, (y-st_y-(j-1)*w_y)/w_y ) )

        return np.vstack(WM).transpose()

    def voronoi_area(self,x,y):
        itv_x = self.itv_x; itv_y = self.itv_y;
        n = len(x)

        xy      = np.transpose(np.vstack([x,y]))
        xy_ext1 = np.transpose(np.vstack([2*itv_x[0]-x,y]))
        xy_ext2 = np.transpose(np.vstack([2*itv_x[1]-x,y]))
        xy_ext3 = np.transpose(np.vstack([x,2*itv_y[1]-y]))
        xy_ext4 = np.transpose(np.vstack([x,2*itv_y[0]-y]))
        xy_ext = np.vstack([xy,xy_ext1,xy_ext2,xy_ext3,xy_ext4])

        V = sp.spatial.Voronoi(xy_ext)
        area = np.array( [  sp.spatial.ConvexHull(V.vertices[V.regions[V.point_region[i]]]).volume for i in range(n) ] )

        #plt.figure()
        #sp.spatial.voronoi_plot_2d(V)
        #plt.xlim(itv_x)
        #plt.ylim(itv_y)

        return area

    def set_xy(self,x,y):
        self.M_BF = self.Matrix_BasisFunction(x,y)
        self.weight = self.voronoi_area(x,y)
        return self

###################################################################################### piecewise linear function
class plinear(linear_model):

    def __init__(self,itv,m):
        self.itv = itv
        self.m = m

    def bump(self,x):
        y = np.interp(x,[-1.0,0.0,1.0],[0.0,1.0,0.0])
        return y

    def Matrix_BasisFunction(self,x):
        [st,en] = self.itv; m = self.m;
        w = (en-st)/(m-1)
        return np.vstack([ self.bump( (x-st-i*w)/w ) for i in range(m) ]).transpose()

    def set_x(self,x):
        [st,en] = self.itv
        self.M_BF = self.Matrix_BasisFunction(x)
        bin_edge = np.hstack([st,(x[:-1]+x[1:])/2,en])
        self.weight = bin_edge[1:] - bin_edge[:-1]
        return self
