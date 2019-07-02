from __future__ import division
from __future__ import print_function

import sys,time,datetime,copy,subprocess,itertools,pickle,warnings,numbers

import numpy as np
import pandas as pd

#################################
## INDEXED_NDARRAY
#################################
class indexed_ndarray():

    ### hash
    def set_hash(self,para_list,para_length):

        self.para_list = para_list
        self.para_length = para_length
        self.length = np.array(para_length).sum()
        self.hash = {}
        self.hash_array = {}
        self.index_tuple = []
        index_ini = 0

        for para,length in zip(para_list,para_length):
            self.hash.update({ para: slice(index_ini,index_ini+length) if length>1 else index_ini })
            self.hash.update({ (para,i): index_ini+i for i in range(length) })
            self.hash_array.update({ para: np.arange(index_ini,index_ini+length,dtype="i8") if length>1 else index_ini })
            self.hash_array.update({ (para,i): index_ini+i for i in range(length) })
            self.index_tuple.extend([ (para,i) for i in range(length) ])
            index_ini += length

        return self

    def create_from_dict(self,dic):
        para_list = dic.keys()
        para_length = [ 1 if isinstance(x,numbers.Number) else len(x) for x in dic.values() ]
        values = np.hstack(list(dic.values()))
        self.set_hash(para_list,para_length).set_values(values)
        return self

    def indexing(self,key_list):
        try:
            index = np.hstack([ self.hash_array[key] for key in key_list ])
        except:
            index = slice(0)
        return index

    def add_hash(self,key_list,key_name):
        self.hash.update({ key_name: self.indexing(key_list) })
        return self

    ### values
    def set_values(self,ndarray):
        self.values = ndarray
        return self

    def initialize_values(self,n):
        if n == 1:
            self.values = np.zeros(self.length)
        else:
            self.values = np.zeros([self.length,n])
        return self

    def set_values_from_dict(self,dic):
        self.values = np.hstack(  [ dic[para] for para in self.para_list ] )
        return self

    def insert_values_from_dict(self,dic):
        for key in dic:
            self.values[self.hash[key]] = dic[key]
        return self

    #### inherit
    def inherit(self):
        cls = indexed_ndarray()
        cls.para_list = self.para_list
        cls.para_length = self.para_length
        cls.length = self.length
        cls.hash = self.hash
        cls.hash_array = self.hash_array
        cls.index_tuple = self.index_tuple
        return cls

    def copy(self):
        cls = self.inherit()
        cls.values = self.values.copy()
        return cls

    ### output
    def to_pd(self):
        if self.values.ndim == 1:
            return pd.Series(self.values,index=pd.MultiIndex.from_tuples(self.index_tuple))
        else:
            return pd.DataFrame(self.values,index=pd.MultiIndex.from_tuples(self.index_tuple))

    def to_dict(self):
        return { para: self.values[self.hash[para]] for para in self.para_list }

    ### basic
    def __getitem__(self,key):
        try:
            return self.values[self.hash[key]]
        except:
            return self.values[self.indexing(key)]

    def __setitem__(self,key,value):
        try:
            self.values[self.hash[key]] = value
        except:
            self.values[self.indexing(key)] = value

    def __len__(self):
        return self.length

    def __str__(self):
        return self.to_pd().__str__()
