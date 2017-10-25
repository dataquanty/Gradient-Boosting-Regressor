#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Oct 2017

@author: dataquanty
"""


from RegTree import RegTree
import numpy as np
cimport numpy as np
cimport cython
np.import_array() 

DTYPEi = np.int16
DTYPEf = np.float64
ctypedef np.int16_t int16
ctypedef np.float64_t float64

class GBRegTree:
    
    def __init__(self,niteration=5,alpha=0.1,max_depth=4,min_leaf=2,nlevels=128):
        self.niteration=niteration
        self.alpha = alpha
        self.max_depth = max_depth
        self.min_leaf = min_leaf
        self.models = []
        self.losses = []
        self.initval = 0.
        self.nlevels = nlevels
        

    def fit(self,X,y):
        cdef np.ndarray[int16,ndim=2] Xs = np.array(X,dtype=DTYPEi)
        cdef np.ndarray[float64,ndim=1] ys = np.array(y,dtype=DTYPEf)
        cdef float med = np.mean(ys)
        cdef float initval = self.initval
        cdef np.ndarray[float64,ndim=1] r1 = np.zeros(ys.shape[0],dtype=DTYPEf)
        cdef np.ndarray[float64,ndim=1] preds = np.zeros(ys.shape[0],dtype=DTYPEf)
        cdef int i
        cdef float alpha = self.alpha
        r1 = ys-np.ones(ys.shape[0],dtype=DTYPEf)*initval
        
        self.models = []
        for i in xrange(self.niteration):
            model = RegTree(min_leaf=self.min_leaf,max_depth=self.max_depth,reg=True,nlevels=self.nlevels)
            model.fit(Xs,r1)
            preds = model.predict(Xs)
            r1 = r1 - alpha*preds
            self.models.append(model)
            self.losses.append(np.mean(np.abs(r1))/ys.shape[0])
    


    def predict(self,X):
        preds = np.array([self.initval]*len(X))
        
        for m in self.models[::-1]:
            preds = m.predict(X)*self.alpha+preds
        
        return preds

