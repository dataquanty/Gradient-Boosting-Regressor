#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Oct 2017

@author: dataquanty
"""

from GBRegTree import GBRegTree as GBM
import numpy as np

from sklearn.datasets import load_diabetes
from sklearn.utils import shuffle
from sklearn.ensemble import GradientBoostingRegressor
from timeit import timeit

def fun():
    data = load_diabetes()
    
    X = data['data']
    Y = data['target']
    
    
    for k in range(X.shape[1]):
        bins = np.arange(np.min(X[:,k]),np.max(X[:,k]),(np.max(X[:,k])-np.min(X[:,k]))/128)
        X[:,k]=np.digitize(X[:,k],bins)
    
    X = X.astype(np.int)
    preds = np.array([0]*len(X))
    
    X, Y, preds = shuffle(X,Y,preds)
    
    offset = int(X.shape[0] * 0.8)
    X_train, y_train = X[:offset], Y[:offset]
    X_test, y_test = X[offset:], Y[offset:]
    
    
    
    
    
    gbm = GBM(niteration=5000,alpha=0.1,max_depth=2,min_leaf=2,nlevels=128)
    """
    gbm = GBM(niteration=10000,alpha=0.1,max_depth=2,min_leaf=2)
    
    """

    gbm.fit(X_train,y_train)
    ypred = gbm.predict(X_train)
    
    print np.sum(np.square(y_train-ypred))/len(y_train)
    print np.sum(np.square(y_test-gbm.predict(X_test)))/len(y_test)
    #print gbm.losses
    
    
    preds = np.vstack((preds.T,gbm.predict(X))).T
    
print timeit(fun,number=1)

"""
gbm = GradientBoostingRegressor(loss='ls',learning_rate=0.05,n_estimators=40,max_depth=5)
gbm.fit(X_train,y_train)
ypred = gbm.predict(X_train)

print np.sum(np.square(y_train-ypred))/len(y_train)
print np.sum(np.square(y_test-gbm.predict(X_test)))/len(y_test)
"""

