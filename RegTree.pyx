
# -*- coding: utf-8 -*-
"""
Created Oct 2017

@author: dataquanty
"""

import numpy as np
cimport numpy as np
cimport cython
cimport RegTreeError
from RegTreeError cimport getError, getEntrGain
from cython.parallel import prange
from libc.stdlib cimport malloc, free
from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free

from libc.math cimport isnan
np.import_array() 

DTYPEi = np.int16
DTYPEf = np.float64
ctypedef np.int16_t int16
ctypedef np.float64_t float64

#cdef int freq[NLEVELS];

# "ctypedef" assigns a corresponding compile-time type to DTYPE_t. For
# every type in the numpy module there's a corresponding compile-time
# type with a _t-suffix.




def majorityVote(vector):
    (values,counts) = np.unique(vector,return_counts=True)
    ind=np.argmax(counts)
    return values[ind]

@cython.boundscheck(False)
@cython.wraparound(False)  
cpdef arrayGetUnique(int[:] x):
    cdef int n = x.shape[0]
    cdef int i,j, count
    cdef int[:] resVal = np.zeros(n,dtype=np.dtype("i"))-1
    cdef int[:] resFreq = np.zeros(n,dtype=np.dtype("i"))-1
    with nogil:
        for i in range(n):
            count=1
            if resFreq[i]!=0:
                resFreq[i]=count
                for j in range(i+1,n):
                    if x[i]==x[j]:
                        resFreq[j]=0
    
    return resFreq



class RegTree:
        
    def __init__(self,min_leaf=2,max_depth=12,reg=False,nlevels=128):
        
        self.size = 0
        self.ncols = 0
        self.nclasses = 0
        self.classes_ = 0
        self.min_leaf = min_leaf
        self.max_depth = max_depth
        self.tree = []
        self.reg = reg
        self.nlevels=nlevels
    
        
    
    @cython.wraparound(False)  
    @cython.boundscheck(False)
    @cython.nonecheck(False)
    def getBestSplitReg(self, short[:,:] X, double[:] y):
        
               
        cdef short ncols = X.shape[1]
        cdef int nlines = X.shape[0]
        cdef short nlevels = self.nlevels
        cdef float minerr = 10000000
        cdef float err
        cdef short resCol, resAttr,c,resAttr_tmp
        cdef int i
        
        #cdef short *freq = <short *>malloc(nlevels * sizeof(short))
        cdef int *freq = <int *>PyMem_Malloc(nlevels * sizeof(int))
        #cdef short[:] x = np.zeros(nlines,dtype=DTYPEi)
        
        
        for c in range(ncols):
            #x = X[:,c]
                        
            for i in range(nlevels):
                freq[i]=0
            for i in range(nlines):
                resAttr_tmp = X[i,c]
                if freq[resAttr_tmp]<1:
                    freq[resAttr_tmp]=1
                    err = getError(X[:,c],y,resAttr_tmp)
                    if err<minerr:
                        minerr = err
                        resCol = c
                        resAttr = resAttr_tmp
                        
                    
        PyMem_Free(freq)
        
        return [minerr,resCol,resAttr]
    
    
    

    def splitReg(self,np.ndarray[int16,ndim=2] X,np.ndarray[float64,ndim=1] y,np.ndarray[int16,ndim=1] rootnode):
        
        cdef short i, k=-1, max_depth = self.max_depth
        
        for i in range(rootnode.shape[0]):
            if rootnode[i]==-1:
                k=i
                break
                
        bestSplit = self.getBestSplitReg(X,y)
        entr = bestSplit[0]
        col = int(bestSplit[1])
        val = int(bestSplit[2])
        
        splitvect = (X[:,col]<=val)
        
        X1, y1 = X[splitvect],y[splitvect]
        X2, y2 = X[(splitvect==False)],y[(splitvect==False)]
        
        #or len(np.unique(X1))<2 or len(np.unique(y1))<2
        rootnode[k]=0
        rootnodepy = [x for x in list(rootnode) if x>-1]
        if(len(y1)<=self.min_leaf+1 or entr==0. or k> (max_depth-2)):
            #print X1,y1
            if(len(y1)==0):
                leafval = 0
            else:
                leafval = np.mean(y1)
                
            self.tree.append([''.join(map(str, rootnodepy)),col,val,leafval])
        else:
            self.tree.append([''.join(map(str, rootnodepy)),col,val])
            self.splitReg(np.array(X1,dtype=DTYPEi),np.array(y1,dtype=DTYPEf),rootnode.copy())
        
        rootnode[k]=1
        rootnodepy = [x for x in list(rootnode) if x>-1]
        if(len(y2)<=self.min_leaf+1 or entr==0. or k>(max_depth-2)):
            if(len(y2)==0):
                leafval = 0
            else:
                leafval = np.mean(y2)
            self.tree.append([''.join(map(str, rootnodepy)),col,val,leafval])
        else:
            self.tree.append([''.join(map(str, rootnodepy)),col,val])
            self.splitReg(np.array(X2,dtype=DTYPEi),np.array(y2,dtype=DTYPEf),rootnode.copy())
         
    
    
    def fit(self,np.ndarray[int16,ndim=2] X,np.ndarray[float64,ndim=1] y):
        self.tree = []
        self.size = X.shape[0]
        self.ncols = X.shape[1]
        self.nclasses = np.unique(y).shape[0]
        self.classes_ = np.unique(y)
        cdef short nmodels = self.max_depth
        cdef np.ndarray[int16,ndim=1] node = np.ones((nmodels,),dtype=DTYPEi)*(-1)
        
        if self.reg == False:
            pass
        else:
            self.splitReg(X,y,node)
        
    
    def predict(self,X):
        ypred = np.array([0]*len(X))
        
        leaves = [e for e in self.tree if len(e)>3]
        #print self.tree
        for l in leaves:
            res = np.array([1]*len(X))
            for i in range(len(l[0])):
                nodeid = l[0][:i+1]
                node = [e for e in self.tree if e[0]==nodeid][0]
                if self.reg == True:
                    if(int(node[0][-1])==0):
                        res = res * 1*(X[:,node[1]]<=node[2])
                    else:
                        res = res * 1*(X[:,node[1]]>node[2])
                else:
                    if(int(node[0][-1])==0):
                        res = res * 1*(X[:,node[1]]==node[2])
                    else:
                        res = res * 1*(X[:,node[1]]!=node[2])
        
            
            ypred = ypred + res*l[3]
        
        return ypred
        


