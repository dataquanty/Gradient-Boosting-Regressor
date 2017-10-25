#cython: boundscheck=False, wraparound=False, nonecheck=False
# -*- coding: utf-8 -*-
"""
Created Oct 2017

@author: dataquanty
"""

cimport numpy as np
cimport cython
from cython.parallel cimport prange, parallel
from libc.math cimport log2 
from libc.stdlib cimport malloc, free




@cython.cdivision(True) 
cdef double getError(short[:] x,double[:] y, short splitx) nogil:
    cdef int ylen = y.shape[0]
    cdef int i
    cdef double med1 = 0. , med2 = 0.
    cdef int n_sub1 = 0 , n_sub2 = 0
    
    #First loop calc mean 
    for i in range(ylen):
        if x[i] > splitx:
            med1 += y[i]
            n_sub1 +=1
        else: 
            med2 += y[i]
            n_sub2 +=1
    
    med1 = med1 /<double>n_sub1
    med2 = med2 /<double>n_sub2
    
    #second loop calc error
    cdef double err1 = 0. , err2 = 0.
    
    for i in range(ylen):
        if x[i] > splitx:
            err1+=(y[i]-med1)**2
        else: 
            err2+=(y[i]-med2)**2

    cdef double weighted = (err1+err2)/ylen
    return weighted



    
cdef double getEntrGain(short[:] x, short[:] y, short splitx, short nclasses) nogil:
    #First loop calc classes
    cdef int i
    cdef int ylen = y.shape[0]
    cdef int n1 = 0, n2 = 0
    cdef double *px1 = <double *>malloc(nclasses * sizeof(double))
    cdef double *px2 = <double *>malloc(nclasses * sizeof(double))
    
    for i in range(nclasses):
        px1[i]=0.
        px2[i]=0.
    
      
    for i in range(ylen):
        if x[i] == splitx:
            n1+=1
            px1[y[i]]+=1.
            
        else: 
            n2+=1
            px2[y[i]]+=1.
    
    if n1==ylen or n2==ylen:
        return 1.
    
    cdef double entr1 = 0., entr2 = 0. , px1_tmp, px2_tmp
    
    for i in range(nclasses):
        px1_tmp = px1[i]/<double>n1
        px2_tmp = px2[i]/<double>n2
        if px1_tmp > 0. :
            entr1 = entr1-px1_tmp*log2(px1_tmp)/log2(nclasses)
        if px2_tmp > 0. :
            entr2 = entr2-px2_tmp*log2(px2_tmp)/log2(nclasses)
        
    
    cdef double weighted = (<double>n1*entr1+<double>n2*entr2)/<double>(ylen)
    free(px1)
    free(px2)

    return weighted
    
    