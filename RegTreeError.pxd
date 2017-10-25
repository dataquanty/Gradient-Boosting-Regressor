#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 15:59:41 2017

@author: dataquanty
"""
import numpy as np
cimport numpy as np

#cdef float getError(np.ndarray[np.int_t,ndim=1] x,np.ndarray[np.float_t,ndim=1] y, int splitx)
cdef double getError(short[:] x,double[:] y, short splitx) nogil

cdef double getEntrGain(short[:] x, short[:] y, short splitx, short nclasses) nogil