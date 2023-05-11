# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 16:20:14 2019

@author: admin
"""
import numpy as np

def art_fun(A,b,k,r):
    m=np.size(A,0)
    n=np.size(A,1)
    X=np.zeros(shape=(n,k))
    x=np.zeros(shape=(n,1))
    nai2=np.sum(np.abs(A*A),1)#矩阵A每一行的元素的平方和
    for iter in range(k):
        for i in range(m):
            if nai2[i]>0:
                arow=np.reshape(A[i, :], [1, n])
                cha=b[i]-np.dot(arow, x)[0, 0]
                c=(cha/nai2[i])*arow.T
                x=x+r*c
        X[:,iter]=np.squeeze(x)
    return x,X





