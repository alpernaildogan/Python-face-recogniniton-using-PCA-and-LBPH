#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np



def asColumnMatrix(X):
    if len(X) == 0:
        print ('error')
    mat = np.empty((X[0].size, 0), dtype=X[0].dtype)
    for col in X:
        mat = np.hstack((mat, np.asarray(col).reshape(-1, 1)))
    return mat


def pca(aj, c, num_components=0):
    [n, d] = aj.shape
    if num_components <= 0 or num_components > n:
        num_components = n
    aj = aj.astype(float)
##    print 'aj without mean'
##    print aj
##    print '\n--------------'
    mu = np.sum(aj, axis=1) / d

##    print 'mean vector'
##    print mu

##    np.savetxt('alper1.txt',np.array(aj), newline='.\n', delimiter = ',')
##....print mu0
##....print '\n'
##....print mu

##    print '\n'
    for i in range(aj.shape[1]):
        aj[:, i] = aj[:, i] - mu

##    print 'aj'
##    print aj
##    print '\n--------'
    G = np.dot(aj.T, aj)  # Dot product of two arrays.

##........print G
##........print '\n'
####........print G0

    [eigenvalues, eigenvectors] = np.linalg.eigh(G)  # Return the eigenvalues and eigenvectors of a matrix.

##....# sort eigenvectors descending by their eigenvalue

    sort = np.argsort(-eigenvalues)

##....print eigenvectors
##....print '\n------------'
##....print eigenvalues
##....print '----------'

    eigenvalues = eigenvalues[sort]

##....print eigenvalues

    W = eigenvectors[:, sort]
##    print 'sorted eigenvectors'
##    print eigenvectors
##    print '----------'
##    print 'sorted eigenvalues'
##    print eigenvalues

    # select only num_components
##....sifirlari ele

    eigenvalues = eigenvalues[0:num_components].copy()
    W = W[:, 0:num_components].copy()

##....print '\n------------'....
##....print W
##....print '\n------------'
##....print eigenvalues

    tt = np.dot(aj, W.T)
    WW = tt / np.linalg.norm(tt)
    yj = np.dot(WW.T, aj)
##    print '\n------------'
##    print 'WW'
##    print WW
##    print '\n------------'
##    print 'yj'
##    print yj

##....np.savetxt('alper1.txt',np.array( yj), newline='\n', delimiter = ',')

    return [ WW, mu, yj]


def project(WW, test, mu):

##    print WW
##    print '----------------------'
##    print atest
##    print '----------------------'
##    print mu

    atest = test.reshape(1, -1)
    ytest = np.dot(WW.T, (atest - mu).T)
##    print 'atest'
##    print atest
##    print 'ytest'
##    print ytest

##....np.savetxt('alper1.txt',np.array( yj), newline='\n', delimiter = ',')

    return ytest




def predict(
    IDs,
    WW,
    atest,
    yj,
    mu,
    ):

    minDist = np.finfo('float').max

    # Machine limits for floating point types.

##    datanumber = -1
    ytest = project(WW, atest.reshape(1, -1), mu)
    ytest0 =np.empty((yj[0].size, yj[1].size), dtype=yj[0].dtype) 
    for i in range(len(yj)):
        ytest0[i,:] = ytest[i]
##    print 'ytest0'
##    print ytest0
##    print '\n --------------'
    dist0 = ytest0-yj
    id = 0
    for i in range(len(yj)):
        dist = np.sqrt(np.sum(np.power(dist0[:,i],2)))
##        print '\n ----------'
##        print 'data'+ str(i+1)
##        print 'distances'
##        print dist
        if dist <= minDist:
            minDist = dist
            datanumber = i
    id = IDs[datanumber]
##    print '\n-------'
##    print 'data number'
##    print datanumber
##    print '\n'
##    print 'id'
##    print id    
    return id
