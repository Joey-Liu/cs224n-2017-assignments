#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 26 20:18:58 2017

@author: joey
"""

import numpy as np

def eigenvalue(beta, phi):
    tmp = beta * np.dot(phi.transpose(), phi)
    lamb = np.linalg.eigvals(tmp)
    return lamb

def getSnInv(alpha, beta, phi):
    return alpha * np.eye(phi.shape[1]) + beta * np.dot(phi.transpose(), phi)

def getmn(SnInv, beta, phi, t):
    Sn = np.linalg.inv(SnInv)
    return beta * Sn.dot(phi.transpose()).dot(t)

def getGamma(lamb, alpha):
    denominator = alpha + lamb
    return np.sum(lamb / denominator)

def updateAlpha(gamma, mn):
    return gamma / np.dot(mn.transpose(), mn)

def updateBeta(N, gamma, mn, phi, t):
    sumsquare = np.sum((t.transpose() - mn.transpose().dot(phi.transpose()))**2)
#    sumsquare = 0        
#    for i in range(N):
#        sumsquare += (t[i] - mn.transpose().dot(phi[i]))**2
    tmp = sumsquare / (N - gamma)
    return 1.0 / tmp

def getExpect(mn, phi):
    return mn.transpose().dot(phi)

def getVariance(beta, phi, Sn):
    return 1.0 / beta + phi.transpose().dot(Sn).dot(phi)

if __name__ == "__main__":
    X = np.array([2, 7, 5, 11, 14])
    t = np.array([19, 62, 37, 94, 120])
    
    N = len(X)
    phi = X
    phi = np.reshape(phi, [5, -1])
    t = np.reshape(t, [5, -1])
    
    alpha = 10000000000
    beta = 0.0002
    SnInv = getSnInv(alpha, beta, phi)
    mn = getmn(SnInv, beta, phi, t)
    lamb = eigenvalue(beta, phi)
    gamma = getGamma(lamb, alpha)
    
    cnt = 0
    while True:
        alphaNew = updateAlpha(gamma, mn)
        betaNew = updateBeta(N, gamma, mn, phi, t)
        
        if np.abs(alphaNew - alpha) < 1e-7 and\
            np.abs(betaNew - beta) < 1e-7:
                break
        cnt += 1
        print alphaNew
        print betaNew
        alpha = alphaNew
        beta = betaNew
        SnInv = getSnInv(alpha, beta, phi)
        mn = getmn(SnInv, beta, phi, t)
        lamb = eigenvalue(beta, phi)
        gamma = getGamma(lamb, alpha)
        
    expe = getExpect(mn, 17)
    sn = np.linalg.inv(SnInv)
    vari = getVariance(beta, np.array([17]), sn)
    print "expe: %f  variance: %f" % (expe, vari)