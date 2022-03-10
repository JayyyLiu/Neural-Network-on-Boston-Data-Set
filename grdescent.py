# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 19:03:09 2019

@author: Jerry Xing
"""
import numpy as np
def grdescent(func, w0, stepsize, maxiter, tolerance = 1e-2):
#% function [w]=grdescent(func,w0,stepsize,maxiter,tolerance)
#%
#% INPUT:
#% func function to minimize
#% w0 = initial weight vector 
#% stepsize = initial gradient descent stepsize 
#% tolerance = if norm(gradient)<tolerance, it quits
#%
#% OUTPUTS:
#% 
#% w = final weight vector
#%
    w = w0
    ## << Insert your solution here
    eps = 2.2204e-14
    iter = 0
    prev_loss = float('inf')
    # YOUR CODE HERE
    while iter < maxiter and stepsize >= eps:
        loss, gradient = func(w)
        if np.linalg.norm(gradient) < tolerance:
            break
        #use heuristic choice of stepsize
        if loss < prev_loss and iter>=1:
           stepsize = 1.01*stepsize
           w = w - stepsize*gradient
           prev_loss = loss

        elif loss >= prev_loss:
           stepsize = 0.5*stepsize
           w = w - stepsize*gradient
           prev_loss = loss

        iter += 1
  
    return w