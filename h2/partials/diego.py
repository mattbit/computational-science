# -*- coding: utf-8 -*-
"""
Created on Sat Oct 07 11:12:39 2017

@author: Diego
"""

import numpy as np
import matplotlib.pyplot as plt
beta=10.0
alfa=30.0

data=[10, 20, 50, 100, 200, 500, 1000]
mean=[]
std=[]
average=[]

for i in range(len(data)):
    xmax=[]
    lkldmax=[]
    n_trials=30
    ave=0
    
    for j in range(n_trials):
        s = np.random.standard_cauchy(data[i])
        for k in range (s.size):
            s[k]=beta*s[k]+alfa
            
        ave+=(np.sum(s)/(1.*s.size))/n_trials
        
        dx=0.01
        xdown=10
        xup=50
        x = np.arange(xdown, xup, dx)
        lkld=0
        for k in range (s.size): 
            lkld+=(-np.log(np.pi*10*(1+((s[k]-x)/beta)**2)))/(s.size*1.)
        
        indices = np.where(lkld == lkld.max())
        xmax.append(xdown+dx*indices[0][0])
    
    lkldmax=0
    for k in range (s.size):
        lkldmax+=(-np.log(np.pi*10*(1+((s[k]-xmax[n_trials-1])/beta)**2)))/(s.size*1.)
    
    average.append(ave)    
    mean.append(np.sum(xmax)/(1.*len(xmax)))
    std.append((np.sum((xmax-mean[i])**2)/(1.*len(xmax)-1))**0.5)
    
    
    plt.figure(1)
    plt.plot(x, lkld, label=data[i])
    plt.plot(xmax[n_trials-1],lkldmax, 'ro')
    plt.legend()

plt.figure(2)
plt.errorbar(np.log(data), mean, std, linestyle='None', marker='^')
plt.axhline(y=30, color='r', linestyle='-')

plt.show()