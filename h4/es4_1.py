# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 13:47:19 2017

@author: Diego
"""

import numpy as np
import matplotlib.pyplot as plt

###############################################################################
#1.1 DIRECT SAMPLING WITH 2 PINS
###############################################################################


n = 2 #number of spheres
L = 8 #length of the region available for the spheres
sigma = 0.75 #with of the spheres

#direct sampling of 2 hard sferes 
    #n_samples=number of samples
    #samples1,2 = samples of the coordinates of the first sphere 1,2
    
def sample2(n_samples,L,sigma):
    samples1 = [] 
    samples2 = []
    while(True):
        p1 , p2  =  np.random.uniform(0+sigma,L-sigma), np.random.uniform(0+sigma,L-sigma)
        if(abs(p1-p2) > 2*sigma):
            samples1.append(p1)
            samples2.append(p2)
        if len(samples1) >= n_samples:
            break
    return samples1, samples2

x,y = sample2(10**5, L, sigma)

#historgram for the distributions of the coordinates of pin 1 and 2
plt.figure(1)
plt.hist(x, bins = 10**2)
plt.title('first pin')
plt.figure(2)
plt.hist(y, bins = 10**2)
plt.title('second pin')
plt.show()



###############################################################################
#1.2 WRONG DIRECT SAMPLING WITH TWO PINS
###############################################################################

'''
exchange this function with sample2, the meaning of the variables is the same
'''

def sample2wrong(n_samples,L,sigma):
    samples1 = [] 
    samples2 = []
    while(True):
        p1  =  np.random.uniform(0+sigma,L-sigma)
        samples1.append(p1)
        while(True):
            p2  =  np.random.uniform(0+sigma,L-sigma)
            if(abs(p1-p2) > 2*sigma):
                samples2.append(p2)
                break
        
        if len(samples1) >= n_samples:
            break
    return samples1, samples2




###############################################################################
#1.3 SAMPLE N PINS
###############################################################################