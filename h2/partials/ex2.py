# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt


#######################################
# 2.2                                 #
#######################################

def sample_cauchy(alpha, beta, N):
    return np.random.standard_cauchy(N) * beta + alpha

beta = 10.0
alpha = 30.0
N = 10


#######################################
# 2.3                                 #
#######################################

def pdf_cauchy(x, alpha, beta):
    return beta / ( np.pi * ( beta**2 + (x - alpha)**2 ) )

def likelihood(sample, alpha, beta):
    return np.sum(np.log(pdf_cauchy(sample, alpha, beta)))

beta = 10.0
Ns = [10, 100, 1000]

alphas = np.arange(10, 50, 0.1)
samples = [sample_cauchy(alpha, beta, N) for N in Ns]

for s in samples:
    lh = np.array([likelihood(s, alpha, beta) for alpha in alphas]) / len(s)
    index_max = np.argmax(lh)

    plt.plot(alphas, lh)  # plot a ‘normalized’ likelihood
    plt.scatter(alphas[index_max], lh[index_max], marker='x')

plt.show()
# @todo: better plot

#######################################
# 2.4                                 #
#######################################

for s in samples:
    print(np.average(s))
