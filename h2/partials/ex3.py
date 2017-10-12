# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar


def sample_exp(tau, N=1):
    """Sample from the truncated exponential distribution."""
    return -tau*np.log(np.random.uniform(np.exp(-20/tau), np.exp(-1/tau), N))


def pdf_exp(tau, x):
    """Probability Density Function of the truncated exponential."""
    return np.exp(-x/tau) / (tau*(np.exp(-1/tau) - np.exp(-20/tau)))


def log_likelihood(tau, sample, sign=1.0):
    """Calculate the log-likelihood given a sample and tau."""
    return sign * np.sum(np.log(pdf_exp(tau, sample)))


plt.xkcd()

#######################################
# 3.2                                 #
#######################################

lambda_true = 10.
Ns = [10, 100, 1000]

samples = [sample_exp(lambda_true, N) for N in Ns]
lambdas = np.arange(5, 50, 0.1)

for s in samples:
    lh = [log_likelihood(l, s) for l in lambdas]
    i_max = np.argmax(lh)

    plt.plot(lambdas, lh/np.abs(lh[i_max]), label=len(s))
    plt.scatter(lambdas[i_max], np.sign(lh[i_max]), marker='x')

plt.legend()
plt.title("Maximum Likelihood for different sample sizes")
plt.show()

#######################################
# 3.4                                 #
#######################################

def estimate_lambda(sample, bounds=(1, 100)):
    return minimize_scalar(
        likelihood,
        args=(sample, -1),
        method='bounded',
        bounds=bounds
    )

def MSE(lambda_true, N=100):
    se = []
    for _ in range(20):
        s = sample_exp(lambda_true, N)
        l_ml = lambda_estimator(s)
        se.append((l_ml - lambda_true)**2)

    return np.average(se)

def I(lam, N):
    # @todo
    return N / lam**3 * (np.exp(-1/lam) - 400*np.exp(-20/lam) + 2*((np.exp(-1/lam) - 20*np.exp(-20/lam))/(np.exp(-1/lam) - np.exp(-20/lam)) + lam))


lam = 10.
s = sample_exp(lam, N=1000)

# lambdas_true = np.arange(1, 20, 1.)

# MSEs = [MSE(l_true) for l_true in lambdas_true]

# plt.plot(lambdas_true, MSEs)
# plt.plot(lambdas_true, 1/I(lambdas_true, N), '--')
# plt.show()
