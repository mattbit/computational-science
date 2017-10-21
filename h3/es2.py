import numpy as np
import matplotlib.pyplot as plt

def sample_rect(N):
    """Returns an array of random points [x, y] inside the square."""
    x= np.random.uniform(0,1,N)
    y= np.random.uniform(0,1-np.exp(-1),N)
    return [x,y]

def sample_tri(N):
    """Returns an array of random points [x, y] uniformly 
    distributed inside the triangle y(x)=x, for x(0,1)."""
    x= (np.random.uniform(0,1,N))^0.5
    y= np.random.uniform(0,1,N)
    return [x,y]


# Visualize the sampling
x,y= sample_rect(100)
plt.plot(x,y, 'ro')
plt.show()

#approximate value of the integral#
def estimate(sample):
    j=0
    for i in range(np.size(sample,1)):
        if sample[1][i]<1-np.exp(-sample[0][i]):
            j+=1
    return (1.*j/(1.*np.size(sample,1)))
    

#statistics

# Sample sizes
Ns = [10, 100, 1000, 10000]

# Number of samples (of length N each) we average on
repeat = 1000

means = np.empty(len(Ns))      # empirical mean of N estimates of π
stds = np.empty(len(Ns))  # empirical variance over N estimates
error = np.empty(len(Ns))    # empirical probability of getting error >= MAX_ERROR

for i, N in enumerate(Ns):
    integral_estimates = np.array([(1-np.exp(-1))*estimate(sample_rect(N)) for _ in range(repeat)])
    stds[i] = (np.var(integral_estimates, ddof=1))**0.5,  # use the unbiased estimator over N-1
    means[i] = np.average(integral_estimates)
    error[i] = np.average(np.abs(integral_estimates - np.exp(-1)))

#!!!!!perchè abbiamo messo max error?????####    


    
