import numpy as np
import matplotlib.pyplot as plt


"""Returns an array of random points [x, y] inside the square."""
def sample_rect(N):
    x= np.random.uniform(0,1,N)
    y= np.random.uniform(0,1-np.exp(-1),N)
    return [x,y]

x,y= sample_rect(100)
plt.plot(x,y, 'ro')
plt.show()


"""Returns an array of random points [x, y] uniformly 
    distributed inside the triangle with corners (0,0), (1,0), (1,1)"""
def sample_tri(N):
    x= (np.random.uniform(0,1,N))**0.5
    y= np.random.uniform(0,x,N)
    return [x,y]
    
# Visualize the sampling
x,y= sample_tri(100)
plt.plot(x,y, 'ro')
plt.show()


"""Returns an array of random points [x, y] uniformly distributed 
    between the  curve y(x)=x**(1/2) and y=0, for x belonging to [0,1[."""
def sample_g(N):
    x= (np.random.uniform(0,1,N))**(1.*2/(1.*3))
    """up=upper bound for y"""
    up=(1-np.exp(-1))*x**0.5
    y= np.random.uniform(0,up,N)
    return [x,y]

# Visualize the sampling
x,y= sample_g(100)
plt.plot(x,y, 'ro')
plt.show()

#approximate value of the integral (accept/reject method)#
def estimate(sample):
    j=0
    for i in range(np.size(sample,1)):
        '''accept/reject step'''
        if sample[1][i]<1-np.exp(-sample[0][i]):
            j+=1
    return (1.*j/(1.*np.size(sample,1)))
    

#statistics

# Sample sizes, i.e. number of random points thrown to evaluate the integral
Ns = [10, 100, 1000]

# Number of samples (of with N points each) we average on
repeat = 1000

means = np.empty([len(Ns),3])      # empirical mean of N estimates of the integral
stds = np.empty([len(Ns),3])  # empirical variance over N estimates
error = np.empty([len(Ns),3])    # empirical probability of getting error >= MAX


for i, N in enumerate(Ns):
    '''these are 3 (1000,1) arrays containing the estimates 
    of the integrals for each of the 3 sample regions'''
    integral_rect = np.array([(1-np.exp(-1))*estimate(sample_rect(N))  for _ in range(repeat)])
    integral_tri = np.array([0.5*estimate(sample_tri(N))  for _ in range(repeat)])
    integral_g = np.array([1.*2*(1.-np.exp(-1))/3.*estimate(sample_g(N)) for _ in range(repeat)])
    
    
    stds[i] = np.array([np.var(integral_rect, ddof=1)**0.5,     np.var(integral_tri, ddof=1)**0.5,      np.var(integral_g, ddof=1)**0.5]) # use the unbiased estimator over N-1
    means[i] = np.array([np.average(integral_rect),     np.average(integral_tri),       np.average(integral_g)])
    error[i] = np.array([np.average(np.abs(integral_rect-np.exp(-1))),      np.average(np.abs(integral_tri-np.exp(-1))),        np.average(np.abs(integral_g-np.exp(-1)))])

print('means=', means)
print('standard dev', stds)
print('errors', error)
print('the true value is', np.exp(-1))

title=('y=1-exp(-1)', 'y=x', 'y=x^0.5')
for i in range(3):
    ax=plt.subplot(221+i) 
    ax.set_xscale("log", nonposx='clip')
    plt.errorbar(Ns, means[i],stds[i])
    ax.set_title(title[i])
    ax.set_xlabel('N of random points')
    plt.axhline(y=np.exp(-1), color='r', linestyle='-')
    plt.show()


###############################################################################
##GAUSSIAN SAMPLING
###############################################################################


def f(x,y,z):
    return abs(np.cos((x**2+y**4)**0.5))*np.tanh(x**2+y**2+z**4)

def f_estimate(N):
    I=0
    for i in range(N):
        x,y,z=np.random.normal(size=3)
        I+=f(x,y,z)/(1.*N)
    return I

for i in range(4):
    print('the estimate of the integral with', 10**(i+1), 'points is')
    Integral=f_estimate(10**(i+1))
    print(Integral)
    
    