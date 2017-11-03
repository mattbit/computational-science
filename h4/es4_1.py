import time
import numpy as np
import matplotlib.pyplot as plt
import plotly.offline as py
from plotly.graph_objs import Layout, Scatter, Heatmap, Histogram, Figure


###############################################################################
#1.1 DIRECT SAMPLING WITH 2 PINS
###############################################################################


NUM_SPHERES = 2  # number of spheres
LENGTH = 8  # length of the region available for the spheres
SIGMA = 0.75  # with of the spheres

#direct sampling of 2 hard sferes 
    #n_samples=number of samples
    #samples1,2 = samples of the coordinates of the first sphere 1,2

def sample2(sampling_function, size, L, sigma):
    samples1 = np.empty(n_samples)
    samples2 = np.empty(n_samples)
    
    for i in range(n_samples):
        p1, p2 = sampling_function(L, sigma)
        samples1[i] = p1
        samples2[i] = p2

    return samples1, samples2 


def sample2_naive(L, sigma):
    while True:
        p1 = np.random.uniform(0 + sigma, L - sigma)
        p2 = np.random.uniform(0 + sigma, L - sigma)

        if abs(p1 - p2) > 2*sigma:
            return p1, p2


def sample2_wrong(L, sigma):
    while True:
        p1 = np.random.uniform(0 + sigma, L - sigma)
        
        while True:
            p2 = np.random.uniform(0 + sigma, L - sigma)
            if abs(p1 - p2) > 2*sigma:
                return p1, p2


class PinSampler(object):
    def __init__(self, num_spheres, length, sigma):
        self.num_spheres = num_spheres
        self.length = length
        self.sigma = sigma

        if num_spheres * 2*sigma >= length:
            raise Exception("Invalid number of spheres!")

    def sample(self, size=1):
        s = np.empty((size, self.num_spheres))
        
        for i in range(size):
            s[i] = self._sample()

        return s


class DirectPinSampler(PinSampler):
    def _sample(self):
        s = np.empty(self.num_spheres)

        i = 0
        while True:
            p = np.random.uniform(self.sigma, self.length - self.sigma)

            if self._overlap(p, s):
                s = np.empty(self.num_spheres)
                i = 0

            else:
                s[i] = p
                i += 1

                if i >= self.num_spheres:
                    return s


    def _overlap(self, p, sample):
        return np.any(np.abs(sample - p) < 2*self.sigma)


# Ns = list(range(1, 20))
# times = []
# for n in Ns:
#     sampler = DirectPinSampler(n, 10, 0.1)
#     start = time.process_time()
#     sampler.sample(10)
#     delta = time.process_time() - start
#     times.append(delta)


# py.plot([Scatter(x=Ns, y=times)])

class InflateDeflatePinSampler(PinSampler):
    def _sample(self):
        # Deflated sample
        deflated = sorted(np.random.uniform(
            0,
            self.length - 2*self.num_spheres*self.sigma,
            size=self.num_spheres
        ))

        # Inflate
        sample = np.empty(self.num_spheres)
        for i, x in enumerate(deflated):
            sample[i] = x + (2*i + 1)*self.sigma

        return sample


# Ns = list(range(1, 18))
# times_direct = []
# times_infdef = []
# for n in Ns:
#     sampler_direct = DirectPinSampler(n, 10, 0.1)
#     sampler_infdef = InflateDeflatePinSampler(n, 10, 0.1)
    
#     start = time.process_time()
#     sampler_direct.sample(10)
#     delta = time.process_time() - start
#     times_direct.append(delta)

#     start = time.process_time()
#     sampler_infdef.sample(10)
#     delta = time.process_time() - start
#     times_infdef.append(delta)


# py.plot([
#     Scatter(x=Ns, y=times_direct, name="Direct"),
#     Scatter(x=Ns, y=times_infdef, name="Inflate-deflate"),
# ])
sampler = InflateDeflatePinSampler(int(20/1.5), 20, 0.75)
s = sampler.sample(10**5).flatten()

py.plot([Histogram(x=s)])





# for i in range(len(x1)):
#     print("({}, {})".format(x1[i], x2[i]))
# lyt = Layout(xaxis=dict(range=(0, 8)), yaxis=dict(range=(0, 8)))
# z, x, y = np.histogram2d(x=x1, y=x2, bins=80)

# data = [Heatmap(x=x, y=y, z=z.T)]
# py.plot(Figure(data=data, layout=lyt))



###############################################################################
#1.2 WRONG DIRECT SAMPLING WITH TWO PINS
###############################################################################

"""
x1, x2 = sample2wrong(10**6, L, sigma)

z, x, y = np.histogram2d(x=x1, y=x2, bins=40, normed=True)

data = [Heatmap(x=x, y=y, z=z)]
# bins = dict(start=0, end=8, size=8/20)
# data = [Histogram(x=x1, xbins=bins, histnorm="probability"), Histogram(x=x2, xbins=bins, histnorm="probability")]
py.plot(data)
"""

#historgram for the distributions of the coordinates of pin 1 and 2
# plt.figure(1)
# plt.hist(x1, bins = 10**2)
# plt.title('first pin')
# plt.figure(2)
# plt.hist(x2, bins = 10**2)
# plt.title('second pin')
# plt.show()




###############################################################################
#1.3 SAMPLE N PINS
###############################################################################