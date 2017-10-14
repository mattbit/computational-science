# -*- coding: utf-8 -*-
import numpy as np
import plotly.offline as py
from plotly.graph_objs import Histogram


class AcceptRejectSampler(object):
    
    def __init__(self, p):
        self.p = p

    def sample(self, size=1):
        return np.array([self._sample() for _ in range(size)])    
    
    def _sample(self):
        while True:
            # Choose one element randomly 
            i = np.random.choice(len(p))
            # Accept/reject
            r = np.random.random()

            if r < self.p[i]:
                return i        

class TowerSampler(object):

    def __init__(self, p):
        self.p = p

        # Compute the cumulative distribution
        self.c = np.zeros(len(p)+1)
        for i in range(len(p)):
            self.c[i+1] = self.c[i] + p[i]

    def sample(self, size=1):
        raise Exception("TowerSampler: work in progress")

    def _sample(self):
        raise Exception("TowerSampler: work in progress")

n_items = 5

r = np.random.random(n_items)
Z = np.sum(r)
p = r / Z
print(p)
ar = AcceptRejectSampler(p)
s = ar.sample(1000)

# Make an histogram
py.plot([Histogram(x=s, histnorm="probability")])

ts = TowerSampler(p)
# …
