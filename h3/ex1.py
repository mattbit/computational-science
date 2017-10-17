# -*- coding: utf-8 -*-

import bisect
import timeit
import numpy as np
import functools as ft
import plotly.offline as py
from plotly.graph_objs import Histogram, Bar, Scatter, Figure, Layout
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

class Sampler(object):
    def sample(self, size=1):
        return np.array([self._sample() for _ in range(size)])

    def _sample(self):
      raise Exception("Sampler: _sample must be implemented!")


class AcceptRejectSampler(Sampler):
    
    def __init__(self, p):
        self.p = p
    
    def _sample(self):
        while True:
            # Choose one element randomly 
            i = np.random.choice(len(self.p))

            # Accept/reject
            r = np.random.random() * np.max(self.p)

            if r < self.p[i]:
                return i        


class TowerSampler(Sampler):

    def __init__(self, p):
        self.p = p

        # Compute the cumulative distribution
        # (@todo: there should be a better way)
        self.c = np.zeros(len(p))
        self.c[0] = p[0]
        for i in range(1, len(p)):
            self.c[i] = self.c[i-1] + p[i]


    def _sample(self):
        r = np.random.random()

        return bisect.bisect_left(self.c, r)


def generate_uniform_pmf(size=5):
    r = np.random.random(size)

    return r / np.sum(r)
    

def generate_exponential_pmf(size=5):
    r = -np.log(np.random.random(size))

    return r / np.sum(r)


def generate_sqrt_pmf(size=5):
    r = 1./np.sqrt(np.random.random(size))

    return r / np.sum(r)


"""
1. Consider n_items = 5. Write the “accept/reject” procedure to
   sample from the distribution. Sample many elements, make an
   histogram of the sampled data and compare with the
   theoretical proportions.

2. Repeat now this exercise with the “Tower sampling” procedure.
"""
'''
n_items = 5
p = generate_uniform_pmf(n_items)


ar = AcceptRejectSampler(p)
ts = TowerSampler(p)

data = [
    Bar(y=p, name="Theoretical probability"),
    Histogram(x=ar.sample(10000), histnorm="probability", name="Accept/Reject sampling"),
    Histogram(x=ts.sample(10000), histnorm="probability", name="Tower sampling")
]

lyt = Layout(title="Discrete sampling", xaxis=dict(title="Item"),
             bargroupgap=0.1)

py.plot(Figure(data=data, layout=lyt), filename="discrete_sampling_01.html",
        auto_open=False)

'''
"""
3. Now we wish to estimate numerically the time used by these two
   procedures. For many different values of n_items, compute the time
   needed to sample 100000 elements.
   a) How does this time grows with n_items in the case of the
      “accept/reject” procedure?
   b) And for the “Tower sampling” procedure?
   c) Can you explain why?
   d) As n_items grows which one seem to be the fastest way to solve
      the problem?
  
   ------------------------------------------------------------------
  
   a) The execution time grows … @todo: this is not simple
   b) The bottlneck in the tower sampling algorithm is the search of
      the correct interval, that can be done in time ~ log(N).
   c) @todo
   d) The tower sampling. 
"""

class SamplingBenchmark(object):
    """Benchmark different Samplers."""
    def __init__(self, pmf_generator, samplers, name="benchmark",
                pmf_num=100, sample_size=1000):
        self.pmf_generator = pmf_generator
        self.samplers = samplers
        self.ns = np.concatenate((
            np.arange(1, 10, step=1),
            np.arange(10, 100, step=10),
            np.arange(100, 1001, step=100)
        ))
        self.name = name
        self.pmf_num = pmf_num
        self.sample_size = sample_size


    def run(self):
        with ProcessPoolExecutor(max_workers=len(self.samplers)) as executor:
            futures = np.empty(len(self.samplers), dtype=object)

            for i, sampler in enumerate(self.samplers):
                futures[i] = executor.submit(self._benchmark_sampler, sampler)


            data = []

            for i in range(len(self.samplers)):
                data.append(Scatter(
                    x=self.ns,
                    y=futures[i].result(),
                    name=self.samplers[i].__name__
                ))

            py.plot(data, filename="{}.html".format(self.name), auto_open=False)


    def _benchmark_sampler(self, sampler):
        time = np.zeros(len(self.ns))

        for i, n_items in enumerate(self.ns):
            for _ in range(self.pmf_num):
                p = self.pmf_generator(n_items)
                s = sampler(p)
                sample = ft.partial(s.sample, self.sample_size)
                time[i] += timeit.timeit(sample, number=1)

            print("{}: n_items = {} completed".format(sampler.__name__, n_items))

        return time


SAMPLE_SIZE = 1000

bench_uniform = SamplingBenchmark(generate_uniform_pmf,
                                 [AcceptRejectSampler, TowerSampler],
                                 name="uniform", sample_size=SAMPLE_SIZE)

bench_uniform.run()


'''

ar_time = np.zeros(len(ns))
ts_time = np.zeros(len(ns))

for i, n_items in enumerate(ns):
    p = generate_uniform_pmf(n_items)

    ar = AcceptRejectSampler(p)
    ar_time[i] = timeit.timeit(ft.partial(ar.sample, SAMPLE_SIZE), number=1)
    
    ts = TowerSampler(p)
    ts_time[i] = timeit.timeit(ft.partial(ts.sample, SAMPLE_SIZE), number=1)

    print("unif: n_items = {} completed.".format(n_items))

py.plot([
    Scatter(x=ns, y=ar_time, name="Accept/Reject"),
    Scatter(x=ns, y=ts_time, name="Tower (with binary search)")
], filename="exec_time_01.html", auto_open=False)

"""
4. Repeat this exercise but now sample the ri from an exponential
   distribution. How does this time grows with nitem in the case
   of the “accept/reject” procedure? Why the difference?
"""
ar_time = np.zeros(len(ns))
ts_time = np.zeros(len(ns))

for i, n_items in enumerate(ns):
    p = generate_exponential_pmf(n_items)

    ar = AcceptRejectSampler(p)
    ar_time[i] = timeit.timeit(ft.partial(ar.sample, SAMPLE_SIZE), number=1)
    
    ts = TowerSampler(p)
    ts_time[i] = timeit.timeit(ft.partial(ts.sample, SAMPLE_SIZE), number=1)

    print("expo: n_items = {} completed.".format(n_items))


py.plot([
    Scatter(x=ns, y=ar_time, name="Accept/Reject"),
    Scatter(x=ns, y=ts_time, name="Tower (with binary search)")
], filename="exec_time_02.html", auto_open=False)

"""
5. Repeat this exercise but now sample the ri by writing
   ri = random.uniform^(−1/2) (or the equivalent formulation in your
   language). From which distribution are we now sampling?
   Compute numerically how the time now grows with n_items in the case
   of the “accept/reject” procedure.
"""
ar_time = np.zeros(len(ns))
ts_time = np.zeros(len(ns))

for i, n_items in enumerate(ns):
    p = generate_sqrt_pmf(n_items)

    ar = AcceptRejectSampler(p)
    ar_time[i] = timeit.timeit(ft.partial(ar.sample, SAMPLE_SIZE), number=1)
    
    ts = TowerSampler(p)
    ts_time[i] = timeit.timeit(ft.partial(ts.sample, SAMPLE_SIZE), number=1)

    print("sqrt: n_items = {} completed.".format(n_items))

py.plot([
    Scatter(x=ns, y=ar_time, name="Accept/Reject"),
    Scatter(x=ns, y=ts_time, name="Tower (with binary search)")
], filename="exec_time_03.html", auto_open=False)


"""
6. What do you conclude on these two methods to sample
   discrete distributions?
"""
# @todo
'''