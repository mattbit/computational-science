# -*- coding: utf-8 -*-

import time
import bisect
import numpy as np
import functools as ft
import plotly.offline as py
from plotly.graph_objs import Histogram, Bar, Scatter, Figure, Layout
from concurrent.futures import ProcessPoolExecutor


class Sampler(object):
    def sample(self, size=1):
        return np.array([self._sample() for _ in range(size)])

    def _sample(self):
        raise Exception("Sampler: _sample must be implemented!")


class AcceptRejectSampler(Sampler):

    def __init__(self, p):
        self.p = p
        self.p_max = p.max()

    def _sample(self):
        while True:
            # Choose one element randomly
            i = np.random.choice(len(self.p))

            # Accept/reject
            r = np.random.random() * self.p_max

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
            self.c[i] = self.c[i - 1] + p[i]

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
    r = 1. / np.sqrt(np.random.random(size))

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
    """Benchmark a collection of Samplers."""

    def __init__(self, pmf_generator, samplers, name="benchmark",
                 sample_size=100000):
        self.pmf_generator = pmf_generator
        self.samplers = samplers
        self.ns = np.concatenate((
            np.arange(1, 10, step=1),
            np.arange(10, 1000, step=10),
            np.arange(10, 100000, step=500)
        ))

        self.name = name
        self.sample_size = sample_size
        self.pmfs = np.empty(len(self.ns), dtype=object)

    def run(self):
        """Benchmark the samplers and plot the execution time."""
        data = []

        self._init_pmfs()

        with ProcessPoolExecutor() as executor:
            for sampler in self.samplers:
                fn = ft.partial(self._benchmark, sampler=sampler)
                times = executor.map(fn, self.pmfs)

                data.append(Scatter(
                    x=self.ns,
                    y=list(times),
                    name=sampler.__name__
                ))

        py.plot(data, filename="benchmark_{}.html".format(
            self.name), auto_open=False)

    def _init_pmfs(self):
        for i, n_items in enumerate(self.ns):
            self.pmfs[i] = self.pmf_generator(n_items)

    def _benchmark(self, pmf, sampler):
        start_time = time.process_time()

        s = sampler(pmf)
        s.sample(self.sample_size)

        end_time = time.process_time()

        print("{}: n = {} completed.".format(sampler.__name__, len(pmf)))

        return end_time - start_time


SAMPLE_SIZE = 100000

samplers = [AcceptRejectSampler, TowerSampler]

bm_uniform = SamplingBenchmark(generate_uniform_pmf, samplers,
                               name="uniform", sample_size=SAMPLE_SIZE)
bm_uniform.run()


"""
4. Repeat this exercise but now sample the ri from an exponential
   distribution. How does this time grows with nitem in the case
   of the “accept/reject” procedure? Why the difference?
"""

bm_exponential = SamplingBenchmark(generate_exponential_pmf, samplers,
                                   name="exponential", sample_size=SAMPLE_SIZE)
bm_exponential.run()

"""
5. Repeat this exercise but now sample the ri by writing
   ri = random.uniform^(−1/2) (or the equivalent formulation in your
   language). From which distribution are we now sampling?
   Compute numerically how the time now grows with n_items in the case
   of the “accept/reject” procedure.
"""

bm_sqrt = SamplingBenchmark(generate_sqrt_pmf, samplers,
                            name="sqrt", sample_size=SAMPLE_SIZE)
bm_sqrt.run()

"""
6. What do you conclude on these two methods to sample
   discrete distributions?
"""
# @todo
