import numpy as np
import matplotlib.pyplot as plt


def f(x):
    return 1 - np.exp(-x)


def estimate_integral(samples, area):
    count = 0
    x, y = samples
    for i in range(len(x)):
        count += (y[i] < f(x[i]))

    return area * count / len(x)


def sample_rect(size=1):
    """Samples from a rectangle (0, 0), (1, f(1))."""
    x = np.random.random(size)
    y = np.random.random(size) * f(1)

    return x, y

"""
First, we proceed much in the same way as when computed π: we simply
draw uniformly numbers in the rectangle (0, 0), (1, 1 − e^−1) and
count the proportion of points below f(x). Show how this allows to
estimate the integrals (and an estimate of the error), and try it
for different values of the numbers of points N.
"""

def print_integral_summary(generate_sample, area, Ns):
    for N in Ns:
        values = np.array([estimate_integral(generate_sample(N), area) for _ in range(1000)])
        value = np.mean(values)
        stddev = np.sqrt(np.var(values))
        error = np.sqrt(np.mean(np.power(values - ANALYTICAL_RESULT, 2)))

        print("N = {:5}\t I = {}\t stddev = {:f}\t error = {:f}".format(N, value,
                                                                stddev, error))

ANALYTICAL_RESULT = np.exp(-1)

Ns = [10, 100, 1000]

print("Sampling from rectangle (0, 0), (1, f(1)).\n"
      "==========================================")
print_integral_summary(sample_rect, f(1), Ns)
print("")


"""2. Explain!"""

def sample_triangle(size=1):
    """Samples from a triangle y < x from (0, 0), to (1, 1)."""
    x = np.random.random(size)**0.5
    y = np.random.uniform(0, x, size)
    
    return x, y

print("Sampling from triangle (0, 0), (1, 1).\n"
      "==========================================")
print_integral_summary(sample_triangle, 0.5, Ns)
print("")

"""3. """

def g(x):
    return (1 - np.exp(-1)) * x**0.5


def sample_g(size=1):
    """Samples uniformly between the x-axis and g(x)."""
    x = np.random.random(size)**(2/3)
    y = np.random.uniform(0, g(x), size)

    return x, y

print("Sampling from under g(x).\n"
      "==========================================")
print_integral_summary(sample_g, 2/3*(1 - np.exp(-1)), Ns)
print("")



# Visualize the sampling
# x, y = sample_triangle(100)
# plt.plot(x, y, '+')
# plt.plot([0, 1, 1, 0], [0, 1, 0, 0])
# plt.show()


"""Returns an array of random points [x, y] uniformly distributed 
    between the  curve y(x)=x**(1/2) and y=0, for x belonging to [0,1[."""




# Visualize the sampling
# x, y = sample_g(100)
# plt.plot(x, y, '+')
# plt.plot(np.linspace(0, 1), g(np.linspace(0, 1)))
# plt.show()

# title = ('y=1-exp(-1)', 'y=x', 'y=x^0.5')
# for i in range(3):
#     ax = plt.subplot(221 + i)
#     ax.set_xscale("log", nonposx='clip')
#     plt.errorbar(Ns, means[i], stds[i])
#     ax.set_title(title[i])
#     ax.set_xlabel('N of random points')
#     plt.axhline(y=np.exp(-1), color='r', linestyle='-')
#     plt.show()


###############################################################################
# GAUSSIAN SAMPLING
###############################################################################


def h(x, y, z):
    return abs(np.cos((x**2 + y**4)**0.5)) * np.tanh(x**2 + y**2 + z**4)


def gaussian_average(h, N):
    I = 0
    for i in range(N):
        x, y, z = np.random.normal(size=3)
        I += h(x, y, z) / N

    return I


for N in [1000, 10000]:
    I = gaussian_average(h, N)
    print("The estimate of the integral with {} points is {}".format(
        N, I))
