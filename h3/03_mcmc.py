import plotly.offline as py
from plotly.graph_objs import *
from plotly import tools
import numpy as np
import concurrent.futures as futures

# Determine the next point of the Markov chain process.
# The subsequent position is either the same as the previous one (if the proposal would bring the point outside of the square)
# or given by a movement inside a smaller square of side "step_size" centered around the previous position
def move(position, step_size):
    step = np.random.uniform(-step_size, step_size, 2)
    proposed_position = position + step

    if abs(proposed_position[0]) <= 1 and abs(proposed_position[1]) <= 1:
        return proposed_position, True
    else:
        return position, False


# input of the function: number of steps of the markov chain, and side of the square
# of the proposed move
# output: chain of realizations, si vector, pi estimate
# optional (only if plot != 0): plot the chain for different lenghts



def markov_chain(num_steps, step_size, show_plot=False, name="markov_chain"):
    # initialization of the array of positions
    x = np.zeros((num_steps, 2))

    # initial position
    x[0] = [0, 0]

    accept_count = 0

    for i in range(1, num_steps):
        x[i], accepted = move(position=x[i-1], step_size=step_size)

        if accepted:
            accept_count += 1

    s = (np.sum(x**2, axis=1) <= 1)*4

    if show_plot:
        # plots with the chain of positions of the markov process
        # until the n-th one; n = 1000, 4000, 20000

        print("The acceptance ratio with step size {} is {}".format(step_size,
            accept_count/num_steps))

        lyt = Layout(
            title = "Sampling π with MCMC",
            yaxis = dict(scaleanchor="x", domain=[-1,1]),
            yaxis2 = dict(scaleanchor="x2", domain=[-1,1]),
            yaxis3 = dict(scaleanchor="x3", domain=[-1,1]),
            shapes = [{'type': 'circle', 'x0': -1, 'y0': -1, 'x1': 1, 'y1': 1, 'xref':'x', 'yref':'y'},
                    {'type': 'circle', 'x0': -1, 'y0': -1, 'x1': 1, 'y1': 1, 'xref':'x2', 'yref':'y2'},
                    {'type': 'circle', 'x0': -1, 'y0': -1, 'x1': 1, 'y1': 1, 'xref':'x3', 'yref':'y3'}]
        )

        fig = tools.make_subplots(rows=1, cols=3)

        for i, num_steps in enumerate([1000, 4000, 20000]):
            trace = Scatter(x=x[:num_steps, 0],
                            y=x[:num_steps, 1],
                            mode='lines+markers',
                            name=num_steps)

            fig.append_trace(trace, 1, i+1)
        
        fig['layout'].update(lyt)
        py.plot(fig, filename="{}.html".format(name))

    return x, s


def estimate_pi(num_steps=20000, step_size=0.1):
    _, s = markov_chain(num_steps, step_size)
    
    return np.mean(s)


NUM_STEPS = 20000

# Default step size
chain, samples = markov_chain(NUM_STEPS, 0.1, show_plot=True, name="default")


# Optimal step size
STEP_SIZE = 1.1812
markov_chain(NUM_STEPS, STEP_SIZE, show_plot=True, name="best_step_size")



with futures.ProcessPoolExecutor() as executor:
    fs = [executor.submit(estimate_pi, NUM_STEPS, STEP_SIZE) for _ in range(100)]
            
m = np.array([f.result() for f in fs])


print("Empirical variance: {}".format(np.sqrt(np.var(m))))


# batching procedure to estimate the variance
def batching(samples, variance):
    if len(samples) <= 1:
        return variance

    if len(samples) % 2 != 0:
        samples = np.append(samples, samples[-1])

    pairs = samples.reshape(len(samples)//2, 2)
    
    samples = np.mean(pairs, axis=1)
    variance = np.append(variance, np.var(samples)/len(samples))
    
    return batching(samples, variance)


variance = batching(samples, np.array([]))

# If you want the correct variance estimate you should run the code until a plateau
# actually pops out in this plot
py.plot([Scatter(y=variance, mode="lines+markers")], filename="batching.html")

def plateau(variance):
    err = 1
    index = 0

    for i in range(len(variance)-1):
        if variance[i+1] - variance[i] <= err:
            err = variance[i+1] - variance[i]
            index = i

    return variance[index]
    

print("Variance of direct uniform sampling:",  np.sqrt(np.pi*(4-np.pi)/NUM_STEPS))
print("Variance estimate through batching:", np.sqrt(plateau(variance)))
