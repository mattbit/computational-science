import numpy as np
from plotly import tools
import plotly.offline as py
from plotly.graph_objs import Scatter, Layout, Figure

"""
Example usage:
```
from mcmc import MarkovChain

mc = MarkovChain(NUM_STEPS, STEP_SIZE)
mc.run()
mc.plot("Sampling π with MCMC")
print("π = {:f}".format(mc.pi_estimate()))
```
"""

class MarkovChain(object):
    def __init__(self, num_steps, step_size):
        self.num_steps = num_steps
        self.step_size = step_size
        self.acceptance_ratio = None
        self.x = np.empty((self.num_steps, 2))

    def reset(self):
        self.acceptance_ratio = None
        self.x = np.empty((self.num_steps, 2))

    def run(self):
        accept_count = 0

        # Initial position
        self.x[0] = [0, 0]

        for i in range(1, self.num_steps):
            self.x[i], accepted = self._move(self.x[i-1]) 
            accept_count += int(accepted)


        self.accept_ratio = accept_count / self.num_steps

        return self.x

        if show_plot:
            self._show_plot(accept_count)


    def _move(self, current_position):
        step = self._generate_step()
        proposed_position = current_position + step

        if abs(proposed_position[0]) <= 1 and abs(proposed_position[1]) <= 1:
            return proposed_position, True
        else:
            return current_position, False

    def _generate_step(self):
        return np.random.uniform(-self.step_size, self.step_size, 2)

    def samples(self):
        return (np.sum(self.x**2, axis=1) <= 1)*4

    def pi_estimate(self):
        return np.mean(self.samples())

    def plot(self, title):
        lyt = Layout(
            title = title,
            yaxis = dict(scaleanchor="x", domain=[-1,1]),
            yaxis2 = dict(scaleanchor="x2", domain=[-1,1]),
            yaxis3 = dict(scaleanchor="x3", domain=[-1,1]),
            shapes = [{'type': 'circle', 'x0': -1, 'y0': -1, 'x1': 1, 'y1': 1, 'xref':'x', 'yref':'y'},
                    {'type': 'circle', 'x0': -1, 'y0': -1, 'x1': 1, 'y1': 1, 'xref':'x2', 'yref':'y2'},
                    {'type': 'circle', 'x0': -1, 'y0': -1, 'x1': 1, 'y1': 1, 'xref':'x3', 'yref':'y3'}]
        )

        fig = tools.make_subplots(rows=1, cols=3)

        for i, num_steps in enumerate([1000, 4000, 20000]):
            trace = Scatter(x=self.x[:num_steps, 0],
                            y=self.x[:num_steps, 1],
                            mode='lines+markers',
                            name=num_steps)

            fig.append_trace(trace, 1, i+1)
        
        fig['layout'].update(lyt)
        py.plot(fig)
