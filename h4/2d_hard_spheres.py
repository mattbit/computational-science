"""
==================================================================
=  A Markov Chain Monte Carlo for simulating hard spheres in 2d. =
==================================================================
"""

import numpy as np
import plotly.offline as py
from plotly.graph_objs import Heatmap, Layout, Figure


"""Basic definitions."""
NUM_SPHERES = 10  # number of spheres
LATTICE_SIZE = 5  # size of the lattice

def generate_initial_condition(size, num_spheres=0):
    """Generates a square lattice with a given number of spheres.""" 
    if num_spheres > size**2:
        raise Exception("The lattice can contain at most {} spheres.".format(size**2))

    lattice = np.zeros((size, size), dtype=int)
    if num_spheres == 0:
        return lattice

    step = size / np.ceil(np.sqrt(num_spheres))
    gutter = int(step / 2)
    step = int(step)
    count = 0
    for i in range(gutter, size, step):
        for j in range(gutter, size, step):
            lattice[i][j] = 1
            count += 1
            if count >= num_spheres:
                return lattice


def print_lattice(lattice):
    """Print the lattice configuration."""

    x = list(range(lattice.shape[0]))
    y = list(range(lattice.shape[1]))
    cs = [[0, 'rgba(255, 255, 255, 0'], [1, 'rgba(0, 0, 0, 1)']]
    h = Heatmap(x=x, y=y, z=lattice, colorscale=cs)
    lyt = Layout(
        xaxis=dict(showgrid=True, dtick=1, showticklabels=False),
        yaxis=dict(showgrid=True, dtick=1, showticklabels=False, scaleanchor="x")
    )
    py.plot(Figure(data=[h], layout=lyt), filename="_lattice.html")



lattice = generate_initial_condition(100, 50)
print_lattice(lattice)
