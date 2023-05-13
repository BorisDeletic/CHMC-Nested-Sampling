import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import numpy as np

params = {
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'font.size': 12,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.figsize': [6.4, 4.8],
    'figure.dpi': 100,
    'lines.linewidth': 2,
    'lines.markersize': 8,
    'axes.linewidth': 1.5,
    'xtick.major.width': 1.5,
    'ytick.major.width': 1.5,
    'xtick.minor.width': 1,
    'ytick.minor.width': 1,
    'xtick.major.size': 6,
    'ytick.major.size': 6,
    'xtick.minor.size': 4,
    'ytick.minor.size': 4,
    'axes.grid': False,
    'grid.alpha': 0.5,
    'grid.linewidth': 1,
    'image.cmap': 'viridis',
    'image.interpolation': 'nearest',
    'xtick.bottom': False,
    'ytick.left': False,
    'xtick.labelbottom': False,
    'ytick.labelleft': False
}

# Update the default parameters
mpl.rcParams.update(params)

class LikelihoodPlot():
    def __init__(self):
        (self.fig, self.ax) = plt.subplots()

    def plot_contours(self, data, vals=None):
        x = np.unique(data['x'].values)
        y = np.unique(data['y'].values)
        X, Y = np.meshgrid(x, y)

        size = int(np.sqrt(len(data['z'].values)))
        self.ax.contour(X, Y, data['z'].values.reshape((size, size)), vals)

    def plot_gradient(self, data):
        self.ax.quiver(data['x'].values, data['y'].values, data['dx'].values, data['dy'].values,
                       angles='xy', scale_units='xy', pivot='tail')


def plot_phi4():
    path = "/Users/borisdeletic/CLionProjects/CHMC-Nested-Sampling/cmake-build-release/app/phi4/"

    likelihood = pd.read_csv(path + "isocontours.dat", delimiter=" ")
    gradient = pd.read_csv(path + "gradient.dat", delimiter=" ")

    contours = [-100, -40, -10, 0, 2]
    plot = LikelihoodPlot()

    plot.plot_gradient(gradient)
    plot.plot_contours(likelihood, contours)
    plot.ax.set_title(r"$\phi^4$-Theory Likelihood and Gradient" + "\n(Disordered Phase)")
    plot.ax.set_xlabel(r"$\phi_i$")
    plot.ax.set_ylabel(r"$\phi_j$")

    plt.show()

def plot_rosenbrock():
    path = "/Users/borisdeletic/CLionProjects/CHMC-Nested-Sampling/cmake-build-debug/"

    likelihood = pd.read_csv(path + "isocontours.dat", delimiter=" ")
    gradient = pd.read_csv(path + "gradient.dat", delimiter=" ")

    contours = [10, 20, 30]
    plot = LikelihoodPlot()

    plot.plot_gradient(gradient)
    plot.plot_contours(likelihood, contours)
    plot.ax.set_title(r"Rosenbrock Function - Auto Gradient")
    plot.ax.set_xlabel(r"$x_1$")
    plot.ax.set_ylabel(r"$x_2$")

    plt.show()

plot_rosenbrock()