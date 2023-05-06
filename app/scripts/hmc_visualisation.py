import matplotlib as mpl
import matplotlib.pyplot as plt
from chmc import *
import itertools

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




class CHMCPlot():
    def __init__(self, loglikelihood, ran_x, ran_y=None):
        (self.fig, self.ax) = plt.subplots()

        self.loglikelihood = loglikelihood
        if ran_y is None:
            self.ran_x = ran_x
            self.ran_y = ran_x
        else:
            self.ran_x = ran_x
            self.ran_y = ran_y

        self.ax.set_xlim(self.ran_x)
        self.ax.set_ylim(self.ran_y)

        delta = 0.025
        x = np.arange(self.ran_x[0], self.ran_x[1], delta)
        y = np.arange(self.ran_y[0], self.ran_y[1], delta)
        self.X, self.Y = np.meshgrid(x, y)
        self.Z = self.loglikelihood(self.X, self.Y)

   #     self.ax.set_xlabel("x")
   #     self.ax.set_ylabel("y")

    def plot_X(self, data):
        x, y = data.T

        x0, y0 = x[0], y[0]
        xL, yL = x[-1], y[-1]

        deltaX = x[1:] - x[:-1]
        deltaY = y[1:] - y[:-1]

        self.ax.plot(x, y, '-o', c='black', mfc='red', mec='black', markersize=3)
        self.ax.plot([xL], [yL], 'D', c='blue', mfc='red', mec='black', markersize=6)
        self.ax.plot([x0], [y0], 'D', c='blue', mfc='red', mec='black', markersize=6)

        self.ax.quiver(x[:-1], y[:-1], deltaX, deltaY, angles='xy', scale_units='xy', scale=3, pivot='tail')
#        self.ax.quiver(x[:-1], y[:-1], deltaX, deltaY, angles='xy', scale_units='xy', scale=1, pivot='tail')

    def plot_failed_X(self, data):
        for p0, p1 in data:
            self.ax.plot([p0[0], p1[0]], [p0[1], p1[1]], marker = 'none', linestyle='--', color='blue')
            self.ax.plot([p1[0]], [p1[1]], marker = 'P', mfc='red', mec='black', markersize=8)



    def plot_contours(self, vals):
        self.ax.contour(self.X, self.Y, self.Z, vals, colors=['#580061', '#580061'])

    def plot_reflections(self, reflections, scale = 1):
        #reflections = reflections[:1]
        for x, n in reflections:
            parallel = [-n[1], n[0]]
            # Generate two points on the line perpendicular to the normal vector, centered on the point x
            point1 = x - parallel/np.linalg.norm(parallel)**2 * scale
            point2 = x + parallel/np.linalg.norm(parallel)**2 * scale

            # Plot the line perpendicular to the normal vector, centered on the point x
            self.ax.plot([point1[0], point2[0]], [point1[1], point2[1]], 'k--')
            self.ax.plot(x[0], x[1], 'rs', mec='black', markersize=6)  # Plot the center point


    def plot_heatmap(self, vmin, vmax):
        heatmap = self.ax.pcolormesh(self.X, self.Y, self.Z,
                       cmap='Blues',
                       shading='nearest')

       # heatmap.set_alpha(0.4)
        #        heatmap.set_clim( vmax=200)
        heatmap.set_clim( vmin=vmin, vmax = vmax)






def plot_Gaussian_CHMC():
    gauss_likelihood = GaussianLikelihood()

    path_length = 20
    uncompressed_constraint = -0.5
    compressed_constraint = -0.035

    x0 = np.array([0.8, 0.85])
    p0 = np.array([0.0, -0.55])

    x1 = np.array([0.1,0.01])
    p1 = np.array([-0.5, 0.55])

    uncompressed = CHMC(gauss_likelihood, 0.3, uncompressed_constraint)
    compressed = CHMC(gauss_likelihood, 0.3, compressed_constraint)

    uncompressed.initialise(x0, p0)
    compressed.initialise(x1, p1)

    data = x0
    data2 = x1
    for i in range(path_length):
        uncompressed.evolve()
        compressed.evolve()
        data = np.vstack((data, uncompressed.x))
        data2 = np.vstack((data2, compressed.x))


    #basic_plot = CHMCPlot(gauss_loglikelihood, [-1.5,1.5])

    # basic_plot.plot_heatmap(vmin=-1.0, vmax=0.4)
    # basic_plot.plot_X(data)
    # basic_plot.plot_contours([uncompressed_constraint])
    # basic_plot.plot_reflections(uncompressed.reflections, scale=1/2)
    # basic_plot.ax.set_title("Constrained Hamiltonian Monte Carlo")

    adaption_plot = CHMCPlot(gauss_loglikelihood, [-1.5,1.5])
    adaption_plot.plot_heatmap(None, 1.5)
    adaption_plot.plot_X(data)
    adaption_plot.plot_X(data2)
    adaption_plot.plot_contours([uncompressed_constraint, compressed_constraint])
    adaption_plot.plot_reflections(uncompressed.reflections, scale=1/2)
    adaption_plot.ax.set_title("Static step size with compressing constraint")


def plot_Square_CHMC():
    square_likelihood = SquareLikelihood()

    path_length = 7
    constraint = -0.1

    x0 = np.array([0.5, 0.45])
    p0 = np.array([-1.5, 0.4])

    chmc = CHMC(square_likelihood, 0.2, constraint)

    chmc.initialise(x0, p0)

    data = x0
    for i in range(path_length):
        chmc.evolve()
        data = np.vstack((data, chmc.x))


    basic_plot = CHMCPlot(square_loglikelihood, [-0.9,0.3], [0.1,0.9])

    #basic_plot.plot_heatmap()
    #basic_plot.plot_arrow(x0, p0)
    basic_plot.plot_failed_X(chmc.failed_x)
    basic_plot.plot_X(data)
    basic_plot.plot_reflections(chmc.reflections)
    basic_plot.plot_contours([constraint])
    basic_plot.ax.set_title("Epsilon Halving")


def plot_Bimodal_CHMC():
    bimodal_likelihood = BimodalLikelihood()

    path_length = 17
    path_length2 = 25
    constraint = -1

    x0 = np.array([2.9, 2.4])
    p0 = np.array([0.8, 3.1])

    x1 = np.array([-3.2, -2.4])
    p1 = np.array([1.2, 2.6])

    chmc = CHMC(bimodal_likelihood, 0.2, constraint)
    chmc2 = CHMC(bimodal_likelihood, 0.2, constraint)

    chmc.initialise(x0, p0)
    chmc2.initialise(x1, p1)

    data = x0
    data2 = x1
    for i in range(path_length):
        chmc.evolve()
        data = np.vstack((data, chmc.x))

    for i in range(path_length2):
        chmc2.evolve()
        data2 = np.vstack((data2, chmc2.x))

    basic_plot = CHMCPlot(bimodal_loglikelihood, [-6,6])

    basic_plot.plot_heatmap(vmin=-7, vmax=2.5)
    #basic_plot.plot_arrow(x0, p0)
  #  basic_plot.plot_failed_X(chmc.failed_x)
    basic_plot.plot_X(data)
    basic_plot.plot_X(data2)
    basic_plot.plot_contours([constraint])
    basic_plot.plot_reflections(chmc.reflections, scale=10)
    basic_plot.plot_reflections(chmc2.reflections, scale=10)
    basic_plot.ax.set_title("Clustering")


#plot_Square_CHMC()
plot_Gaussian_CHMC()
#plot_Bimodal_CHMC()

plt.show()