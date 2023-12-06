from scipy import special
import anesthetic as ns
import matplotlib.pyplot as plt
import numpy as np

# path = "/Users/borisdeletic/CLionProjects/CHMC-Nested-Sampling/cmake-build-release/app/tests/gaussian_shells"
# path = "/Users/borisdeletic/CLionProjects/CHMC-Nested-Sampling/cmake-build-release/app/scaling/32/Phi4_0.256000_0.020000"
path = "/Users/borisdeletic/CLionProjects/CHMC-Nested-Sampling/cmake-build-release/app/Phi4_posterior_sampling"

samples = ns.read_chains(path)


# stats1 = samples.stats(nsamples=2000)
# params = ['logZ', 'D_KL', 'logL_P', 'd_G']
# fig, axes = ns.make_2d_axes(params, figsize=(6, 6), facecolor='w', upper=False)
# stats1.plot_2d(axes, label="model 1")
# axes.iloc[-1, 0].legend(bbox_to_anchor=(len(axes), len(axes)), loc='upper right')
n = 32

post = samples.posterior_points()
prior = samples.prior()

mag_squared = (prior['mag']**2).mean()
mag = np.abs(prior['mag']).mean()

chi = n**2 * (mag_squared - mag**2)
print(chi)
params = ['mag', 'mag_squared']

fig, axes = ns.make_2d_axes(params, figsize=(6, 6), facecolor='w')
prior.plot_2d(axes, alpha=0.9, label="prior")

plt.show()
