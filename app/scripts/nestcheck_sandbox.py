# import numpy as np
# import dynesty
#
# # Define the dimensionality of our problem.
# ndim = 3
#
# # Define our 3-D correlated multivariate normal log-likelihood.
# C = np.identity(ndim)
# C[C==0] = 0.95
# Cinv = np.linalg.inv(C)
# lnorm = -0.5 * (np.log(2 * np.pi) * ndim +
#                 np.log(np.linalg.det(C)))
#
# def loglike(x):
#     return -0.5 * np.dot(x, np.dot(Cinv, x)) + lnorm
#
# # Define our uniform prior via the prior transform.
# def ptform(u):
#     return 20. * u - 10.
#
# # Define our gradient with and without the Jacobian applied.
# def grad_x(x):
#     return -np.dot(Cinv, x)  # without Jacobian
#
# def grad_u(x):
#     return -np.dot(Cinv, x) * 20.  # with Jacobian for uniform [-10, 10)
#
# # Sample with `grad_u` (including Jacobian).
# sampler = dynesty.NestedSampler(loglike, ptform, ndim, sample='hslice',
#                                 gradient=grad_u)
# sampler.run_nested()
# res1 = sampler.results
#
# print(res1.summary())
# print(res1.asdict())
#
#

# import dynesty.results
#
# res = dynesty.results.Results({
#     'logl': [1, 2, 3],
#     'samples_it': [1,2,3],
#     'samples_id': [1,2,3],
#     'samples_u': [[0,0], [1,1], [0.5,0]],
#     'samples': [[0,0], [1,1], [0.5,0]],
#     'nlive': 2,
#     'niter':3,
#     'ncall':3,
#     'eff':None
# })
#
# print(res.summary())

from scipy import special
import anesthetic as ns
import matplotlib.pyplot as plt
import numpy as np

path = "/Users/borisdeletic/CLionProjects/CHMC-Nested-Sampling/cmake-build-release/app/tests/gaussian_shells"

samples = ns.read_chains(path)


stats1 = samples.stats(nsamples=2000)
params = ['logZ', 'D_KL', 'logL_P', 'd_G']
fig, axes = ns.make_2d_axes(params, figsize=(6, 6), facecolor='w', upper=False)
stats1.plot_2d(axes, label="model 1")
axes.iloc[-1, 0].legend(bbox_to_anchor=(len(axes), len(axes)), loc='upper right')

post = samples.posterior_points()
prior_samples = samples.prior()

print(samples.columns.values)

fig, axes = ns.make_2d_axes(['p1', 'p2'])
prior_samples.plot_2d(axes, label="prior")
samples.plot_2d(axes, label="posterior")
axes.iloc[-1, 0].legend(bbox_to_anchor=(len(axes)/2, len(axes)), loc='lower center', ncol=2)

def sum_(x):
    return sum(x)

logZ = samples.logZ(nsamples=1000)
samples['logX'] = samples.logX()
samples['logZ'] = np.logaddexp.accumulate(samples.logw())

print(samples['logZ'])
fig, ax = plt.subplots()
ax.set_ylim([-2,-1.5])
# ax.plot(samples['logX'].get_weights())
ax.plot(samples['logZ'].values)

# samples.plot_1d(['logX'])
# ax.plot(np.exp(samples['logX'].values) * samples['logX'].get_weights())
# stats1.plot_2d(['logX', 'logZ'])

#samples.gui()

plt.show()
