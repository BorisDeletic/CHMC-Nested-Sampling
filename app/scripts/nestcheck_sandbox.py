import numpy as np
import dynesty

# Define the dimensionality of our problem.
ndim = 3

# Define our 3-D correlated multivariate normal log-likelihood.
C = np.identity(ndim)
C[C==0] = 0.95
Cinv = np.linalg.inv(C)
lnorm = -0.5 * (np.log(2 * np.pi) * ndim +
                np.log(np.linalg.det(C)))

def loglike(x):
    return -0.5 * np.dot(x, np.dot(Cinv, x)) + lnorm

# Define our uniform prior via the prior transform.
def ptform(u):
    return 20. * u - 10.

# Define our gradient with and without the Jacobian applied.
def grad_x(x):
    return -np.dot(Cinv, x)  # without Jacobian

def grad_u(x):
    return -np.dot(Cinv, x) * 20.  # with Jacobian for uniform [-10, 10)

# Sample with `grad_u` (including Jacobian).
sampler = dynesty.NestedSampler(loglike, ptform, ndim, sample='hslice',
                                gradient=grad_u)
sampler.run_nested()
res1 = sampler.results

print(res1.summary())
print(res1.asdict())
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

