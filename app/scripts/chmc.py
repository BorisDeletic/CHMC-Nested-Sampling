import numpy as np

#classic CHMC values
var = np.array([1.3,1])
corr = 0.7

#clustering values
# var = np.array([1.5,1.5])
# corr = 0

@np.vectorize
def gauss_loglikelihood(x, y, mu=0, corr=0):
    return -0.5 * (((x-mu)/var[0])**2 + ((y-mu)/var[1])**2 - corr * (x-mu) * (y-mu) / (var[0] * var[1]))

@np.vectorize
def gauss_gradient(x, y, mu=0, corr=0):
    return np.array([-(x-mu) / var[0]**2 + 0.5 * corr * (y-mu) / (var[0] * var[1]),
                     -(y-mu) / var[1]**2 + 0.5 * corr * (x-mu) / (var[0] * var[1])])


@np.vectorize
def square_loglikelihood(x, y):
    return -x**6 - y**6

@np.vectorize
def bimodal_loglikelihood(x, y, mu=2):
    gauss1 = np.exp(gauss_loglikelihood(x,y,-mu))
    gauss2 = np.exp(gauss_loglikelihood(x,y,mu))

    return np.log(gauss1 + gauss2)


class SquareLikelihood():
    def loglikelihood(self, x):
        return square_loglikelihood(x[0], x[1])
    def gradient(self, x):
        return np.array([-6*x[0]**5,
                         -6*x[1]**5])


class BimodalLikelihood():
    def __init__(self):
        self.mu = 2
    def loglikelihood(self, x):
        return bimodal_loglikelihood(x[0], x[1], self.mu)
    def gradient(self, x):
        gauss1 = np.exp(gauss_loglikelihood(x[0],x[1],-self.mu))
        gauss2 = np.exp(gauss_loglikelihood(x[0],x[1],self.mu))

        return np.array([
            -2 * (x[0] + self.mu) * gauss1 - 2 * (x[0] - self.mu) * gauss2,
            -2 * (x[1] + self.mu) * gauss1 - 2 * (x[1] - self.mu) * gauss2,
        ]) / np.exp(self.loglikelihood(x))


class GaussianLikelihood():
    def __init__(self, var, mu, corr):
        self.var = var
        self.mu = mu
        self.corr = corr

    def loglikelihood(self, x, y):
        return -0.5 * (((x-self.mu)/self.var[0])**2 + ((y-self.mu)/self.var[1])**2 \
                       - self.corr * (x-self.mu) * (y-self.mu) / (self.var[0] * self.var[1]))

    def gradient(self, x, y):
        return np.array([-(x-self.mu) / self.var[0]**2 + 0.5 * self.corr * (y-self.mu) / (self.var[0] * self.var[1]),
                         -(y-self.mu) / self.var[1]**2 + 0.5 * self.corr * (x-self.mu) / (self.var[0] * self.var[1])])

    def vec_loglikelihood(self, x):
        return self.loglikelihood(x[0], x[1])

    def vec_gradient(self, x):
        return self.gradient(x[0], x[1])



# 2D CHMC algorithm
class CHMC():
    def __init__(self, prior, likelihood, epsilon, constraint):
        self.prior = prior
        self.likelihood = likelihood
        self.epsilon = epsilon
        self.constraint = constraint

    def initialise(self, x, p):
        self.reflections = []
        self.failed_x = []
        self.x = x
        self.p = p
        self.likelihood_grad = self.likelihood.vec_gradient(self.x)
        self.prior_grad = self.prior.vec_gradient(self.x)

    def reflect(self):
        self.likelihood_grad = self.likelihood.vec_gradient(self.x)

        fac = np.dot(self.p, self.likelihood_grad) / (np.dot(self.likelihood_grad, self.likelihood_grad))

        self.p = self.p - 2 * fac * self.likelihood_grad

    def evolve(self):
        halfP = self.p + 0.5 * self.epsilon * self.prior_grad

        nextX = self.x + self.epsilon * halfP

        if self.likelihood.vec_loglikelihood(nextX) < self.constraint:
            print("reflect")
          #  self.failed_x.append((self.x, nextX))
            self.reflect()
            self.reflections.append((self.x, self.likelihood_grad))

            halfP = self.p + 0.5 * self.epsilon * self.prior_grad
            nextX = self.x + self.epsilon * halfP

            if self.likelihood.vec_loglikelihood(nextX) < self.constraint:
                self.failed_x.append((self.x, nextX))

             #   halfP = self.p + 0.25 * self.epsilon * self.grad
                nextX = self.x + 0.5 * self.epsilon * halfP



        self.x = nextX
        self.prior_grad = self.prior.vec_gradient(self.x)

        self.p = halfP + 0.5 * self.epsilon * self.prior_grad
