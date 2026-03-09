import numpy as np
from scipy.stats import uniform, truncnorm


class UniformLogPrior:
    def __init__(self, label, lower_bound, upper_bound, out_of_bounds_val=-1e10):
        self.label = label
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.out_of_bounds_val = out_of_bounds_val
        self.dist = uniform(
            loc=self.lower_bound, scale=self.upper_bound - self.lower_bound
        )

    def __call__(self, x):
        return self.dist.logpdf(x)


class GaussianLogPrior:
    def __init__(self, label, mu, sigma, lower_bound=-np.inf, upper_bound=np.inf):
        self.label = label
        self.mu = mu
        self.sigma = sigma
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.a = (lower_bound - mu) / sigma
        self.b = (upper_bound - mu) / sigma
        self.dist = truncnorm(a=self.a, b=self.b, loc=self.mu, scale=self.sigma)

    def __call__(self, x, shared_x=False):
        if shared_x:
            return self.dist.logpdf(x[:, np.newaxis]).T
        else:
            return self.dist.logpdf(x)