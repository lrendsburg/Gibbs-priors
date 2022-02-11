import numpy as np
from numpy.linalg import inv
from scipy.stats import multivariate_normal

from abc import ABCMeta, abstractmethod
import pickle

def load_dict(filename):
    with open(filename, 'rb') as handle:
        dct = pickle.load(handle)
    return dct


def save_dict(dct, filename):
    with open(filename, 'wb') as handle:
        pickle.dump(dct,
                    handle,
                    protocol=pickle.HIGHEST_PROTOCOL)

def load_covariances(mode):
    """Loads prior and likelihood covariances depending on mode."""
    if mode=='skewed_prior':
        Sigma_0 = load_dict('../data/cov_skewed.p')
        Sigma =  load_dict('../data/cov_iso.p')
    elif mode=='skewed_like':
        Sigma_0 = load_dict('../data/cov_iso.p')
        Sigma =  load_dict('../data/cov_skewed.p')
    return Sigma_0, Sigma

def get_gibbs_prior_covariance(Sigma_0, Sigma, n, method, K=500):
    """method is in {'inner', 'outer'}. Corresponds to the order of the KL divergence."""
    assert method in ['inner', 'outer'], "Method needs to be 'inner' or 'outer'."
    d = Sigma.shape[0]

    Sigma_n = inv(inv(Sigma_0) + n * inv(Sigma))
    A = n * Sigma_n @ inv(Sigma)
    Gamma = 1 / n * Sigma
    if method == 'inner':
        Lambda = inv(inv(Sigma_n) * np.eye(d))
    elif method == 'outer':
        Lambda = Sigma_n * np.eye(d)
    F = Lambda + (A @ Gamma) @ A.T

    Sigma_G_vec = inv(np.eye(d ** 2) - np.kron(A, A)) @ F.reshape(-1, 1)
    Sigma_G = Sigma_G_vec.reshape(d, d)
    return Sigma_G

def get_posterior_covariances(Sigma_0, Sigma, n, method):
    """Compute covariance matrices of posterior and approximation."""
    assert method in ['inner', 'outer'], "Method needs to be 'inner' or 'outer'."
    d = Sigma.shape[0]
    Sigma_n = inv(inv(Sigma_0) + n * inv(Sigma))
    if method == 'inner':
        Lambda_n = inv(inv(Sigma_n) * np.eye(d))
    elif method == 'outer':
        Lambda_n = Sigma_n * np.eye(d)
    return Sigma_n, Lambda_n

def get_threshold(density_vals, threshold):
    """Given a list of (unnormalized) density_vals=(f_1,...,f_n) and a threshold t, find the value
    g such that sum_{i: f_i<g}f_i/Z\approx t, where Z is the normalization constant.
    Interpretation: The area of points with density at most g has mass t. Used for contour plots."""
    Z = density_vals.sum()
    probas = np.sort(density_vals) / Z
    total_mass = 0
    pos = -1
    # Collect mass until mass threshold is reached
    while total_mass < threshold:
        pos += 1
        total_mass += probas[pos]
    # This means that adding mass until that point yields total mass threshold
    f_threshold = Z * probas[pos]
    return f_threshold

def get_levels(Z, num_levels, upper_bound=False):
    """Given a grid of data values Z and number of levels num_levels, compute sensible levels to plot.
    Here: outside of the areas contain powers of two of total mass"""
    density_vals = Z.flatten()
    powers = np.array([2 ** (-float(i)) for i in np.arange(1, num_levels+1)])
    levels = [get_threshold(density_vals=density_vals, threshold=.7)]
    if upper_bound:
        levels += [1e6]
    return levels

def grid_eval_fn(f, X, Y):
    """Evaluates the function f on the grid given by X and Y."""
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            theta = np.array([X[i, j], Y[i, j]])
            Z[i, j] = f(theta)
    return Z


class BayesianModel(metaclass=ABCMeta):
    """Skeleton for a Bayesian model in the PIGS context."""

    # Required properties:
    # self.d: dimension of latent space
    # self.ADVI_threshold: threshold (stopping criterion for training) for using ADVI in this model

    # Methods of the purely Bayesian model
    def __init__(self):
        """Initialize parameters used by all models."""
        self.ADVI_threshold = 0.05
        self.num_MCMC = 10
        self.latent_dim = None
        return

    @abstractmethod
    def get_params(self, theta):
        """Decomposes a latent variable theta into its different parts."""
        pass

    @abstractmethod
    def get_prior_density(self):
        """Returns the density function of the prior p(theta)."""
        pass

    @abstractmethod
    def sample_prior(self, n_samples=1):
        """Generates n_samples samples from the prior distribution."""
        pass

    @abstractmethod
    def sample_like(self, theta, n_samples=1):
        """Generates n_samples samples from the likelihood at theta."""
        pass

    # The following methods might not be given, if no closed-form posterior exists
    def compute_posterior_params(self, y):
        """Compute the parameters for the posterior distribution theta|y at observation y."""
        pass

    def get_posterior_density(self, y):
        """Returns the density function of the posterior p(theta|y) at observation y."""
        pass

    def sample_posterior(self, y, n_samples=1):
        """Generates n_samples samples from the posterior distribution at observation y."""
        pass

    # My PIGS algorithm
    def generate_gibbs_sequence(self, theta_0=None, T=100, posterior='exact', burn_in=0, thinning=1):
        """Generate Gibbs sequence with initial theta_0 (sampled from prior) by alternately sampling from
        likelihood and (approximation to) posterior."""
        if posterior == 'exact':
            sample_fn = self.sample_posterior

        # Initialize thetas
        if theta_0 is None:
            theta_0 = self.sample_prior()
        thetas = np.zeros((T + 1, len(theta_0.flatten())))
        thetas[0] = theta_0

        # Perform gibbs sampling with likelihood and (approximated) posterior
        for t in range(T):
            y = self.sample_like(theta=thetas[t])
            thetas[t + 1] = sample_fn(y=y)

        # Burn-in and thinning
        thetas = thetas[burn_in:][::thinning]
        return thetas

    # Method for testing the class
    def test(self, test_posterior=False, test_gibbs=False, test_ADVI=False, test_NUTS=False):
        """Tests whether implementation of purely Bayesian aspect (prior, likelihood, posterior) compiles."""
        print("Testing Bayesian model")
        # Evaluate prior
        prior_samples = self.sample_prior(n_samples=2)  # Generate prior samples
        prior_density = self.get_prior_density()  # Get prior density
        for i in range(2):
            proba = prior_density(prior_samples[i])  # Evaluate prior density at prior samples

        # Evaluate likelihood
        y_obs = self.sample_like(theta=prior_samples[0], n_samples=2)  # Generate observation at a prior sample

        # Evaluate posterior
        if test_posterior:
            print("Testing posterior")
            posterior_samples = self.sample_posterior(y=y_obs[0],
                                                      n_samples=2)  # Generate samples from posterior at first observation
            posterior_density = self.get_posterior_density(y=y_obs[0])  # Get posterior density
            for i in range(2):
                proba = posterior_density(posterior_samples[i])  # Evaluate posterior density at posterior samples

        print("Success!")
        return


class GaussianMean(BayesianModel):
    """Bayesian model for estimating the mean of a Gaussian distribution with known covariance matrix,
    given i.i.d.. samples from the distribution."""
    # Methods of the purely Bayesian model
    def __init__(self, mu_0, Sigma_0, Sigma, n_observ, test_model=True):
        """
        :param ADVI_threshold: ADVI training threshold
        :param mu_0: d-dimensional prior mean
        :param Sigma_0: dxd prior covariance matrix
        :param Sigma: dxd likelihood covariance matrix
        :param n_observ (int): number of observations for sampling from likelihood
        """
        super().__init__()
        self.ADVI_threshold = 0.05

        # Model hyperparameters
        self.mu_0 = mu_0
        self.Sigma_0 = Sigma_0
        self.Sigma = Sigma
        self.n_observ = n_observ
        self.latent_dim = len(mu_0)

        # Posterior parameters
        self.Sigma_0_inv = inv(self.Sigma_0)
        self.Sigma_inv = inv(self.Sigma)

        # Test if model compiles
        if test_model:
            self.test(test_posterior=True, test_gibbs=True, test_ADVI=True)

    def get_params(self, theta):
        """In this model, theta has no meaningful decomposition. This function is just the identity and not needed."""
        return theta

    # Prior and likelihood
    def get_prior_density(self):
        """Returns the density function of the prior p(theta)."""
        return lambda theta: multivariate_normal.pdf(x=theta, mean=self.mu_0, cov=self.Sigma_0)

    def sample_prior(self, n_samples=1):
        """Generates n_samples samples from the prior distribution."""
        return multivariate_normal.rvs(mean=self.mu_0, cov=self.Sigma_0, size=n_samples)

    def sample_like(self, theta, n_samples=1):
        """Generates n_samples samples from the likelihood at theta."""
        return multivariate_normal.rvs(mean=theta, cov=self.Sigma, size=(n_samples, self.n_observ))

    # Posterior methods
    def compute_posterior_params(self, y):
        """Compute the parameters for the posterior distribution theta|y at observation y."""
        y = y.reshape(self.n_observ, self.latent_dim)
        y_mean = np.mean(y, axis=0)
        Sigma_n = inv(self.Sigma_0_inv + self.n_observ * self.Sigma_inv)
        mu_n = Sigma_n @ (self.Sigma_0_inv @ self.mu_0 + self.n_observ * self.Sigma_inv @ y_mean)
        return mu_n, Sigma_n

    def get_posterior_density(self, y):
        """Returns the density function of the posterior p(theta|y) at observation y."""
        mu_n, Sigma_n = self.compute_posterior_params(y=y)
        return lambda theta: multivariate_normal.pdf(x=theta, mean=mu_n, cov=Sigma_n)

    def sample_posterior(self, y, n_samples=1):
        """Generates n_samples samples from the posterior distribution at observation y."""
        mu_n, Sigma_n = self.compute_posterior_params(y=y)
        return multivariate_normal.rvs(mean=mu_n, cov=Sigma_n, size=n_samples)

    def sample_exact_ADVI_mf(self, y, method, n_samples=1):
        """Exact solution to ADVI mean field approximation at observation y.
        method is in {'inner', 'outer'}. Corresponds to the order of the KL divergence."""
        assert method in ['inner', 'outer'], "Method needs to be 'inner' or 'outer'."
        mu_n, Sigma_n = self.compute_posterior_params(y=y)
        if method == 'inner':
            Lambda = inv(inv(Sigma_n) * np.eye(Sigma_n.shape[0]))
        elif method == 'outer':
            Lambda = Sigma_n * np.eye(Sigma_n.shape[0])  # Only difference: set non-diagonal elements to 0
        return multivariate_normal.rvs(mean=mu_n, cov=Lambda, size=n_samples)

    def get_exact_ADVI_mf_density(self, y, method):
        """method is in {'inner', 'outer'}. Corresponds to the order of the KL divergence."""
        assert method in ['inner', 'outer'], "Method needs to be 'inner' or 'outer'."
        mu_n, Sigma_n = self.compute_posterior_params(y=y)
        if method == 'inner':
            Lambda = inv(inv(Sigma_n) * np.eye(Sigma_n.shape[0]))
        elif method == 'outer':
            Lambda = Sigma_n * np.eye(Sigma_n.shape[0])
        return lambda theta: multivariate_normal.pdf(x=theta, mean=mu_n, cov=Lambda)

    def generate_gibbs_sequence_exact_ADVI_mf(self, method, theta_0=None, T=100, burn_in=0, thinning=1):
        """Same as generate_gibbs_sequence, but with exact ADVI mf approximation.
        method is in {'inner', 'outer'}. Corresponds to the order of the KL divergence."""
        assert method in ['inner', 'outer'], "method needs to be 'inner' or 'outer'."
        # Initialize thetas
        if theta_0 is None:
            theta_0 = self.sample_prior()
        thetas = np.zeros((T+1, len(theta_0.flatten())))
        thetas[0] = theta_0

        # Perform gibbs sampling with likelihood and (approximated) posterior
        for t in range(T):
            y = self.sample_like(theta=thetas[t])
            thetas[t+1] = self.sample_exact_ADVI_mf(y=y, method=method)

        # Burn-in and thinning
        thetas = thetas[burn_in:][::thinning]
        return thetas

    def get_exact_approximation_prior(self, y, method):
        """Returns density of the observation-dependent prior \pi_y \propto q(.|y) / f(y| .)."""
        mu_n, Sigma_n = self.compute_posterior_params(y=y)
        y = y.reshape(self.n_observ, self.latent_dim)
        y_mean = np.mean(y, axis=0)
        if method == 'inner':
            Lambda = inv(inv(Sigma_n) * np.eye(Sigma_n.shape[0]))
        elif method == 'outer':
            Lambda = Sigma_n * np.eye(Sigma_n.shape[0])
        Sigma_y = inv(inv(Lambda) - self.n_observ * self.Sigma_inv)
        mu_y = Sigma_y @ (inv(Lambda) @ mu_n - self.n_observ * self.Sigma_inv @ y_mean)
        return lambda theta: multivariate_normal.pdf(x=theta, mean=mu_y, cov=Sigma_y)