import numpy as np
from numpy.linalg import inv
from scipy.stats import invgamma, multivariate_normal, lognorm, cauchy, uniform, norm

from abc import ABCMeta, abstractmethod

from src.ADVI import ADVI

from src import transforms
import torch
import torch.distributions as dists

dtype = torch.float
device = torch.device("cpu")


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

    # Methods for ADVI
    @abstractmethod
    def joint_logprob(self, theta, y):
        """Compute log probability log p(theta, y) of joint density for latent variable
        theta and observation y."""
        pass

    @abstractmethod
    def bijector(self):
        """Generates appropriate bijector for mapping latent variable to R^D for ADVI."""
        pass


    def sample_ADVI(self, y, method, n_samples=1):
        """Generates n_samples samples from the ADVI approximation to the posterior distribution at observation y."""
        joint_logprob_y = lambda theta: self.joint_logprob(theta=theta, y=y)
        advi = ADVI(joint_logprob_fn=joint_logprob_y, bijector=self.bijector(),
                    trafo_latent_dim=self.latent_dim, method=method, num_MCMC=self.num_MCMC, threshold=self.ADVI_threshold)
        advi.train()
        thetas = advi.sample(n_samples=n_samples)
        return np.array(thetas)

    # Methods for NUTS
    # todo implement

    # Hypothesis test for prior =? Gibbs marginal samples
    # todo implement

    # My PIGS algorithm
    def generate_gibbs_sequence(self, theta_0=None, T=100, posterior='exact', burn_in=0, thinning=1):
        """Generate Gibbs sequence with initial theta_0 (sampled from prior) by alternately sampling from
        likelihood and (approximation to) posterior."""
        # todo implement sample_fn = NUTS
        if posterior=='exact':
            sample_fn = self.sample_posterior
        if posterior=='ADVI_mf':
            sample_fn = lambda y: self.sample_ADVI(y=y, method='mean_field')
        if posterior=='ADVI_fr':
            sample_fn = lambda y: self.sample_ADVI(y=y, method='full_rank')

        # Initialize thetas
        if theta_0 is None:
            theta_0 = self.sample_prior()
        thetas = np.zeros((T+1, len(theta_0.flatten())))
        thetas[0] = theta_0

        # Perform gibbs sampling with likelihood and (approximated) posterior
        for t in range(T):
            y = self.sample_like(theta=thetas[t])
            thetas[t+1] = sample_fn(y=y)

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
        y_obs = self.sample_like(theta=prior_samples[0], n_samples=2) # Generate observation at a prior sample

        # Evaluate posterior
        if test_posterior:
            print("Testing posterior")
            posterior_samples = self.sample_posterior(y=y_obs[0], n_samples=2)  # Generate samples from posterior at first observation
            posterior_density = self.get_posterior_density(y=y_obs[0])  # Get posterior density
            for i in range(2):
                proba = posterior_density(posterior_samples[i])  # Evaluate posterior density at posterior samples

        if test_gibbs:
            print("Testing gibbs")
            # todo implement; should include a successful hypothesis test of samples from prior

        # Test ADVI
        if test_ADVI:
            print("Testing ADVI")
            advi_samples_mf = self.sample_ADVI(y=y_obs[0], method='mean_field', n_samples=2)  # Generate samples from ADVI approximation to posterior at first observation
            advi_samples_fr = self.sample_ADVI(y=y_obs[0], method='full_rank', n_samples=2)  # Generate samples from ADVI approximation to posterior at first observation

        # Test NUTS
        if test_NUTS:
            print("Testing NUTS")
            # todo implement

        print("Success!")
        return

class Bayesian_regression(BayesianModel):
    """Bayesian linear regression model for fixed features X and hyperparameters.
    Can evaluate prior, exact posterior, and approximate posterior (based on VI) densities."""
    # Methods of the purely Bayesian model
    def __init__(self, a_0, b_0, mu_0, Lambda_0, X, test_model=True):
        """
        :param ADVI_threshold: ADVI training threshold
        :param a_0: positive scaler shape parameter of inverse Gamma prior for sigmasq
        :param b_0: positive scaler scale parameter of inverse Gamma prior for sigmasq
        :param mu_0: d-dimensional shape parameter of Gaussian prior for beta
        :param Lambda_0: dxd precision parameter of Gaussian prior for beta
        :param X: nxd feature matrix (hyperparameter for this model)
        """
        super().__init__()
        self.ADVI_threshold = 0.05

        self.a_0 = a_0
        self.b_0 = b_0
        self.mu_0 = mu_0
        self.Lambda_0 = Lambda_0
        self.Lambda_0_inv = inv(Lambda_0)
        self.X = X
        self.XTX = X.T @ X
        self.n = X.shape[0]

        self.latent_dim = X.shape[1] + 1
        self.trafo_latent_dim = self.latent_dim

        # Initialize posterior parameters
        # Two of them (a_n and Lambda_n) do not depend on observation y and are computed directly
        self.a_n = self.a_0 + self.n / 2
        self.Lambda_n = X.T @ X + self.Lambda_0
        self.Lambda_n_inv = inv(self.Lambda_n)
        self.b_n = self.b_0
        self.mu_n = self.mu_0

        # Test if model compiles
        if test_model:
            self.test(test_posterior=True, test_gibbs=True, test_ADVI=True)

    def get_params(self, theta):
        """Decomposes latent variable into beta, sigmasq"""
        beta, sigmasq = theta[:-1], theta[-1]
        return beta, sigmasq

    # Prior and likelihood
    def get_prior_density(self):
        """Returns the density function of the prior p(theta)."""
        def prior_density_fn(theta):
            beta, sigmasq = self.get_params(theta)
            if sigmasq <= 0:
                p_density = 0
            else:
                p_sigmasq = invgamma.pdf(x=sigmasq, a=self.a_0, loc=0, scale=self.b_0)
                p_beta = multivariate_normal.pdf(x=beta, mean=self.mu_0, cov=sigmasq * self.Lambda_0_inv)
                p_density = p_sigmasq * p_beta
            return p_density
        return prior_density_fn

    def sample_prior(self, n_samples=1):
        """Generates n_samples samples from the prior distribution."""
        samples_prior = np.zeros((n_samples, self.latent_dim))
        samples_prior[:,-1] = invgamma.rvs(a=self.a_0, loc=0, scale=self.b_0, size=n_samples) # Sample sigmas
        for i in range(n_samples):
            samples_prior[i, :-1] = multivariate_normal.rvs(mean=self.mu_0, cov=samples_prior[i, -1] * self.Lambda_0_inv) # Sample betas
        return samples_prior

    def sample_like(self, theta, n_samples=1):
        """Generates n_samples samples from the likelihood at theta."""
        beta, sigmasq = self.get_params(theta)
        return multivariate_normal.rvs(mean=(self.X @ beta).flatten(), cov=sigmasq * np.eye(self.n), size=n_samples)


    # Posterior methods
    def compute_posterior_params(self, y):
        """Compute the parameters for the posterior distribution theta|y at observation y."""
        beta_hat = (np.linalg.pinv(self.XTX) @ self.X.T) @ y
        mu_n = self.Lambda_n_inv @ (self.Lambda_0 @ self.mu_0 + (self.XTX) @ beta_hat)
        b_n = self.b_0 + 1 / 2 * (y.T @ y + (self.mu_0.T @ self.Lambda_0) @ self.mu_0 - (mu_n.T @ self.Lambda_n) @ mu_n)
        return b_n, mu_n

    def get_posterior_density(self, y):
        """Returns the density function of the posterior p(theta|y) at observation y."""
        b_n, mu_n = self.compute_posterior_params(y=y)
        def posterior_density_fn(theta):
            beta, sigmasq = self.get_params(theta)
            if sigmasq <= 0:
                p_density = 0
            else:
                p_sigmasq = invgamma.pdf(x=sigmasq, a=self.a_n, loc=0, scale=b_n)
                p_beta = multivariate_normal.pdf(x=beta, mean=mu_n, cov=sigmasq * self.Lambda_n_inv)
                p_density = p_sigmasq * p_beta
            return p_density
        return posterior_density_fn

    def sample_posterior(self, y, n_samples=1):
        """Generates n_samples samples from the posterior distribution at observation y."""
        samples_posterior = np.zeros((n_samples, self.latent_dim))
        b_n, mu_n = self.compute_posterior_params(y=y)
        samples_posterior[:, -1] = invgamma.rvs(a=self.a_n, loc=0, scale=b_n, size=n_samples)  # Sample sigmas
        for i in range(n_samples):
            samples_posterior[i, :-1] = multivariate_normal.rvs(mean=mu_n, cov=samples_posterior[i, -1] * self.Lambda_n_inv)  # Sample betas
        return samples_posterior

    def joint_logprob(self, theta, y):
        """Compute log probability log p(theta, y) of joint density for latent variable
        theta and observation y."""
        beta, sigmasq = self.get_params(theta)
        # Convert hyperparameters to torch
        a_0 = torch.tensor(self.a_0, dtype=dtype)
        b_0 = torch.tensor(self.b_0, dtype=dtype)
        mu_0 = torch.tensor(self.mu_0, dtype=dtype)
        Lambda_0_inv = torch.tensor(self.Lambda_0_inv, dtype=dtype)
        X = torch.tensor(self.X, dtype=dtype)
        y = torch.tensor(y, dtype=dtype)

        # Compute logprobs
        logprob_sigmasq = -(a_0 + 1) * torch.log(sigmasq) - b_0 / sigmasq # Compute logprob of InvGamma by hand without the constant (doesn't matter for the gradients)
        logprob_beta = dists.MultivariateNormal(loc=mu_0, covariance_matrix=sigmasq * Lambda_0_inv).log_prob(beta)
        logprob_like = dists.MultivariateNormal(loc=X @ beta, covariance_matrix=sigmasq * torch.eye(self.n)).log_prob(y)
        logprob = logprob_sigmasq + logprob_beta + logprob_like
        return logprob

    def bijector(self):
        """Map that takes a latent variable theta and maps it to R^D for ADVI.
           Here: identity for beta and logarithm for sigmasq."""
        identity_trafo = transforms.IdentityTransform()
        log_trafo = transforms.invert_transform(transforms.ExpTransform())
        return transforms.ComposeTransform(transforms=[identity_trafo, log_trafo], n_dims=[self.latent_dim-1, 1])

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

    def joint_logprob(self, theta, y):
        """Compute log probability log p(theta, y) of joint density for latent variable
        theta and observation y."""
        y = torch.tensor(y, dtype=dtype)
        mu_0 = torch.tensor(self.mu_0, dtype=dtype)
        Sigma_0 = torch.tensor(self.Sigma_0, dtype=dtype)
        Sigma = torch.tensor(self.Sigma, dtype=dtype)

        logprob_theta = dists.MultivariateNormal(loc=mu_0, covariance_matrix=Sigma_0).log_prob(theta)
        logprob_like = dists.MultivariateNormal(loc=theta, covariance_matrix=Sigma).log_prob(y).sum()
        logprob = logprob_theta + logprob_like
        return logprob

    def bijector(self):
        """Map that takes a latent variable theta and maps it to R^D for ADVI.
           Here: identity for beta and logarithm for sigmasq."""
        identity_trafo = transforms.IdentityTransform()
        return transforms.ComposeTransform(transforms=[identity_trafo], n_dims=[self.latent_dim])

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


class Volatility(BayesianModel):
    """Stochastic volatility in time-series model. Latent variable of interest is the time
    series h_1,...h_T, which describe log volatilities for the price data y_1,...y_T (observations).
    No closed-form posterior available."""
    # Methods of the purely Bayesian model
    def __init__(self, cauchy_params, uniform_params, lognormal_params, len_sequence, test_model=True):
        """
        Latent variable theta=(mu, phi, sigma, h_1,..., h_T) always specified in that order.

        :param cauchy_params: tuple containing (loc, scale) parameters of the Cauchy prior for mu
        :param unif_params: tuple containing (lower, upper) of the Uniform prior for phi
        :param lognormal_params: tuple containing (loc, scale) parameters of the Lognormal prior for sigma
        :param test_model: Boolean for testing the model at initialization. Default is True.
        :param len_sequence: Integer describing the length T of the time series.
        """
        super().__init__()
        self.ADVI_threshold = 0.05
        self.num_MCMC = 10
        self.latent_dim = 3 + len_sequence

        self.cauchy_params = cauchy_params
        self.uniform_params = uniform_params
        self.uniform_locscale_params = (uniform_params[0], uniform_params[1] - uniform_params[0]) # Form required by scipy
        self.lognormal_params = lognormal_params
        self.len_sequence = len_sequence

        # Test if model compiles
        if test_model:
            self.test(test_gibbs=False, test_ADVI=True)

    def get_params(self, theta):
        """Decomposes a latent variable theta into its different parts mu, phi, sigma_sq, and h."""
        theta = theta.flatten()
        mu, phi, sigma_sq, h = theta[0], theta[1], theta[2], theta[3:]
        return mu, phi, sigma_sq, h

    # Prior and likelihood
    def get_prior_density(self):
        """Returns the density function of the prior p(theta)."""
        def prior_density_fn(theta):
            mu, phi, sigma_sq, h = self.get_params(theta=theta)
            p_mu = cauchy.pdf(mu, *self.cauchy_params)
            p_phi = uniform.pdf(phi, *self.uniform_locscale_params)
            p_sigma = dists.LogNormal(*self.lognormal_params).log_prob(torch.tensor(sigma_sq, dtype=dtype)).exp() # scipy logprob was not working and had a weird parametrization
            # In-place updates for probability of sequence
            p_h = norm.pdf(h[0], loc=mu, scale=sigma_sq / (1-phi**2))
            for t in range(1, self.len_sequence):
                p_h *= norm.pdf(h[t], loc=mu + phi * (h[t-1]-mu), scale=sigma_sq)
            p_density = p_mu * p_phi * p_sigma * p_h
            return p_density
        return prior_density_fn

    def sample_prior(self, n_samples=1, mu_0=None, phi_0=None, sigma_sq_0=None):
        """Generates n_samples samples from the prior distribution.
        If mu, phi, or sigma_sq are specified, they are fixed instead of sampled"""
        Thetas = np.zeros((n_samples, self.latent_dim))

        for i in range(n_samples):
            if mu_0 is not None:
                mu = mu_0
            else:
                mu = cauchy.rvs(*self.cauchy_params)
            if phi_0 is not None:
                phi = phi_0
            else:
                phi = uniform.rvs(*self.uniform_locscale_params)
            if sigma_sq_0 is not None:
                sigma_sq = sigma_sq_0
            else:
                sigma_sq = np.array(dists.LogNormal(*self.lognormal_params).sample())
            h = np.zeros(self.len_sequence)
            h[0] = norm.rvs(loc=mu, scale=sigma_sq / (1-phi**2))
            for t in range(1, self.len_sequence):
                h[t] = norm.rvs(loc=mu + phi * (h[t-1] - mu), scale=sigma_sq)
            Thetas[i] = np.concatenate([[mu], [phi], [sigma_sq], h])
        return Thetas

    def sample_like(self, theta, n_samples=1):
        """Generates n_samples samples from the likelihood at theta."""
        mu, phi, sigma_sq, h = self.get_params(theta=theta)
        return multivariate_normal.rvs(mean=np.zeros_like(h), cov=np.diagflat(np.exp(h)), size=n_samples)

    def joint_logprob(self, theta, y):
        """Compute log probability log p(theta, y) of joint density for latent variable
        theta and observation y."""
        mu, phi, sigma_sq, h = self.get_params(theta=theta)
        assert self.uniform_params[0] < phi and phi < self.uniform_params[1], "Phi outside support"
        assert sigma_sq > 0, "Sigma outside support"

        # Convert hyperparameters to torch
        y = torch.tensor(y, dtype=dtype)

        # Compute logprobs
        logprob_mu = dists.Cauchy(*self.cauchy_params).log_prob(mu)
        logprob_phi = dists.Uniform(*self.uniform_params).log_prob(phi)
        logprob_sigma = dists.LogNormal(*self.lognormal_params).log_prob(sigma_sq)
        logprob_h = dists.Normal(loc=mu, scale=sigma_sq / (1-phi**2)).log_prob(h[0])
        for t in range(1, self.len_sequence):
            logprob_h += dists.Normal(loc=mu + phi * (h[t-1] - mu), scale=sigma_sq).log_prob(h[t])
        logprob_y = dists.MultivariateNormal(loc=torch.zeros(self.len_sequence, dtype=dtype),
                                             covariance_matrix=torch.diagflat(torch.exp(h/2))).log_prob(y)
        assert logprob_mu.abs() < 1e9, "Logprob mu exploded"
        assert logprob_phi.abs() < 1e9, "Logprob phi exploded"
        assert logprob_sigma.abs() < 1e9, "Logprob sigma_sq exploded"
        assert logprob_h.abs() < 1e9, "Logprob h exploded"
        assert logprob_y.abs() < 1e9, "Logprob y exploded"
        logprob = logprob_mu + logprob_phi + logprob_sigma + logprob_h + logprob_y
        return logprob

    def bijector(self):
        """Map that takes a latent variable theta and maps it to R^D for ADVI."""
        identity_trafo = transforms.IdentityTransform()
        log_trafo = transforms.invert_transform(transforms.ExpTransform())
        logit_trafo = transforms.LogitTransform(*self.uniform_params)
        return transforms.ComposeTransform(transforms=[identity_trafo, logit_trafo, log_trafo, identity_trafo],
                                           n_dims=[1, 1, 1, self.len_sequence])


# class Template(BayesianModel):
#     """Model description."""
#     # Methods of the purely Bayesian model
#     def __init__(self, test_model=True):
#         """Description of hyperparameters."""
#         super().__init__()
#         self.ADVI_threshold = 0.05
#
#         # Test if model compiles
#         if test_model:
#             self.test(test_posterior=True, test_gibbs=True, test_ADVI=True)
#
#     def get_params(self, theta):
#         """Decomposes a latent variable theta into its different parts."""
#         pass
#
#     # Prior and likelihood
#     def get_prior_density(self):
#         """Returns the density function of the prior p(theta)."""
#         pass
#
#     def sample_prior(self, n_samples=1):
#         """Generates n_samples samples from the prior distribution."""
#         pass
#
#     def sample_like(self, theta, n_samples=1):
#         """Generates n_samples samples from the likelihood at theta."""
#         pass
#
#
#     # Posterior methods
#     def compute_posterior_params(self, y):
#         """Compute the parameters for the posterior distribution theta|y at observation y."""
#         pass
#
#     def get_posterior_density(self, y):
#         """Returns the density function of the posterior p(theta|y) at observation y."""
#         pass
#
#     def sample_posterior(self, y, n_samples=1):
#         """Generates n_samples samples from the posterior distribution at observation y."""
#         pass
#
#     def joint_logprob(self, theta, y):
#         """Compute log probability log p(theta, y) of joint density for latent variable
#         theta and observation y."""
#         # Convert hyperparameters to torch
#
#         # Compute logprobs
#         pass
#
#     def bijector(self):
#         """Map that takes a latent variable theta and maps it to R^D for ADVI."""
#         pass