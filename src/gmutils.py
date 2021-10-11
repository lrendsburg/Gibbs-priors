import numpy as np
from numpy.linalg import inv

from scipy.stats import multivariate_normal, ortho_group, poisson

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from src.utils import get_levels, load_dict

# Plots
linewidth = 8
SMALL_SIZE = 20.74
NORMAL_SIZE = 24.88
LARGE_SIZE = 35.83

plt.style.use('seaborn-whitegrid')
plt.rcParams['font.size'] = NORMAL_SIZE
plt.rcParams['axes.labelsize'] = NORMAL_SIZE
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['xtick.labelsize'] = NORMAL_SIZE
plt.rcParams['ytick.labelsize'] = NORMAL_SIZE
plt.rcParams['legend.fontsize'] = NORMAL_SIZE
plt.rcParams['figure.titlesize'] = LARGE_SIZE
plt.rcParams['axes.axisbelow'] = True

# Latex

plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'STIXGeneral'


def load_covariances(mode):
    """Loads prior and likelihood covariances depending on mode."""
    if mode=='skewed_prior':
        Sigma_0 = load_dict('../data/cov_skewed.p')
        Sigma =  load_dict('../data/cov_iso.p')
    elif mode=='skewed_like':
        Sigma_0 = load_dict('../data/cov_iso.p')
        Sigma =  load_dict('../data/cov_skewed.p')
    return Sigma_0, Sigma

def sample_cov(dim, mu=.02):
    """Generate a random dim-dimensional covariance matrix (positive definite)
    with high correlation."""
    U = ortho_group.rvs(dim)
    eigvals = poisson.pmf(np.arange(dim), mu=mu)
    print(eigvals)
    return (U.T @ np.diag(eigvals)) @ U


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

    # Approximate Lyapunov solution Sigma_G
    # Sigma_G = np.eye(F.shape[0])
    # A_k = np.eye(F.shape[0])
    # for i in range(K):
    #     Sigma_G += (A_k @ F) @ A_k.T
    #     A_k = A_k @ A

    # Other approximation

    Sigma_G_vec = inv(np.eye(d ** 2) - np.kron(A, A)) @ F.reshape(-1, 1)
    Sigma_G = Sigma_G_vec.reshape(d, d)
    return Sigma_G


def get_Sigma_S(Sigma_0, Sigma):
    n = Sigma.shape[0]
    Sigma_n = inv(inv(Sigma_0) + n * inv(Sigma))
    Lambda = inv(inv(Sigma_n) * np.eye(Sigma_n.shape[0])) # Set off-diagonals to 0
    Sigma_S = inv(inv(Lambda) - n * inv(Sigma))
    print(np.linalg.eig(Sigma_n)[0])
    print(np.linalg.eig(Lambda)[0])
    print(np.linalg.eig(Sigma_S)[0])
    return Sigma_S

# Generic plot function for nice densities
def grid_eval_fn(f, X, Y):
    """Evaluates the function f on the grid given by X and Y."""
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            theta = np.array([X[i, j], Y[i, j]])
            Z[i, j] = f(theta)
    return Z



def get_lims(samples):
    """Automatically find appropriate xlims and ylims to plot a density.
    Draw 100 samples from a distribution, center at mean and take lims based on largest marginal variance."""
    center = samples.mean(axis=0)
    std = max(samples.std(axis=0))
    xlims = [center[0] - 3 * std, center[0] + 3 * std]
    ylims = [center[1] - 3 * std, center[1] + 3 * std]
    return xlims, ylims


def plot_densities(density_fns, xlims, ylims, title, labels, colors, axis_labels=[r'$\theta_1$', r'$\theta_2$'], include_ticks=True, save_path=None):
    """Plot multiple densities."""
    n_steps = 500
    x = np.linspace(*xlims, n_steps)
    y = np.linspace(*ylims, n_steps)
    X, Y = np.meshgrid(x, y)


    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 12))
    legend_elements = []
    for i, density in enumerate(density_fns):
        Z = grid_eval_fn(density, X, Y)
        ax.contourf(X, Y, Z, levels=get_levels(Z, 1, upper_bound=True), colors=colors[i], alpha=0.5)
        legend_elements += [Line2D([0], [0], color=colors[i], lw=2, label=labels[i])]

    # Title, ticks, legends
    ax.set_xlim(xlims)
    ax.set_ylim(ylims)
    ax.legend(handles=legend_elements)
    # ax.legend(handles=legend_elements, loc='upper right')
    ax.set_title(title)
    ax.set_xlabel(axis_labels[0])
    ax.set_ylabel(axis_labels[1])

    if not include_ticks:
        ax.set_xticklabels([])
        ax.set_yticklabels([])

    if save_path is not None:
        plt.savefig(fname=save_path, bbox_inches='tight')

    plt.show()
    return


def plot_both(densities, xlims, ylims, titles, labels, colors, alphas=None, theta_0=None, sup_title=None, include_ticks=True, save_path=None):
    """Plots prior and posterior densities."""
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(2*linewidth, linewidth))

    plot_names = ['prior', 'posterior']
    axis_labels=[r'$\theta_1$', r'$\theta_2$']

    for j, ax in enumerate(axs):
        plot_name = plot_names[j]

        n_steps = 200
        x = np.linspace(*xlims[plot_name], n_steps)
        y = np.linspace(*ylims[plot_name], n_steps)
        X, Y = np.meshgrid(x, y)

        legend_elements = []
        for i, density in enumerate(densities[plot_name]):
            if alphas is None:
                alpha = .5
            else:
                alpha = alphas[plot_name][i]

            Z = grid_eval_fn(density, X, Y)
            ax.contourf(X, Y, Z, levels=get_levels(Z, 1, upper_bound=True),
                        colors=colors[plot_name][i], alpha=alpha)
            if labels[plot_name][i] is not None:
                legend_elements += [Line2D([0], [0], color=colors[plot_name][i], lw=2,
                                           label=labels[plot_name][i])]

        if theta_0 is not None and plot_name == 'posterior':
            ax.scatter(theta_0[0], theta_0[1], s=200, marker='x', color='black')

        # Title, ticks, legends
        ax.set_xlim(xlims[plot_name])
        ax.set_ylim(ylims[plot_name])
        ax.legend(handles=legend_elements)
        # ax.legend(handles=legend_elements, loc='upper right')
        ax.set_title(titles[plot_name])
        ax.set_xlabel(axis_labels[0])
        ax.set_ylabel(axis_labels[1])

        if not include_ticks:
            ax.set_xticklabels([])
            ax.set_yticklabels([])

    if sup_title is not None:
        plt.suptitle(sup_title)

    if save_path is not None:
        plt.savefig(fname=save_path, bbox_inches='tight')

    plt.show()
    return


def get_mean_embedding_of_gaussian(mean, cov, bandwidth):
    """Returns the mean embedding function of a Gaussian distribution with mean mean and covariance cov
    for a Gaussian kernel with a specified bandwidth."""
    return lambda x: multivariate_normal.pdf(x=x, mean=mean, cov=cov + bandwidth * np.eye(cov.shape[0]))

def get_gaussian_witness_fn(means, covs, bandwidth):
    """Returns the witness function for two Gaussians with a Gaussian kernel.
    Also returns the MMD."""
    embedding_1 = get_mean_embedding_of_gaussian(means[0], covs[0], bandwidth)
    embedding_2 = get_mean_embedding_of_gaussian(means[1], covs[1], bandwidth)
    witness = lambda x: embedding_1(x) - embedding_2(x)
    return witness

def get_empirical_mean_embedding(X, kernel_fn):
    """Returns the mean embedding of a distribution based on samples X for the kernel kernel_fn."""
    def empirical_mean_embedding_fn(x):
        kernel_vals = np.array([kernel_fn(y, x) for y in X])
        return kernel_vals.mean()
    return empirical_mean_embedding_fn

def get_empirical_witness_fn(X, Y, kernel_fn):
    """Returns the witness function of two distributions based on samples X and Y for the kernel kernel_fn.
    Also returns the empirical MMD."""
    embedding_1 = get_empirical_mean_embedding(X, kernel_fn)
    embedding_2 = get_empirical_mean_embedding(Y, kernel_fn)
    witness = embedding_1 - embedding_2

def plot_function(fn, xlims, ylims, title, axis_labels=[r'$\theta_1$', r'$\theta_2$'], include_ticks=True, save_path=None):
    """Generates heatmap of a function"""
    # Function
    n_steps = 300
    x = np.linspace(*xlims, n_steps)
    y = np.linspace(*ylims, n_steps)
    Y, X = np.meshgrid(x, y)
    Z = grid_eval_fn(fn, X, Y)
    z_min, z_max = Z.min(), - Z.min()

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 12))

    c = ax.pcolormesh(X, Y, Z, cmap='RdBu', vmin=z_min, vmax=z_max, shading='flat')
    fig.colorbar(c, ax=ax)

    # Title, ticks, legends
    ax.set_xlim(xlims)
    ax.set_ylim(ylims)

    # ax.legend(handles=legend_elements, loc='upper right')
    ax.set_title(title)
    ax.set_xlabel(axis_labels[0])
    ax.set_ylabel(axis_labels[1])

    if not include_ticks:
        ax.set_xticklabels([])
        ax.set_yticklabels([])

    if save_path is not None:
        plt.savefig(fname=save_path, bbox_inches='tight')

    plt.show()
    pass

def gaussian_entropy(Sigma):
    d = Sigma.shape[0]
    entropy = .5 * (d * np.log(2*np.pi*np.e) + np.log(np.linalg.det(Sigma)))
    return entropy