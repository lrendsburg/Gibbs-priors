import numpy as np
import pickle
from scipy.stats import invgamma, multivariate_normal, gaussian_kde

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.animation import FuncAnimation
import matplotlib.gridspec as gs
import seaborn as sns

# Plots
linewidth = 3.30719
SMALL_SIZE = 14
NORMAL_SIZE = 25
LARGE_SIZE = 32

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


def load_dict(filename):
    with open(filename, 'rb') as handle:
        dct = pickle.load(handle)
    return dct


def save_dict(dct, filename):
    with open(filename, 'wb') as handle:
        pickle.dump(dct,
                    handle,
                    protocol=pickle.HIGHEST_PROTOCOL)

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
    levels = [get_threshold(density_vals=density_vals, threshold=pow) for pow in powers]
    # powers = np.array([2 ** (-float(i)) for i in np.arange(1, num_levels + 3)])
    # quantiles = [100 * powers[:j].sum() for j in np.arange(1, num_levels+3)][2:]
    # levels = list(np.percentile(Z.flatten(), quantiles))
    levels = [get_threshold(density_vals=density_vals, threshold=.68)]
    if upper_bound:
        levels += [1e6]
    return levels

def plot_latent(target_density, samples_list=None, plot_kde=False, plot_samples=False, num_levels=5, xlims=[-2, 2], ylims=[0, 2],
                title=None, axis_labels=None, labels=None, colors=None, save_path=None):
    """Plot prior density as a contour plot and a set of samples."""
    # If only one set of samples is given, convert it to list artificially for compatibility reasons
    if type(samples_list)!=list:
        samples_list = [samples_list]
    n_dists = len(samples_list)
    assert n_dists<7, "Too many distributions for labeling!"
    # Initialize all labels if not given
    if title is None:
        title = r'Distributions on latent variable $\theta$'
    if axis_labels is None:
        axis_labels = [r'$\theta_1$', r'$\theta_2$']
    if labels is None:
        labels = ['Target density'] + [f'Distribution {k}' for k in np.arange(1, n_dists + 1)]
    if colors is None:
        colors = ['black', 'blue', 'green', 'red', 'cyan', 'magenta', 'yellow'][:n_dists + 1] # Pick colors of predefined order

    # Make grid
    x = np.linspace(*xlims, 50)
    y = np.linspace(*ylims, 50)
    X, Y = np.meshgrid(x, y)
    Z_target = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            theta = np.array([X[i, j], Y[i, j]])
            Z_target[i, j] = target_density(theta=theta)

    # Make plot
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 12))
    # Plot target density
    CS = ax.contour(X, Y, Z_target, levels=get_levels(Z_target, num_levels), colors=colors[0])
    legend_elements = [Line2D([0], [0], color=colors[0], lw=2, label=labels[0])]

    # Plot samples
    for k, samples in enumerate(samples_list, 1):
        if plot_kde:
            # Fit kernel density and evaluate
            kernel = gaussian_kde(samples.T)
            Z_samples = np.zeros_like(X)
            for i in range(X.shape[0]):
                for j in range(X.shape[1]):
                    Z_samples[i, j] = kernel(np.array([X[i, j], Y[i, j]]))
            # Plot contour
            ax.contour(X, Y, Z_samples, levels=get_levels(Z_samples, num_levels), colors=colors[k])
        if plot_samples:
            # Plot samples
            ax.scatter(samples[:, 0], samples[:, 1], color=colors[k], s=3)
        legend_elements += [Line2D([0], [0], color=colors[k], lw=2, label=labels[k])]

    # Title, ticks, legends
    ax.set_xlim(xlims)
    ax.set_ylim(ylims)
    ax.legend(handles=legend_elements, loc='upper right')
    ax.set_title(title)
    ax.set_xlabel(axis_labels[0])
    ax.set_ylabel(axis_labels[1])

    if save_path is not None:
        plt.savefig(fname=save_path, bbox_inches='tight')

    plt.show()
    return

def plot_latent_multitarget(target_density_list, samples_list=None, plot_kde=False, plot_samples=False, num_levels=5, xlims=[-2, 2], ylims=[0, 2],
                title=None, axis_labels=None, labels=None, colors=None, linestyles=None, save_path=None):
    """Plot prior density as a contour plot and a set of samples."""
    # If only one set of samples is given, convert it to list artificially for compatibility reasons
    if type(samples_list)!=list:
        samples_list = [samples_list]
    if type(target_density_list)!=list:
        target_density_list = [target_density_list]
    n_dists = len(samples_list) + len(target_density_list)
    assert n_dists<7, "Too many distributions for labeling!"
    # Initialize all labels if not given
    if title is None:
        title = r'Distributions on latent variable $\theta$'
    if axis_labels is None:
        axis_labels = [r'$\theta_1$', r'$\theta_2$']
    if labels is None:
        labels = ['Target density'] + [f'Distribution {k}' for k in np.arange(1, n_dists)]
    if colors is None:
        colors = ['black', 'blue', 'green', 'red', 'cyan', 'magenta', 'yellow'][:n_dists] # Pick colors of predefined order
    if linestyles is None:
        linestyles = ['solid'] * n_dists

    # Make grid
    x = np.linspace(*xlims, 50)
    y = np.linspace(*ylims, 50)
    X, Y = np.meshgrid(x, y)


    # Make plot
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 12))

    legend_elements = []
    for k, target_density in enumerate(target_density_list):
        Z_target = np.zeros_like(X)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                theta = np.array([X[i, j], Y[i, j]])
                Z_target[i, j] = target_density(theta=theta)

        # Plot target density
        ax.contour(X, Y, Z_target, levels=get_levels(Z_target, num_levels), colors=colors[k], linestyles=linestyles[k])
        legend_elements += [Line2D([0], [0], color=colors[k], lw=2, ls=linestyles[k], label=labels[k])]

    # Plot samples
    for k, samples in enumerate(samples_list, len(target_density_list)):
        if plot_kde:
            # Fit kernel density and evaluate
            kernel = gaussian_kde(samples.T)
            Z_samples = np.zeros_like(X)
            for i in range(X.shape[0]):
                for j in range(X.shape[1]):
                    Z_samples[i, j] = kernel(np.array([X[i, j], Y[i, j]]))
            # Plot contour
            ax.contour(X, Y, Z_samples, levels=get_levels(Z_samples, num_levels), colors=colors[k], linestyles=linestyles[k])
        if plot_samples:
            # Plot samples
            ax.scatter(samples[:, 0], samples[:, 1], color=colors[k], s=3)
        legend_elements += [Line2D([0], [0], color=colors[k], lw=2, ls=linestyles[k], label=labels[k])]

    # Title, ticks, legends
    ax.set_xlim(xlims)
    ax.set_ylim(ylims)
    ax.legend(handles=legend_elements, loc='upper right')
    ax.set_title(title)
    ax.set_xlabel(axis_labels[0])
    ax.set_ylabel(axis_labels[1])

    if save_path is not None:
        plt.savefig(fname=save_path, bbox_inches='tight')

    plt.show()
    return


def create_animation(paths,
                     colors,
                     names,
                     save_path,
                     figsize=(12, 12),
                     x_lim=(-2, 2),
                     y_lim=(-1, 3),
                     n_seconds=5):
    """Create an animation.
    Parameters
    ----------
    paths : list
        List of arrays representing the paths (history of x,y coordinates) the
        optimizer went through.
    colors :  list
        List of strings representing colors for each path.
    names : list
        List of strings representing names for each path.
    figsize : tuple
        Size of the figure.
    x_lim, y_lim : tuple
        Range of the x resp. y axis.
    n_seconds : int
        Number of seconds the animation should last.
    Returns
    -------
    anim : FuncAnimation
        Animation of the paths of all the optimizers.
    """
    if not (len(paths) == len(colors) == len(names)):
        raise ValueError

    path_length = max(len(path) for path in paths)

    # minimum = (1.0, 1.0)

    fig, ax = plt.subplots(figsize=figsize)

    scatters = [ax.scatter(None,
                           None,
                           label=label,
                           c=c) for c, label in zip(colors, names)]

    ax.legend(prop={"size": 25})
    ax.set_xlim(*x_lim)
    ax.set_ylim(*y_lim)
    # ax.plot(*minimum, "rD")

    def animate(i):
        for path, scatter in zip(paths, scatters):
            scatter.set_offsets(path[:i, :])

        ax.set_title(str(i))


    ms_per_frame = 1000 * n_seconds / path_length

    FuncAnimation(fig, animate, frames=path_length, interval=ms_per_frame).save(save_path)
    return






