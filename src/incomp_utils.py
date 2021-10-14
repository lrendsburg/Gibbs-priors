import numpy as np

def PGS(x_0, y_0, update_x, update_y, N):
    """Perform N steps of (Pseudo) Gibbs-sampling.

    Args:
        x_0, y_0: initial values.
        update_x: function that draws an x for a given y.
        update_y: function that draws an y for a given x.
        N: amount of iterations.

    Returns:
        sequence (N x 2 array): matrix with PGS sequence, rows indicate step.
    """
    sequence = np.zeros((N+1,2))
    sequence[0,:] = [x_0, y_0]
    for n in range(N):
        sequence[n+1, 0] = update_x(sequence[n, 1])
        sequence[n+1, 1] = update_y(sequence[n+1, 0])
    return sequence

def split_sequence(sequence, burn_in):
    """Split (Pseudo) Gibbs-sampling sequence into samples from both update orders.

    Args:
        sequence (N x 2 array): matrix with PGS sequence, rows indicate step.
        burn_in: discard first burn_in rows to assure convergence of samples.

    Returns:
        sequence_xy, sequence_yx (N - burn_in x 2 arrays): samples from stationary distributions for both update orders.
    """
    N = np.shape(sequence)[0]
    sequence_xy = sequence[burn_in+1:, :]
    sequence_yx = np.zeros_like(sequence_xy)
    sequence_yx[:, 0] = sequence_xy[:, 0]
    sequence_yx[:, 1] = sequence[burn_in:N-1, 1]
    return sequence_xy, sequence_yx

def update_x_normal(y, M):
    """Draw x given y, according to joint model, specified by M.

    Args:
        y: value to condition on.
        M: model parameter.

    Returns:
        x: one draw from the specified conditional.
    """
    y_hat = np.array([1, y, y**2])
    mu_y = -0.5 * np.dot(y_hat, M[1, :]) / np.dot(y_hat, M[2, :])
    sigma_y = np.sqrt(-0.5 / np.dot(y_hat, M[2, :]))
    x = np.random.normal(mu_y, sigma_y)
    return x

def update_y_normal(x, M):
    """Draw y given x, according to joint model, specified by M.

    Args:
        x: value to condition on.
        M: model parameter.

    Returns:
        y: one draw from the specified conditional.
    """
    x_hat = np.array([1, x, x**2])
    mu_x = -0.5 * np.dot(x_hat, M[:, 1]) / np.dot(x_hat, M[:, 2])
    sigma_x = np.sqrt(-0.5 / np.dot(x_hat, M[:, 2]))
    y = np.random.normal(mu_x, sigma_x)
    return y

def true_density_normal(x, y, M):
    """Evaluate density of the model with parameter M at (x, y).

    Args:
        x, y: point of density evaluation.
        M: model parameter.

    Returns:
        p: density at specified point.
    """
    x_hat = np.array([1, x, x**2])
    y_hat = np.array([1, y, y**2])
    p = np.exp(np.dot(x_hat, np.dot(M, y_hat)))
    return p