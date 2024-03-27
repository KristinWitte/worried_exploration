import jax
from jax.scipy.linalg import cholesky
import jax.numpy as jnp
import numpy as np
from jax import jit
from typing import Union
from decision_strategies import *


@jit
def softmax(v: np.ndarray, b: float):
    prob = jnp.exp(v / b)
    prob = jnp.nan_to_num(prob, nan=0)
    prob += 1e-20  # Make sure probabilities are never exactly zero

    # Normalise
    prob = prob / prob.sum()
    prob = jnp.clip(prob, 0.0001, 0.9999)
    return prob


@jit
def square_dist(X: np.ndarray, ls: float, Xs: np.ndarray = None) -> np.ndarray:
    """Calculates distance between values in two arrays, subject to a given length scale parameter.

    Adapted from PyMC3: https://github.com/pymc-devs/pymc/blob/main/pymc/gp/cov.py

    Args:
        X (np.ndarray): N-dimensional array
        ls (float): Length scale parameter.
        Xs (np.ndarray, optional): N-dimensional array for which to calculate distance
        with X. If not provided, distance is calculated for X with itself. Defaults to None.

    Returns:
        np.ndarray: Distance matrix
    """
    X = jnp.multiply(X, 1.0 / ls)
    X2 = jnp.sum(jnp.square(X), 1)
    if Xs is None:
        sqd = -2.0 * jnp.dot(X, jnp.transpose(X)) + (
            jnp.reshape(X2, (-1, 1)) + jnp.reshape(X2, (1, -1))
        )
    else:
        Xs = jnp.multiply(Xs, 1.0 / ls)
        Xs2 = jnp.sum(jnp.square(Xs), 1)
        sqd = -2.0 * jnp.dot(X, jnp.transpose(Xs)) + (
            jnp.reshape(X2, (-1, 1)) + jnp.reshape(Xs2, (1, -1))
        )
    return jnp.clip(sqd, 0.0, jnp.inf)


@jit
def exp_quad(X: np.ndarray, ls: float, Xs: np.ndarray = None) -> np.ndarray:
    """Exponentiated Quadratic kernel (also known as Squared Exponential or Radial
    Basis Function kernel).

    Args:
        X (np.ndarray): N-dimensional array
        ls (float): Length scale parameter.
        Xs (np.ndarray, optional): N-dimensional array for which to calculate covariance
        with X. If not provided, covariance is calculated for X with itself. Defaults to None.

    Returns:
        np.ndarray: Covariance
    """
    return jnp.exp(-0.5 * square_dist(X, ls, Xs))


@jit
def dist_to_exp_quad(dist: np.ndarray, ls: float) -> np.ndarray:
    """Converts a (non-squared) distance matrix to exponentiated quadratic (RBF) covariance matrix

    Args:
        dist (np.ndarray): Distance matrix
        ls (float): Length scale parameter

    Returns:
        np.ndarray: Covariance matrix
    """
    cov = (dist * (1 / ls)) ** 2
    return jnp.exp(-0.5 * cov)



@jit
def gp_predict(
    X: np.ndarray, y: np.ndarray, distances: np.ndarray, ls: float, mean: float = 0
) -> Union[np.ndarray, np.ndarray]:
    """Provides estimates of the mean and variance for a gaussian process at a set of points, based on
    prior observations. Uses a Exponentiated Quadratic kernel.

    Note: Takes a precomputed (non-squared) distance matrix instead of raw data to speed up computation slightly

    This code is adapted from the GPy implementation of GP posteriors:
    https://github.com/SheffieldML/GPy/blob/devel/GPy/inference/latent_function_inference/posterior.py

    This uses a slower but more accurate and more stable method to calculate covariance that involves
    solving equations properly rather than using matrix inversion
    (see https://github.com/SheffieldML/GPy/issues/253)

    Args:
        X (np.ndarray): Locations of previous observations.
        y (np.ndarray): Results of previous observations (e.g. reward received)
        distances (np.ndarray): Distance matrix for all possible X locations.
        ls (float): Length scale parameter.
        mean (float, optional): Mean function. Assumed to be uniform, and is set to the value provided.
        Defaults to 0.

    Returns:
        Union[np.ndarray, np.ndarray]: Returns mean and variance at the supplied points.
    """

    K = dist_to_exp_quad(distances[X[:, None], X[None, :]], ls)

    K = K + (1e-4 * jnp.identity(K.shape[0]))
    print(K)
    print(K.shape)
    LW = cholesky(K, lower=True, overwrite_a=True, check_finite=False)
    woodbury_vector = jax.scipy.linalg.cho_solve(
        (LW, True), y[:, None], check_finite=False, overwrite_b=True
    )

    Kx = dist_to_exp_quad(
        distances[X, :], ls
    )  # X test is all indices in the distance matrix

    # Compute mean
    mu = jnp.dot(Kx.T, woodbury_vector) + mean

    Kxx = jnp.diag(dist_to_exp_quad(distances, ls))
    tmp = jax.scipy.linalg.solve_triangular(
        LW, Kx, lower=True, check_finite=False, overwrite_b=True
    )

    # Compute variance
    var = (Kxx - jnp.square(tmp).sum(0))[:, None]

    return mu, var


def trial_GP_func_UCB(
    ls: float,
    tau: float,
    beta: float,
    mean: float,
    X_observed: np.ndarray,
    y_observed: np.ndarray,
    distances: np.ndarray,
) -> np.ndarray:
    """
    Vectorises the GP function across subjects & blocks using the upper confidence bound
    decision strategy.

    Args:
        ls (float): Length scale parameter for GP kernel
        tau (float): Softmax temperature
        beta (float): Upper confidence bound scaling parameter
        mean (float): Value of constant mean function
        X_observed (np.ndarray): Previously chosen points on the grid
        y_observed (np.ndarray): Previously received rewards for choosing points X_observed
        distances (np.ndarray): Distance matrix for all possible X locations.

    Returns:
        np.ndarray: Choice probability
    """

    # Get mean and variance
    mu, var = gp_predict(X_observed, y_observed, distances, ls, mean)

    #print(X_observed.shape)

    # UCB
    utility = ucb(mu, var, beta)

    # Softmax
    mu_p = softmax(utility - utility.max(), tau)

    return mu_p

def trial_GP_func_UCB_n(
    ls: float,
    tau: float,
    beta: float,
    mean: float,
    X_observed: np.ndarray,
    y_observed: np.ndarray,
    distances: np.ndarray,
    novel: np.ndarray,
) -> np.ndarray:
    """
    Vectorises the GP function across subjects & blocks using the novelty bonus
    decision strategy.

    Args:
        ls (float): Length scale parameter for GP kernel
        tau (float): Softmax temperature
        beta (float): Upper confidence bound scaling parameter
        mean (float): Value of constant mean function
        X_observed (np.ndarray): Previously chosen points on the grid
        y_observed (np.ndarray): Previously received rewards for choosing points X_observed
        distances (np.ndarray): Distance matrix for all possible X locations.

    Returns:
        np.ndarray: Choice probability
    """

    # Get mean and variance
    mu, var = gp_predict(X_observed, y_observed, distances, ls, mean)

    # reshape novelty to have same shape as var (121,1)
    novel = novel.reshape(-1,1)

    # UCB (we simply recycle the decision function from UCB but replace the variance with the dummy-coded novelty)
    utility = ucb(mu, novel, beta)

    # Softmax
    mu_p = softmax(utility - utility.max(), tau)

    return mu_p


def trial_GP_func_POS(
    ls: float,
    tau: float,
    hmin: float,
    mean: float,
    X_observed: np.ndarray,
    y_observed: np.ndarray,
    distances: np.ndarray,
) -> np.ndarray:
    """
    Vectorises the GP function across subjects & blocks using the probability of safety
    decision strategy.

    Args:
        ls (float): Length scale parameter for GP kernel
        tau (float): Softmax temperature
        best_outcome (float): Best outcome observed so far
        mean (float): Value of constant mean function
        X_observed (np.ndarray): Previously chosen points on the grid
        y_observed (np.ndarray): Previously received rewards for choosing points X_observed
        distances (np.ndarray): Distance matrix for all possible X locations.

    Returns:
        np.ndarray: Choice probability
    """

    # Get mean and variance
    mu, var = gp_predict(X_observed, y_observed, distances, ls, mean)

    # POS
    utility = pos(mu, var, hmin)

    # Softmax
    mu_p = softmax(utility, tau)

    return mu_p


# vmap - Mean function (always 50) and X (coordinates for the grid) are not vmapped
trial_GP_func_UCB_vmap = jax.vmap(
    trial_GP_func_UCB, in_axes=(0, 0, 0, None, 0, 0, None)
)
trial_GP_func_UCB_n_vmap = jax.vmap(
    trial_GP_func_UCB_n, in_axes=(0, 0, 0, None, 0, 0, None, 0)
)
trial_GP_func_POS_vmap = jax.vmap(
    trial_GP_func_POS, in_axes=(0, 0, 0, None, 0, 0, None)
)


def trial_func_UCB(
    trial: int,
    X_observed: np.ndarray,
    y_observed: np.ndarray,
    ls: np.ndarray,
    tau: np.ndarray,
    beta: np.ndarray,
    mean: float,
    distances: np.ndarray,
) -> np.ndarray:
    """
    Applies the GP model across subjects and blocks for a given trial using the upper confidence
    bound decision strategy

    Parameters must be provided as 1D arrays, with one parameter value per subject & block

    Args:
        trial (int): Trial number
        X_observed (np.ndarray): Previously chosen points on the grid
        y_observed (np.ndarray): Previously received rewards for choosing points X_observed
        ls (np.ndarray): Length scale parameter for GP kernel
        tau (np.ndarray): Softmax temperature
        beta (np.ndarray): Upper confidence bound scaling parameter
        mean (float): Value of constant mean function
        distances (np.ndarray): Distance matrix for all possible X locations.

    Returns:
        np.ndarray: Choice probability
    """

    # Get data for trials so far
    this_choices = X_observed[..., :trial]
    this_y = y_observed[..., :trial]

    #print(this_choices.shape)

    # Get choice probabilities
    mu_p = trial_GP_func_UCB_vmap(ls, tau, beta, mean, this_choices, this_y, distances)


    return mu_p

def trial_func_UCB_n(
    trial: int,
    X_observed: np.ndarray,
    y_observed: np.ndarray,
    ls: np.ndarray,
    tau: np.ndarray,
    beta: np.ndarray,
    mean: float,
    distances: np.ndarray,
    novel: np.ndarray,
) -> np.ndarray:
    """
    Applies the GP model across subjects and blocks for a given trial using the upper confidence
    bound decision strategy

    Parameters must be provided as 1D arrays, with one parameter value per subject & block

    Args:
        trial (int): Trial number
        X_observed (np.ndarray): Previously chosen points on the grid
        y_observed (np.ndarray): Previously received rewards for choosing points X_observed
        ls (np.ndarray): Length scale parameter for GP kernel
        tau (np.ndarray): Softmax temperature
        beta (np.ndarray): Upper confidence bound scaling parameter
        mean (float): Value of constant mean function
        distances (np.ndarray): Distance matrix for all possible X locations.

    Returns:
        np.ndarray: Choice probability
    """

    # Get data for trials so far
    this_choices = X_observed[..., :trial]
    this_y = y_observed[..., :trial]

    #print("gp function ", this_choices.shape)

    # Get choice probabilities
    mu_p = trial_GP_func_UCB_n_vmap(ls, tau, beta, mean, this_choices, this_y, distances, novel)

    #print(mu_p.shape)

    return mu_p



def trial_func_POS(
    trial: int,
    X_observed: np.ndarray,
    y_observed: np.ndarray,
    ls: np.ndarray,
    tau: np.ndarray,
    hmin: np.ndarray,
    mean: float,
    distances: np.ndarray,
) -> np.ndarray:
    """
    Applies the GP model across subjects and blocks for a given trial using the probability
    of safety strategy

    Parameters must be provided as 1D arrays, with one parameter value per subject & block

    Args:
        trial (int): Trial number
        X_observed (np.ndarray): Previously chosen points on the grid
        y_observed (np.ndarray): Previously received rewards for choosing points X_observed
        ls (np.ndarray): Length scale parameter for GP kernel
        tau (np.ndarray): Softmax temperature
        hmin (np.ndarray): Safety threshold
        mean (float): Value of constant mean function
        distances (np.ndarray): Distance matrix for all possible X locations.

    Returns:
        np.ndarray: Choice probability
    """

    # Get data for trials so far
    this_choices = X_observed[..., :trial]
    this_y = y_observed[..., :trial]

    # Get choice probabilities
    # POI function is used because it is identical to POS, only the threshold differs
    mu_p = trial_GP_func_POS_vmap(ls, tau, hmin, mean, this_choices, this_y, distances)

    return mu_p



# Set trial number as static
trial_func_UCB_jit = jit(trial_func_UCB, static_argnums=(0,))
trial_func_UCB_n_jit = jit(trial_func_UCB_n, static_argnums=(0,))
trial_func_POS_jit = jit(trial_func_POS, static_argnums=(0,))


def model_func_UCB_n(
    ls: np.ndarray,
    tau: np.ndarray,
    beta: np.ndarray,
    mean: float,
    X_observed: np.ndarray,
    y_observed: np.ndarray,
    distances: np.ndarray,
    novel: np.ndarray,
    n_trials: int,
) -> np.ndarray:
    """
    Iterates over trials, applying the GP model across blocks & subjects at each trial.
    Uses the novelty bonus decision strategy.

    Parameters must be provided as 1D arrays, with one parameter value per subject & block

    Args:
        ls (np.ndarray): Length scale parameter for GP kernel
        tau (np.ndarray): Softmax temperature
        beta (np.ndarray): Upper confidence bound scaling parameter
        mean (float): Value of constant mean function
        X_observed (np.ndarray): Previously chosen points on the grid. Should be provided as a
        2D array of shape (n subjects * n blocks, n trials)
        y_observed (np.ndarray): Previously received rewards for choosing points X_observed
        Should be provided as a 2D array of shape (n subjects * n blocks, n trials)
        distances (np.ndarray): Distance matrix for all possible X locations.
        n_trials (int): Number of trials. The model is applied to all but the last trial,
        as no choices are observed after the last outcome.

    Returns:
        np.ndarray: Choice probability
    """

    # List of results
    res_list = []

    # The last block isn't necessary as there's no choice after it
    # This starts from 1 because we select trials later using [:, :trial]
    for i in range(1, n_trials):
        nov = novel[:,i,:] # select correct trial of novelty

        res_list.append(
            trial_func_UCB_n_jit(
                i, X_observed, y_observed, ls, tau, beta, mean, distances, nov
            )
        )
    #print(jnp.stack([i for i in res_list]).shape)
    return jnp.stack([i for i in res_list])


def model_func_UCB(
    ls: np.ndarray,
    tau: np.ndarray,
    beta: np.ndarray,
    mean: float,
    X_observed: np.ndarray,
    y_observed: np.ndarray,
    distances: np.ndarray,
    n_trials: int,
) -> np.ndarray:
    """
    Iterates over trials, applying the GP model across blocks & subjects at each trial.
    Uses the upper confidence bound decision strategy.

    Parameters must be provided as 1D arrays, with one parameter value per subject & block

    Args:
        ls (np.ndarray): Length scale parameter for GP kernel
        tau (np.ndarray): Softmax temperature
        beta (np.ndarray): Upper confidence bound scaling parameter
        mean (float): Value of constant mean function
        X_observed (np.ndarray): Previously chosen points on the grid. Should be provided as a
        2D array of shape (n subjects * n blocks, n trials)
        y_observed (np.ndarray): Previously received rewards for choosing points X_observed
        Should be provided as a 2D array of shape (n subjects * n blocks, n trials)
        distances (np.ndarray): Distance matrix for all possible X locations.
        n_trials (int): Number of trials. The model is applied to all but the last trial,
        as no choices are observed after the last outcome.

    Returns:
        np.ndarray: Choice probability
    """

    # List of results
    res_list = []

    # The last block isn't necessary as there's no choice after it
    # This starts from 1 because we select trials later using [:, :trial]
    for i in range(1, n_trials):
        res_list.append(
            trial_func_UCB_jit(
                i, X_observed, y_observed, ls, tau, beta, mean, distances
            )
        )

    return jnp.stack([i for i in res_list])



def model_func_POS(
    ls: np.ndarray,
    tau: np.ndarray,
    hmin: np.ndarray,
    mean: float,
    X_observed: np.ndarray,
    y_observed: np.ndarray,
    distances: np.ndarray,
    n_trials: int,
) -> np.ndarray:
    """
    Iterates over trials, applying the GP model across blocks & subjects at each trial.
    Uses the probability of safety decision strategy.

    Parameters must be provided as 1D arrays, with one parameter value per subject & block

    Args:
        ls (np.ndarray): Length scale parameter for GP kernel
        tau (np.ndarray): Softmax temperature
        hmin (np.ndarray): Safety threshold
        mean (float): Value of constant mean function
        X_observed (np.ndarray): Previously chosen points on the grid. Should be provided as a
        2D array of shape (n subjects * n blocks, n trials)
        y_observed (np.ndarray): Previously received rewards for choosing points X_observed
        Should be provided as a 2D array of shape (n subjects * n blocks, n trials)
        distances (np.ndarray): Distance matrix for all possible X locations.
        n_trials (int): Number of trials. The model is applied to all but the last trial,
        as no choices are observed after the last outcome.

    Returns:
        np.ndarray: Choice probability
    """

    # List of results
    res_list = []

    # The last block isn't necessary as there's no choice after it
    for i in range(1, n_trials):
        res_list.append(
            trial_func_POS_jit(
                i, X_observed, y_observed, ls, tau, hmin, mean, distances
            )
        )

    return jnp.stack([i for i in res_list])


# Set number of trials to static
model_func_UCB_jit = jit(model_func_UCB, static_argnums=(7,))
model_func_UCB_n_jit = jit(model_func_UCB_n, static_argnums=(8,))
model_func_POS_jit = jit(model_func_POS, static_argnums=(7,))

