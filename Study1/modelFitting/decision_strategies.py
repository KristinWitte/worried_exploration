import jax
from jax.scipy.linalg import cholesky
import jax.numpy as jnp
import numpy as np
from jax import jit
from typing import Union


@jit
def ucb(mu: np.ndarray, sigma: np.ndarray, beta: float) -> np.ndarray:
    """
    Upper confidence bound decision rule.

    Note: Setting beta to < 0 is equal to lower confidence bound with positive beta.

    Args:
        mu (np.ndarray): Mean function
        sigma (np.ndarray): Standard error
        beta (float): Exploration factor, higher results in uncertainty being valued more.

    Returns:
        np.ndarray: Upper confidence bound
    """
    return mu + beta * jnp.sqrt(sigma)


@jit
def pos(mu: np.ndarray, sigma: np.ndarray, hmin: float = 50) -> np.ndarray:
    """
    Probability of safety decision rule. Calculates the probability that a given option has
    a value greater than safety threshold hmin.

    Args:
        mu (np.ndarray): [description]
        sigma (np.ndarray): [description]
        hmin (float, optional): [description]. Defaults to 50.

    Returns:
        np.ndarray: Probability of safety
    """
    return jax.scipy.stats.norm.cdf((mu - hmin + 1e-10) / sigma + 1e-10)

