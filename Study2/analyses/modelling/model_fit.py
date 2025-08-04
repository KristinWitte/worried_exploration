from numpyro.infer import MCMC, NUTS
import numpyro
import numpyro.distributions as dist
import numpy as np
import random
import jax
import jax.numpy as jnp
from jax import jit
from gp_model_functions import (
    model_func_POS_jit,
    model_func_UCB_jit,
    model_func_UCB_n_jit,
    square_dist,
    trial_func_UCB_jit,
    trial_func_UCB_n_jit,
    trial_func_POS_jit,
)
from typing import Union, Tuple


def kristinSimB0(
    ls: np.ndarray,
    tau: np.ndarray,
    n_subs: int,
    n_blocks: int,
    n_trials: int,
    grid_array: np.ndarray,
    distances: np.ndarray,
    envs: np.ndarray,
    start_x: np.ndarray,
    start_y: np.ndarray,
    seed: int = 123,
) -> Union[np.ndarray, np.ndarray, np.ndarray]:
    """
    Simulates data from the UCB model.

    Args:
        ls (np.ndarray): Length scale parameters for each subject
        tau (np.ndarray): Softmax temperature parameter for each subject
        beta (np.ndarray): UCB variance weighting parameter for each subject
        n_subs (int): Number of subjects
        n_blocks (int): Number of blocks
        n_trials (int): Number of trials per block
        grid_array (np.ndarray): Grid array to choose outcomes from
        distances (np.ndarray): Distance matrix
        seed (int, optional): RNG seed. Defaults to 123.

    Returns:
        Union[np.ndarray, np.ndarray, np.ndarray]: Simulated choices, simulated locations, and simulated outcomes
    """
    rng = np.random.RandomState(seed)

    
    ###### version with different grids
   #  # initialise x_sim array

    x_sim = np.zeros((n_subs * n_blocks, n_trials))
    sim_y = np.zeros((n_subs * n_blocks, n_trials))
     
  #
    

    choices = np.zeros((n_subs * n_blocks, n_trials))  

    # Loop through each subject/block and trial and generate choices according to the probabilities
    for block in range(
        n_subs * n_blocks
    ):  # This loops throught both blocks and subjects

            # get the correct grid for this block and sub
        grid = grid_array[envs[block]]
        choice_outcomes = grid.reshape(np.size(grid))
        
    ## use the real starting square in the data
        x_sim[block, 0] = start_x[block]
        sim_y[block, 0] = start_y[block]
    
        x_sim = x_sim.astype(np.int32)
        
            # Get outcomes for each location
        # loop through trials and get sim choices
        for trial in range(1,n_trials): # leave out first trial bc that is starting square

            p = trial_func_UCB_jit(
                    trial,
                    x_sim,
                    choice_outcomes[x_sim],
                    jnp.repeat(ls, n_blocks),
                    jnp.repeat(tau, n_blocks),
                    jnp.zeros(n_subs * n_blocks),
                    0,
                    distances
                )
        
            # get sim choices (code copied from create_sim_choices)

        # Initalise empty array for choices
    
            print(p)
        
            choices[block, trial] = int(random.choices(range(121), weights=p[block, :, :], k=1)[0]) 
            print(choices)
            x_sim[block,trial] = choices[block,trial]
            sim_y[block,trial] = choice_outcomes[x_sim[block,trial]]


    return x_sim, sim_y

def kristinSimUCB(
    ls: np.ndarray,
    tau: np.ndarray,
    beta: np.ndarray,
    n_subs: int,
    n_blocks: int,
    n_trials: int,
    grid_array: np.ndarray,
    distances: np.ndarray,
    envs: np.ndarray,
    start_x: np.ndarray,
    start_y: np.ndarray, 
    seed: int = 123,
) -> Union[np.ndarray, np.ndarray, np.ndarray]:
    """
    Simulates data from the UCB model.

    Args:
        ls (np.ndarray): Length scale parameters for each subject
        tau (np.ndarray): Softmax temperature parameter for each subject
        beta (np.ndarray): UCB variance weighting parameter for each subject
        n_subs (int): Number of subjects
        n_blocks (int): Number of blocks
        n_trials (int): Number of trials per block
        grid_array (np.ndarray): Grid array to choose outcomes from
        distances (np.ndarray): Distance matrix
        seed (int, optional): RNG seed. Defaults to 123.

    Returns:
        Union[np.ndarray, np.ndarray, np.ndarray]: Simulated choices, simulated locations, and simulated outcomes
    """
    rng = np.random.RandomState(seed)

    
    ###### version with different grids
   #  # initialise x_sim array

    x_sim = np.zeros((n_subs * n_blocks, n_trials))
    sim_y = np.zeros((n_subs * n_blocks, n_trials))
     
  #
    

    choices = np.zeros((n_subs * n_blocks, n_trials))  

    # Loop through each subject/block and trial and generate choices according to the probabilities
    for block in range(
        n_subs * n_blocks
    ):  # This loops throught both blocks and subjects

            # get the correct grid for this block and sub
        grid = grid_array[envs[block]]
        choice_outcomes = grid.reshape(np.size(grid))
        
    ## use the real starting square in the data
        x_sim[block, 0] = start_x[block]
        sim_y[block, 0] = start_y[block]
    
        x_sim = x_sim.astype(np.int32)
        
            # Get outcomes for each location
        # loop through trials and get sim choices
        for trial in range(1,n_trials): # leave out first trial bc that is starting square

            p = trial_func_UCB_jit(
                    trial,
                    x_sim,
                    choice_outcomes[x_sim],
                    jnp.repeat(ls, n_blocks),
                    jnp.repeat(tau, n_blocks),
                    jnp.repeat(beta, n_blocks),
                    0,
                    distances
                )
        
            # get sim choices
    
    
            choices[block, trial] = int(random.choices(range(121), weights=p[block, :, :], k=1)[0]) 
            
            x_sim[block,trial] = choices[block,trial]
            sim_y[block,trial] = choice_outcomes[x_sim[block,trial]]

        
    print('xsim {0}'.format(x_sim))
        
    return x_sim, sim_y


def kristinSimUCB_n(
    ls: np.ndarray,
    tau: np.ndarray,
    beta: np.ndarray,
    n_subs: int,
    n_blocks: int,
    n_trials: int,
    grid_array: np.ndarray,
    distances: np.ndarray,
    envs: np.ndarray,
    start_x: np.ndarray,
    start_y: np.ndarray, 
    seed: int = 123,
) -> Union[np.ndarray, np.ndarray, np.ndarray]:
    """
    Simulates data from the novelty bonus model

    Args:
        ls (np.ndarray): Length scale parameters for each subject
        tau (np.ndarray): Softmax temperature parameter for each subject
        beta (np.ndarray): novelty weighting parameter for each subject
        n_subs (int): Number of subjects
        n_blocks (int): Number of blocks
        n_trials (int): Number of trials per block
        grid_array (np.ndarray): Grid array to choose outcomes from
        distances (np.ndarray): Distance matrix
        seed (int, optional): RNG seed. Defaults to 123.

    Returns:
        Union[np.ndarray, np.ndarray, np.ndarray]: Simulated choices, simulated locations, and simulated outcomes
    """
    rng = np.random.RandomState(seed)
    
    # initialise x_sim array
    x_sim = np.zeros((n_subs * n_blocks, n_trials))
    sim_y = np.zeros((n_subs * n_blocks, n_trials))

    

    choices = np.zeros((n_subs * n_blocks, n_trials))  
    # initialise novelty
    novelty = np.zeros((n_subs*n_blocks, n_trials, 11**2))
    novelty += 1

    # Loop through each subject/block and trial and generate choices according to the probabilities
    for block in range(
        n_subs * n_blocks
    ):  # This loops throught both blocks and subjects

            # get the correct grid for this block and sub
        grid = grid_array[envs[block]]
        choice_outcomes = grid.reshape(np.size(grid))

            ## use the real starting square in the data
        x_sim[block, 0] = start_x[block]
        sim_y[block, 0] = start_y[block]
    
    
        x_sim = x_sim.astype(np.int32)
        
            # Get outcomes for each location
        for trial in range(1, n_trials): # get location for the next trial

            novelty[block,trial,:] = novelty[block,trial-1,:]
            p = trial_func_UCB_n_jit(
                        trial,
                        x_sim,
                        choice_outcomes[x_sim],
                        jnp.repeat(ls, n_blocks),
                        jnp.repeat(tau, n_blocks),
                        jnp.repeat(beta, n_blocks),
                        0,
                        distances,
                        novelty[:,trial,:]
                    )

            print(p.shape)
            print(random.choices(range(121), weights=p[block, :, :], k=1))
            choices[block, trial] = int(random.choices(range(121), weights=p[block, :, :], k=1)[0]) # have to take [0] to get that value out of the list although it is just one value either way

            # update novelty by that one new choice (doesn't work another way afaik)
            
            novelty[block,trial, choices[block,trial].astype(np.int32)] = 0
            x_sim[block,trial] = choices[block,trial]
            sim_y[block,trial] = choice_outcomes[x_sim[block,trial]]

    
    return x_sim, sim_y

def kristinSimPOS(
    ls: np.ndarray,
    tau: np.ndarray,
    n_subs: int,
    n_blocks: int,
    n_trials: int,
    grid_array: np.ndarray,
    distances: np.ndarray,
    envs: np.ndarray,
    start_x: np.ndarray,
    start_y: np.ndarray,
    seed: int = 123,
) -> Union[np.ndarray, np.ndarray, np.ndarray]:
    """
    Simulates data from the UCB model.

    Args:
        ls (np.ndarray): Length scale parameters for each subject
        tau (np.ndarray): Softmax temperature parameter for each subject
        beta (np.ndarray): UCB variance weighting parameter for each subject
        n_subs (int): Number of subjects
        n_blocks (int): Number of blocks
        n_trials (int): Number of trials per block
        grid_array (np.ndarray): Grid array to choose outcomes from
        distances (np.ndarray): Distance matrix
        seed (int, optional): RNG seed. Defaults to 123.

    Returns:
        Union[np.ndarray, np.ndarray, np.ndarray]: Simulated choices, simulated locations, and simulated outcomes
    """
    rng = np.random.RandomState(seed)

    
    ###### version with different grids
   #  # initialise x_sim array

    x_sim = np.zeros((n_subs * n_blocks, n_trials))
    sim_y = np.zeros((n_subs * n_blocks, n_trials))
     
  #
    

    choices = np.zeros((n_subs * n_blocks, n_trials))  

    # Loop through each subject/block and trial and generate choices according to the probabilities
    for block in range(
        n_subs * n_blocks
    ):  # This loops throught both blocks and subjects

            # get the correct grid for this block and sub
        grid = grid_array[envs[block]]
        choice_outcomes = grid.reshape(np.size(grid))
        
    ## use the real starting square in the data
        x_sim[block, 0] = start_x[block]
        sim_y[block, 0] = start_y[block]
    
        x_sim = x_sim.astype(np.int32)
        
            # Get outcomes for each location
        # loop through trials and get sim choices
        for trial in range(1,n_trials): # leave out first trial bc that is starting square

            p = trial_func_POS_jit(
                    trial,
                    x_sim,
                    choice_outcomes[x_sim],
                    jnp.repeat(ls, n_blocks),
                    jnp.repeat(tau, n_blocks),
                    jnp.zeros(n_subs * n_blocks),
                    0,
                    distances
                )
        
    
    
            choices[block, trial] = int(random.choices(range(121), weights=p[block, :, :], k=1)[0]) 
            
            x_sim[block,trial] = choices[block,trial]
            sim_y[block,trial] = choice_outcomes[x_sim[block,trial]]

        
    print('xsim {0}'.format(x_sim))


    return x_sim, sim_y

def create_sim_choices(
    p: np.ndarray,
    n_sim_subs: int,
    n_blocks: int,
    x: np.ndarray,
    y: np.ndarray,
    seed: int = 123,
) -> Union[np.ndarray, np.ndarray, np.ndarray]:
    """
    Creates simulated choices for the model based on an array of choice probabilities.

    Note: This is slow and could be sped up a lot, but it's only run once so it's not a big problem.

    Args:
        p (np.ndarray): Choice probabilities for each location
        n_sim_subs (int): Number of simulated subjects
        n_blocks (int): Number of blocks
        x (np.ndarray): Observed locations
        y (np.ndarray): Observed outcomes
        seed (int, optional): RNG seed. Defaults to 123.

    Returns:
        Union[np.ndarray, np.ndarray, np.ndarray]: Choices, observed outcomes, and observed locations
    """

    # Random state
    rng = np.random.RandomState(seed)

    # Reshape probabilities
    p_reshaped = p.squeeze().transpose((1, 0, 2))

    # Initalise empty array for choices
    choices = np.zeros((n_sim_subs * n_blocks, 10))

    # Loop through each subject/block and trial and generate choices according to the probabilities
    for block in range(
        n_sim_subs * n_blocks
    ):  # This loops throught both blocks and subjects
        for trial in range(10):
            choices[block, trial] = np.argwhere(
                rng.random() < np.cumsum(np.array(p_reshaped[block, trial, :]))
            )[0][0]
            

    return (
        choices.astype(int),
        y[: n_sim_subs * n_blocks, :],
        x[: n_sim_subs * n_blocks, :],
    )

def create_subject_params(
    name: str, n_subs: int
) -> Union[dist.Normal, dist.HalfNormal, dist.Normal]:
    """
    Creates group mean, group sd and subject-level offset parameters.

    Args:
        name (str): Name of the parameter
        n_subs (int): Number of subjects

    Returns:
        Union[dist.Normal, dist.HalfNormal, dist.Normal]: Group mean, group sd, and subject-level offset parameters
    """

    group_mean = numpyro.sample("{0}_group_mean".format(name), dist.Normal(0, 0.5))
    group_sd = numpyro.sample("{0}_group_sd".format(name), dist.HalfNormal(0.5))
    offset = numpyro.sample(
        "{0}_offset".format(name), dist.Normal(0, 0.5), sample_shape=(n_subs,)
    )

    return group_mean, group_sd, offset

def create_subject_params2(
    name: str, n_subs: int
) -> Union[dist.Normal, dist.HalfNormal, dist.Normal]:
    """
    Creates group mean, group sd and subject-level offset parameters.

    Args:
        name (str): Name of the parameter
        n_subs (int): Number of subjects

    Returns:
        Union[dist.Normal, dist.HalfNormal, dist.Normal]: Group mean, group sd, and subject-level offset parameters
    """

    group_mean = numpyro.sample("{0}_group_mean".format(name), dist.Normal(0, 0.5))
    group_sd = numpyro.sample("{0}_group_sd".format(name), dist.HalfNormal(0.5))
    offset0 = numpyro.sample(
        "{0}_offset0".format(name), dist.Normal(0, 0.5), sample_shape=(n_subs,)
    )
    offset1 = numpyro.sample(
        "{0}_offset1".format(name), dist.Normal(0, 0.5), sample_shape=(n_subs,)
    )   
    
    

    return group_mean, group_sd, offset0, offset1


def grid_model_UCB_LCB(
    coordinates: np.ndarray,
    observed_choices: np.ndarray,
    observed_outcomes: np.ndarray,
    missing: np.ndarray,
    distances: np.ndarray,
    n_subs: int,
    n_blocks: int,
    n_trials: int,
):
    """
    UCB model for the grid task.

    Args:
        coordinates (np.ndarray): Coordinates of locations where outcomes have been observed
        observed_choices (np.ndarray): Observed choices
        observed_outcomes (np.ndarray): Observed outcomes
        missing (np.ndarray): Boolean array indicating missing choices
        distances (np.ndarray): Distances between each pair of coordinates
        n_subs (int): Number of subjects
        n_blocks (int): Number of blocks
        n_trials (int): Number of trials per block
    """

    # Length scale
    ls_group_mean, ls_group_sd, ls_offset = create_subject_params("ls", n_subs)

    # Temperature
    tau_group_mean, tau_group_sd, tau_offset = create_subject_params("tau", n_subs)

    # UCB beta
    beta_group_mean, beta_group_sd, beta_offset = create_subject_params("beta", n_subs)
    
    
    # Subject-level
    ls_subject_transformed = numpyro.deterministic(
        "ls_subject_transformed",
        jax.scipy.special.expit(ls_group_mean + ls_group_sd * ls_offset) * 10 + 0.1,
    )
    tau_subject_transformed = numpyro.deterministic(
        "tau_subject_transformed",
        jax.scipy.special.expit(tau_group_mean + tau_group_sd * tau_offset) * 0.5
        + 0.00005,
    )
    beta_subject_transformed = numpyro.deterministic(
        "beta_subject_transformed",
        jax.scipy.special.expit(beta_group_mean + beta_group_sd * beta_offset) * 5 - 2.5,
    )
    
    
    # Run model
    p = model_func_UCB_jit(
        jnp.repeat(ls_subject_transformed, n_blocks),
        jnp.repeat(tau_subject_transformed, n_blocks),
        jnp.repeat(beta_subject_transformed, n_blocks),
        0,
        coordinates,
        observed_outcomes,
        distances,
        n_trials,
    )

    numpyro.sample(
        "obs",
        dist.Categorical(p.squeeze().transpose((1, 0, 2))[~missing[:, 1:]]),
        obs=observed_choices[:, 1:][~missing[:, 1:]],
    )

def grid_model_UCB_LCB_n(# novelty bias version (recycle beta parameter from ucb lcb)
    coordinates: np.ndarray,
    observed_choices: np.ndarray,
    observed_outcomes: np.ndarray,
    missing: np.ndarray,
    distances: np.ndarray,
    novel: np.ndarray,
    n_subs: int,
    n_blocks: int,
    n_trials: int,
):
    """
    UCB model for the grid task.

    Args:
        coordinates (np.ndarray): Coordinates of locations where outcomes have been observed
        observed_choices (np.ndarray): Observed choices
        observed_outcomes (np.ndarray): Observed outcomes
        missing (np.ndarray): Boolean array indicating missing choices
        distances (np.ndarray): Distances between each pair of coordinates
        n_subs (int): Number of subjects
        n_blocks (int): Number of blocks
        n_trials (int): Number of trials per block
    """

    # Length scale
    ls_group_mean, ls_group_sd, ls_offset = create_subject_params("ls", n_subs)

    # Temperature
    tau_group_mean, tau_group_sd, tau_offset = create_subject_params("tau", n_subs)

    # UCB beta
    beta_group_mean, beta_group_sd, beta_offset = create_subject_params("beta", n_subs)
    
    
    # Subject-level
    ls_subject_transformed = numpyro.deterministic(
        "ls_subject_transformed",
        jax.scipy.special.expit(ls_group_mean + ls_group_sd * ls_offset) * 10 + 0.1,
    )
    tau_subject_transformed = numpyro.deterministic(
        "tau_subject_transformed",
        jax.scipy.special.expit(tau_group_mean + tau_group_sd * tau_offset) * 0.5
        + 0.00005,
    )
    beta_subject_transformed = numpyro.deterministic(
        "beta_subject_transformed",
        jax.scipy.special.expit(beta_group_mean + beta_group_sd * beta_offset) * 5 - 2.5,
    )
    
    
    # Run model
    p = model_func_UCB_n_jit(
        jnp.repeat(ls_subject_transformed, n_blocks),
        jnp.repeat(tau_subject_transformed, n_blocks),
        jnp.repeat(beta_subject_transformed, n_blocks),
        0,
        coordinates,
        observed_outcomes,
        distances,
        novel,
        n_trials,
    )

    #print(p.shape)

    numpyro.sample(
        "obs",
        dist.Categorical(p.squeeze().transpose((1, 0, 2))[~missing[:, 1:]]),
        obs=observed_choices[:, 1:][~missing[:, 1:]],
    )

def grid_model_UCB_b0(
    coordinates: np.ndarray,
    observed_choices: np.ndarray,
    observed_outcomes: np.ndarray,
    missing: np.ndarray,
    distances: np.ndarray,
    n_subs: int,
    n_blocks: int,
    n_trials: int,
  
):
    """
    UCB model for the grid task but with beta fixed to 0.

    Args:
        coordinates (np.ndarray): Coordinates of locations where outcomes have been observed
        observed_choices (np.ndarray): Observed choices
        observed_outcomes (np.ndarray): Observed outcomes
        missing (np.ndarray): Boolean array indicating missing choices
        distances (np.ndarray): Distances between each pair of coordinates
        n_subs (int): Number of subjects
        n_blocks (int): Number of blocks
        n_trials (int): Number of trials per block
    """

    # Length scale
    ls_group_mean, ls_group_sd, ls_offset = create_subject_params("ls", n_subs)

    # Temperature
    tau_group_mean, tau_group_sd, tau_offset = create_subject_params("tau", n_subs)
    
    
    # Subject-level
    ls_subject_transformed = numpyro.deterministic(
        "ls_subject_transformed",
        jax.scipy.special.expit(ls_group_mean + ls_group_sd * ls_offset) * 10 + 0.1,
    )
    tau_subject_transformed = numpyro.deterministic(
        "tau_subject_transformed",
        jax.scipy.special.expit(tau_group_mean + tau_group_sd * tau_offset) * 0.5
        + 0.00005,
    )
    
    
    # Run model
    p = model_func_UCB_jit(
        jnp.repeat(ls_subject_transformed, n_blocks),
        jnp.repeat(tau_subject_transformed, n_blocks),
        jnp.zeros(n_subs * n_blocks),
        0,
        coordinates,
        observed_outcomes,
        distances,
        n_trials,
    )

    numpyro.sample(
        "obs",
        dist.Categorical(p.squeeze().transpose((1, 0, 2))[~missing[:, 1:]]),
        obs=observed_choices[:, 1:][~missing[:, 1:]],
    )


def grid_model_POS(
    coordinates: np.ndarray,
    observed_choices: np.ndarray,
    observed_outcomes: np.ndarray,
    missing: np.ndarray,
    distances: np.ndarray,
    n_subs: int,
    n_blocks: int,
    n_trials: int,
):
    """
    POS model for the grid task.

    Args:
        coordinates (np.ndarray): Coordinates of locations where outcomes have been observed
        observed_choices (np.ndarray): Observed choices
        observed_outcomes (np.ndarray): Observed outcomes
        missing (np.ndarray): Boolean array indicating missing choices
        distances (np.ndarray): Distances between each pair of coordinates
        n_subs (int): Number of subjects
        n_blocks (int): Number of blocks
        n_trials (int): Number of trials per block
    """

    # Length scale
    ls_group_mean, ls_group_sd, ls_offset = create_subject_params("ls", n_subs)

    # Temperature
    tau_group_mean, tau_group_sd, tau_offset = create_subject_params("tau", n_subs)

    # Subject-level

    ls_subject_transformed = numpyro.deterministic(
        "ls_subject_transformed",
        jax.scipy.special.expit(ls_group_mean + ls_group_sd * ls_offset) * 10 + 0.1,
    )
    tau_subject_transformed = numpyro.deterministic(
        "tau_subject_transformed",
        jax.scipy.special.expit(tau_group_mean + tau_group_sd * tau_offset) * 0.5
        + 0.00005,
    )

    # Run model
    p = model_func_POS_jit(
        jnp.repeat(ls_subject_transformed, n_blocks),
        jnp.repeat(tau_subject_transformed, n_blocks),
        jnp.zeros(n_subs * n_blocks)+0.1,
        0,
        coordinates,
        observed_outcomes,
        distances,
        n_trials,
    )

    numpyro.sample(
        "obs",
        dist.Categorical(p.squeeze().transpose((1, 0, 2))[~missing[:, 1:]]),
        obs=observed_choices[:, 1:][~missing[:, 1:]],
    )


def sample(
    model_func: callable,
    x_coords: np.ndarray,
    choices: np.ndarray,
    outcomes: np.ndarray,
    missing: np.ndarray,
    distances: np.ndarray,
    n_subs: int,
    n_blocks: int,
    n_trials: int,
    #cond: np.ndarray,
    n_samples: int = 4000,
    n_warmup: int = 2000,
    num_chains: int = 1,
    seed: int = 0,
    max_tree_depth: Union[int, Tuple] = 10,
) -> MCMC:
    """
    Samples from the posterior distribution of the model using NUTS, implemented in NumPyro.

    Args:
        model_func (callable): Model function
        x_coords (np.ndarray): Coordinates of locations where rewards have been observed
        choices (np.ndarray): Subject choice locations. For real data, this will be the same as x_coords,
        but for synthetic data, this will be a different set of locations.
        outcomes (np.ndarray): Rewards received at each location provided by x_coords.
        missing (np.ndarray): Boolean array indicating where choices are missing.
        distances (np.ndarray): Matrix of distances between all locations on the grid
        n_subs (int): Number of subjects
        n_blocks (int): Number of blocks
        n_trials (int): Number of trials per block
        n_samples (int, optional): Number of samples. Defaults to 4000.
        n_warmup (int, optional): Number of warmup iterations. Defaults to 2000.
        num_chains (int, optional): Number of chains to run. Defaults to 1.
        seed (int, optional): Random seed. Defaults to 0.
        max_tree_depth (Union[int, Tuple], optional): Maximum tree depth for the NUTS algorithm. Defaults to 10.

    Returns:
        MCMC: Results of sampling.
    """

    nuts_kernel = NUTS(model_func, max_tree_depth=max_tree_depth)

    mcmc = MCMC(
        nuts_kernel, num_samples=n_samples, num_warmup=n_warmup, num_chains=num_chains
    )

    rng_key = jax.random.PRNGKey(seed)
    mcmc.run(
        rng_key,
        x_coords,
        choices,
        outcomes,
        missing,
        distances,
        n_subs,
        n_blocks,
        n_trials,
        #cond,
    )

    return mcmc


def sample_n(
    model_func: callable,
    x_coords: np.ndarray,
    choices: np.ndarray,
    outcomes: np.ndarray,
    missing: np.ndarray,
    distances: np.ndarray,
    novelty: np.ndarray,
    n_subs: int,
    n_blocks: int,
    n_trials: int,
    n_samples: int = 4000,
    n_warmup: int = 2000,
    num_chains: int = 1,
    seed: int = 0,
    max_tree_depth: Union[int, Tuple] = 10,
) -> MCMC:
    """
    Samples from the posterior distribution of the model using NUTS, implemented in NumPyro.
    version for the novelty bonus model

    Args:
        model_func (callable): Model function
        x_coords (np.ndarray): Coordinates of locations where rewards have been observed
        choices (np.ndarray): Subject choice locations. For real data, this will be the same as x_coords,
        but for synthetic data, this will be a different set of locations.
        outcomes (np.ndarray): Rewards received at each location provided by x_coords.
        missing (np.ndarray): Boolean array indicating where choices are missing.
        distances (np.ndarray): Matrix of distances between all locations on the grid
        n_subs (int): Number of subjects
        n_blocks (int): Number of blocks
        n_trials (int): Number of trials per block
        n_samples (int, optional): Number of samples. Defaults to 4000.
        n_warmup (int, optional): Number of warmup iterations. Defaults to 2000.
        num_chains (int, optional): Number of chains to run. Defaults to 1.
        seed (int, optional): Random seed. Defaults to 0.
        max_tree_depth (Union[int, Tuple], optional): Maximum tree depth for the NUTS algorithm. Defaults to 10.

    Returns:
        MCMC: Results of sampling.
    """

    nuts_kernel = NUTS(model_func, max_tree_depth=max_tree_depth)

    mcmc = MCMC(
        nuts_kernel, num_samples=n_samples, num_warmup=n_warmup, num_chains=num_chains
    )

    rng_key = jax.random.PRNGKey(seed)
    mcmc.run(
        rng_key,
        x_coords,
        choices,
        outcomes,
        missing,
        distances,
        novelty,
        n_subs,
        n_blocks,
        n_trials,
        #cond,
    )

    return mcmc
