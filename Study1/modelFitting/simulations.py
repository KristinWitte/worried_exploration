# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 13:50:49 2022

@author: krist
"""

from decision_strategies import *
from gp_model_functions import *
from model_fit import *
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import requests
import os
import pickle
import argparse

if __name__ == "__main__":

    # Get model name
    parser = argparse.ArgumentParser(description="Run parameter recovery")
    parser.add_argument(
        "decision_model",
        type=str,
        default="ucb",
        help="Name of the decision model: ucb, poi, pos",
    )
    parser.add_argument(
        "kraken",
        type=int,
        default=0,
        help="kraken condition, 0 = absent, 1= present",
    )
    parser.add_argument(# what parameter estimates to simulate from
        "--estim",
        type=str,
        default="default",
        help='1 or 2'
    )
    parser.add_argument(# what data 
        "--dat",
        type=str,
        default="master",
        help='1 or 2'
    )      
    parser.add_argument(
        "--n_samples",
        type=int,
        default=2000,
        help="Number of samples to use for parameter recovery",
    )
    parser.add_argument(
        "--n_warmup",
        type=int,
        default=1000,
        help="Number of warmup samples to use for parameter recovery",
    )

    decision_model = parser.parse_args().decision_model
    kraken_present = parser.parse_args().kraken
    n_samples = parser.parse_args().n_samples
    n_warmup = parser.parse_args().n_warmup
    estim = parser.parse_args().estim
    dat = parser.parse_args().dat

    # Use SLURM job ID (for array jobs on high performance computing cluster)
    try:
        runID = int(os.environ["SLURM_ARRAY_TASK_ID"]) - 1
    except:
        runID = 1

    # Load data

    data = pd.read_csv(
        r"~/modelFitting/{0}.csv".format(dat)
    )    
    
    data = data[data["krakenPres"] == kraken_present]

    data = data[data["blocknr"] != 6]

    # Get important numbers
    N_SUBS = len(data["ID"].unique())
    N_BLOCKS = len(data.loc[data["ID"] == 1, "blocknr"].unique())
    N_TRIALS = len(data["click"].unique())

    # Initialise arrays
    choices = np.zeros((N_SUBS, N_BLOCKS, N_TRIALS))
    y = np.zeros((N_SUBS, N_BLOCKS, N_TRIALS))

    GRID_SIZE = 11

    # Extract and transform data
    for n_sub, sub in enumerate(data["ID"].unique()):
        sub_df = data[data["ID"] == sub]
        for n_block, block in enumerate(sub_df["blocknr"].unique()):
            sub_block_df = sub_df[sub_df["blocknr"] == block]
            sub_block_df = sub_block_df.fillna({'x': 0, 'y': 0})  # Set missing X and Y to zero
            choices[n_sub, n_block, :] = np.ravel_multi_index(
                sub_block_df[["x", "y"]].values.T.astype(int), (GRID_SIZE, GRID_SIZE)
            )
            y[n_sub, n_block, :] = (sub_block_df["z"] - 50) / 100
    choices = choices.astype(int)

    # Reshape data
    choices_flat = np.reshape(choices, (N_SUBS * N_BLOCKS, N_TRIALS))
    y_flat = np.reshape(y, (N_SUBS * N_BLOCKS, N_TRIALS))

    # Deal with missing data
    missing = np.isnan(y_flat)
    y_flat[np.isnan(y_flat)] = 0

    # One dimensional column vectors of inputs
    x1 = np.arange(GRID_SIZE)[:, None]
    x2 = np.arange(GRID_SIZE)[:, None]

    # Make cartesian grid
    X = jnp.dstack(np.meshgrid(x1.squeeze(), x2.squeeze())).reshape(-1, 2)

    # Get distances
    distances = jnp.sqrt(square_dist(X, 1))

    # Get grid info for simulations
    def make_grids(grids, size=11):
        out_grids = []

        for g in grids.columns:
            new_grid = np.zeros((11, 11))
            gtemp = grids.iloc[:,g] # get 1 column (1 grid of the 30)
            for k in range(len(gtemp)):# loop through rows
                row = gtemp[k]
                for j in range(11):# loops through columns of the grid
                    new_grid[int(k), int(j)] = (row[str(j)] - 50) / 100
            out_grids.append(new_grid[:size, :size].copy())

        return out_grids
   
   # get underlying rewards for the reward grids
    grids = pd.read_json("~/modelFitting/sample_grid.json")
    grid_arrays = make_grids(grids)

    # Generate parameters for simulations
    rng_key = jax.random.PRNGKey(0)
    N_SUBS_SIM = N_SUBS
    missing = missing[:N_SUBS_SIM*N_BLOCKS, :]

    # read in parameter estimates
    # cluster directory
    if estim == "default":
        estimates = pd.read_csv(r"~/modelFitting/estimates{0}.csv".format(decision_model))
        estimates = estimates[estimates["kraken_present"] == kraken_present]
        ls_subject = estimates["ls"].to_numpy()
        tau_subject = estimates["tau"].to_numpy()
        
        if decision_model.lower() == "cb" or decision_model.lower() == "ucb_lcb_l" or decision_model.lower() == "cb_n":
            beta_subject = estimates["beta"].to_numpy()
    else:
        estimates = pd.read_csv(r"~/modelFitting/{0}.csv".format(estim))
        estimates = estimates[estimates["kraken_present"] == kraken_present]
        ls_subject = estimates["ls"].to_numpy()
        tau_subject = estimates["tau"].to_numpy()
        
        if decision_model.lower() == "cb" or decision_model.lower() == "ucb_lcb_l":
            beta_subject = estimates["beta"].to_numpy()      
   

    # we don't actually know the environments they saw so randomly select some
    rng = np.random.RandomState(123)
    envs = rng.randint(
        0, 30, (N_SUBS_SIM * N_BLOCKS)
    )

    # Simulate and run models
    print('ntrials {0}'.format(N_TRIALS))
    print('nblocks {0}'.format(N_BLOCKS))
    print('nsubs {0}'.format(N_SUBS_SIM))
    

    
    if decision_model.lower() == "cb":
        sim_choices, sim_y = kristinSimUCB(
            ls_subject,
            tau_subject,
            beta_subject,
            N_SUBS_SIM,
            N_BLOCKS,
            N_TRIALS,
            grid_arrays,
            distances,
            envs,
        )
    elif decision_model.lower() == "cb_n":
        sim_choices, sim_y = kristinSimUCB_n(
            ls_subject,
            tau_subject,
            beta_subject,
            N_SUBS_SIM,
            N_BLOCKS,
            N_TRIALS,
            grid_arrays,
            distances,
            envs,
        )

        

    output_dir = "results/parameter_recovery/{0}".format(decision_model.lower())
    if estim != "default":
        output_dir = "results/parameter_recovery/{0}_{1}".format(decision_model.lower(), estim)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save results
    with open(
        os.path.join(output_dir, "parameter_recovery_kraken_{0}.pkl".format(kraken_present)), "wb"
    ) as f:
        pickle.dump(
            {
                "sim_choices_ucb": sim_choices,
                "sim_y_ucb": sim_y

            },
            f,
        )

    # save it as a nice csv
    # get everything back into the shape we have in the original file
sim_choices_flat = np.reshape(sim_choices, (N_SUBS * N_BLOCKS* N_TRIALS))
sim_z_flat = np.reshape(sim_y, (N_SUBS * N_BLOCKS* N_TRIALS))

sim_choices_x, sim_choices_y = np.unravel_index(sim_choices_flat, (GRID_SIZE, GRID_SIZE))


## save as csv so I can work with this in R

data = {'x':sim_choices_x, 'y': sim_choices_y, 'z':sim_z_flat}

df = pd.DataFrame(data)

df.to_csv('~/modelFitting/results/parameter_recovery/{1}/simdatKraken{0}.csv'.format(kraken_present, decision_model.lower()))
