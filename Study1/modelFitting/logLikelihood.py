# -*- coding: utf-8 -*-
"""
Created on Thu Jul  7 17:33:27 2022

@author: krist
"""

from decision_strategies import *
from gp_model_functions import *
from model_fit import *
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import arviz as az
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
        "kraken_present",# condition (safe vs risky, kraken being present = risky round)
        type=int,
        default=0,
        help='Set to 0 to select blocks without the kraken, 1 to select blocks with the kraken, 2 selects both'
    )
    parser.add_argument(
        "--dat",
        type=str,
        default="master",
        help='name of the data that should be fit'
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
    n_samples = parser.parse_args().n_samples
    n_warmup = parser.parse_args().n_warmup
    kraken_present = parser.parse_args().kraken_present
    #runID = parser.parse_args().runID
    dat = parser.parse_args().dat
    

    #print('Run ID = {0}'.format(runID))
    print('Decision model = {0}'.format(decision_model))
    print('Kraken present = {0}'.format(kraken_present))
    print('Number of samples = {0}'.format(n_samples))
    print('Number of warmup samples = {0}'.format(n_warmup))
    
    
    logp_df = {
        "model": [],
        "kraken_present": [],
        "logp": [],
    }
   
    data = pd.read_csv(
        r"~/modelFitting/{0}.csv".format(dat)
    )
    if kraken_present < 2:# if we only want to fit one of the two conditions
    
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
    
    # missing blocks are currently set to zeros, change this to be NA
    # luckily only the missing blocks start with a 0 (for a different dataset, this needs to be verified again)
    y_flat[y_flat[:,0]==0,:] = np.nan

    # Deal with missing data
    missing = np.isnan(y_flat)
    y_flat[np.isnan(y_flat)] = 0

    # Convert to JAX arrays (allows for faster computation on the GPU)
    choices_flat = jax.device_put(choices_flat.astype(int))
    y_flat = jax.device_put(y_flat)
    # missing = jax.device_put(missing)

    # One dimensional column vectors of inputs
    x1 = np.arange(GRID_SIZE)[:, None]
    x2 = np.arange(GRID_SIZE)[:, None]

    # Make cartesian grid
    X = jnp.dstack(np.meshgrid(x1.squeeze(), x2.squeeze())).reshape(-1, 2)

    # Get distances
    distances = jnp.sqrt(square_dist(X, 1))
    
    # get what condition is when (if fitting both conditions together)
    # cond needs to have length n_sub*n_block so I need to fill in some stuff for missing blocks
    cond = np.zeros(N_SUBS*N_BLOCKS)
    cond[~missing[:,0]] = data.loc[data["click"] == 1, "krakenPres"]
    
    az_data = []
    for runID in [1,2]:
        try: # I messed up again so now runID sometimes is [1,2] and sometimes [0,1]
            az_data.append(
                az.from_netcdf(
                    "results/model_fitting/{2}_master/model_fit__kraken-{0}_run-{1}.nc".format(
                        kraken_present, runID, decision_model
                    )
                )
            )
        except:
            az_data.append(
                az.from_netcdf(
                    "results/model_fitting/{2}_master/model_fit__kraken-{0}_run-{1}.nc".format(
                        kraken_present, 0, decision_model
                    )
                )
            )  

    dataset = az.concat(*az_data, dim="draw")

    posterior_samples = {}
    
    for k in dataset.posterior.keys():
        posterior_samples[k] = dataset.posterior[k].to_numpy().squeeze()
        
        
    if decision_model.lower() == "ucb_b0":

        p = model_func_UCB_jit(
            jnp.repeat(
                posterior_samples["ls_subject_transformed"].mean(axis=0), N_BLOCKS
            ),
            jnp.repeat(
                posterior_samples["tau_subject_transformed"].mean(axis=0), N_BLOCKS
            ),
            jnp.zeros_like(
                jnp.repeat(
                    posterior_samples["tau_subject_transformed"].mean(
                        axis=0
                    ),
                    N_BLOCKS,
                )
            ),
            0,
            choices_flat,
            y_flat,
            distances,
            N_TRIALS,
        ).squeeze()
        
    elif decision_model.lower() == "ucb_b0_2":
        # create composite parameters again
        ls =jnp.repeat(posterior_samples["ls_subject_transformed0"].mean(axis=0), N_BLOCKS)
    # the excluded blocks are all in the safe condition so everyone has 5 risky blocks
        ls = ls.at[cond == 1].set(jnp.repeat(posterior_samples["ls_subject_transformed1"].mean(axis=0), 5))
    
        tau = jnp.repeat(posterior_samples["tau_subject_transformed0"].mean(axis=0), N_BLOCKS)
        tau = tau.at[cond == 1].set(jnp.repeat(posterior_samples["tau_subject_transformed1"].mean(axis=0), 5))
    
        beta = jnp.zeros(N_SUBS * N_BLOCKS)
        
        p = model_func_UCB_jit(
        ls,
        tau,
        beta,
        0,
            choices_flat,
            y_flat,
            distances,
            N_TRIALS,
        ).squeeze()
        
    logp = (dist.Categorical(
        p.squeeze().transpose((1, 0, 2))[~missing[:, 1:]]
        )
        .log_prob(choices_flat[:, 1:][~missing[:, 1:]])
        .sum()
        )

    logp_df["model"].append(decision_model)
    logp_df["kraken_present"].append(kraken_present)
    logp_df["logp"].append(logp)
        
    logp_df = pd.DataFrame(logp_df)
    logp_df.to_csv("results/model_fitting/logp_df_{0}_{1}.csv".format(decision_model.lower(), kraken_present), index=False)