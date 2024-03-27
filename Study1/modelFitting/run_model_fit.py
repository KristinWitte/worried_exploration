import os
# use SLURM job ID for parallelisation on high performance computing cluster
try:
    SLURMID = int(os.environ["SLURM_ARRAY_TASK_ID"])
except:
    SLURMID = 1

# select which GPU to use
os.environ["CUDA_VISIBLE_DEVICES"] = str(SLURMID % 4)

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

    runID = (SLURMID % 2)+1
    kraken_present = (SLURMID % 3) % 2# task condition: 0 = safe, 1 = risky
    decision_model = parser.parse_args().decision_model
    n_samples = parser.parse_args().n_samples
    n_warmup = parser.parse_args().n_warmup
    dat = parser.parse_args().dat
    

    print('Run ID = {0}'.format(runID))
    print('Decision model = {0}'.format(decision_model))
    print('Kraken present = {0}'.format(kraken_present))
    print('Number of samples = {0}'.format(n_samples))
    print('Number of warmup samples = {0}'.format(n_warmup))

    # Load data
    # cluster directory:
    data = pd.read_csv(
        r"~/modelFitting/{0}.csv".format(dat)
    )
   
    # select only one of the two conditions
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

    # if model is ucb_lcb_n then create novelty bonus

    if decision_model.lower() == "ucb_lcb_n":
        novelty = np.zeros((N_SUBS * N_BLOCKS, N_TRIALS, GRID_SIZE**2))
        novelty += 1
        for i in range(N_SUBS * N_BLOCKS):
            for j in range(1,N_TRIALS):
                novelty[i,j,:] = novelty[i,j-1,:]# copy novelty from previous trial (bc what was not novel on previous trial is not novel on this trial)
                novelty[i,j,choices_flat[i,j-1]] = 0 # obs from this trial is no longer novel

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


    # Simulate and run models
    if decision_model.lower() == "ucb_lcb":

        model_output = sample(
            grid_model_UCB_LCB,
            choices_flat,
            choices_flat,
            y_flat,
            missing,
            distances,
            N_SUBS,
            N_BLOCKS,
            N_TRIALS,
            n_samples=n_samples,
            n_warmup=n_warmup,
            seed=runID,
        )

    elif decision_model.lower() == "ucb_lcb_n":# novelty bonus model

        model_output = sample_n(
            grid_model_UCB_LCB_n,
            choices_flat,
            choices_flat,
            y_flat,
            missing,
            distances,
            novelty,
            N_SUBS,
            N_BLOCKS,
            N_TRIALS,
            n_samples=n_samples,
            n_warmup=n_warmup,
            seed=runID,
        )

    elif decision_model.lower() == "ucb_b0":

        model_output = sample(
            grid_model_UCB_b0,
            choices_flat,
            choices_flat,
            y_flat,
            missing,
            distances,
            N_SUBS,
            N_BLOCKS,
            N_TRIALS,
            n_samples=n_samples,
            n_warmup=n_warmup,
            seed=runID,
        )

    elif decision_model.lower() == "ucb_l0":

        model_output = sample(
            grid_model_UCB_l0,
            choices_flat,
            choices_flat,
            y_flat,
            missing,
            distances,
            N_SUBS,
            N_BLOCKS,
            N_TRIALS,
            n_samples=n_samples,
            n_warmup=n_warmup,
            seed=runID,
        )

    elif decision_model.lower() == "pos":

        model_output = sample(
            grid_model_POS,
            choices_flat,
            choices_flat,
            y_flat,
            missing,
            distances,
            N_SUBS,
            N_BLOCKS,
            N_TRIALS,
            n_samples=n_samples,
            n_warmup=n_warmup,
            seed=runID,
        )

    posterior_samples = model_output.get_samples()

    # Save results
    output_dir = "results/model_fitting/{0}_{1}".format(decision_model.lower(), dat)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(
        os.path.join(output_dir, "model_fit__kraken-{0}_run-{1}.pkl".format(kraken_present, runID)), "wb"
    ) as f:
        pickle.dump(
            {
                "posterior_samples": posterior_samples,
                "choices": choices_flat,
                "y": y_flat
            },
            f,
        )

    # Convert to ArViz
    az_data = az.from_numpyro(model_output)

    # Save ArViz NetCDF
    print('Saving to NetCDF, location = {0}'.format(os.path.join(output_dir, "model_fit__kraken-{0}_run-{1}.nc".format(kraken_present, runID))))
    az_data.to_netcdf(os.path.join(output_dir, "model_fit__kraken-{0}_run-{1}.nc".format(kraken_present, runID)))


    