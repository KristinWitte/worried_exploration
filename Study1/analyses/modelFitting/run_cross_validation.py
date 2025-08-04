## this script only fits the models to the data for each of the folds
# the likelihood is calculated in run_cross_validation_likelihood.py

import os
# Use SLURM job ID (for parallelisation on high performance computing cluster)
try:
    runID = int(os.environ["SLURM_ARRAY_TASK_ID"]) - 1
except:
    runID = 1

os.environ["CUDA_VISIBLE_DEVICES"] = str(runID % 4)


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
from numpyro.infer import log_likelihood

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
        "kraken_present",
        type=int,
        default=0,
        help="Set to 0 to select blocks without the kraken, 1 to select blocks with the kraken",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=2000,
        help="Number of samples to use",
    )
    parser.add_argument(
        "--n_warmup",
        type=int,
        default=1000,
        help="Number of warmup samples to use",
    )

    decision_model = parser.parse_args().decision_model
    n_samples = parser.parse_args().n_samples
    n_warmup = parser.parse_args().n_warmup
    kraken_present = parser.parse_args().kraken_present

    # Get CV fold and run
    cv_fold = runID % 5
    runID = runID % 2

    print("Run ID = {0}".format(runID))
    print("CV fold = {0}".format(cv_fold))
    print("Decision model = {0}".format(decision_model))
    print("Kraken present = {0}".format(kraken_present))
    print("Number of samples = {0}".format(n_samples))
    print("Number of warmup samples = {0}".format(n_warmup))

    # Load data
    data = pd.read_csv(
        r"~/modelFitting/master.csv"
    )
    data = data[data["krakenPres"] == kraken_present]

    data = data[data["blocknr"] != 6]

    # Adjust block numbers
    data.loc[data["blocknr"] > 6, "blocknr"] -= 1

    # Ensure each subject's blocks are numbered 1 to 6
    for sub in data["ID"].unique():
        data.loc[data["ID"] == sub, "blocknr"] = data.loc[
            data["ID"] == sub, "blocknr"
        ].replace(
            dict(zip(data.loc[data["ID"] == sub, "blocknr"].unique(), range(1, 7)))
        )

    # Select data for CV fitting
    fit_data = data[data["blocknr"] != cv_fold + 1]
    test_data = data[data["blocknr"] == cv_fold + 1]

    # Get important numbers
    N_SUBS = len(fit_data["ID"].unique())
    N_BLOCKS = len(fit_data["blocknr"].unique())
    N_TRIALS = len(fit_data["click"].unique())

    # Initialise arrays
    choices_fit = np.zeros((N_SUBS, N_BLOCKS, N_TRIALS))
    y_fit = np.zeros((N_SUBS, N_BLOCKS, N_TRIALS))
    choices_test = np.zeros((N_SUBS, 1, N_TRIALS))
    y_test = np.zeros((N_SUBS, 1, N_TRIALS))

    GRID_SIZE = 11

    # Extract and transform data
    for n_sub, sub in enumerate(fit_data["ID"].unique()):

        # Fit data
        sub_df = fit_data[fit_data["ID"] == sub]
        for n_block, block in enumerate(sub_df["blocknr"].unique()):
            sub_block_df = sub_df[sub_df["blocknr"] == block]
            sub_block_df = sub_block_df.fillna(
                {"x": 0, "y": 0}
            )  # Set missing X and Y to zero
            choices_fit[n_sub, n_block, :] = np.ravel_multi_index(
                sub_block_df[["x", "y"]].values.T.astype(int), (GRID_SIZE, GRID_SIZE)
            )
            y_fit[n_sub, n_block, :] = (sub_block_df["z"] - 50) / 100

        # Test data
        sub_df = test_data[test_data["ID"] == sub]
        choices_test[n_sub, 0, :] = np.ravel_multi_index(
            sub_block_df[["x", "y"]].values.T.astype(int), (GRID_SIZE, GRID_SIZE)
        )
        y_test[n_sub, 0, :] = (sub_block_df["z"] - 50) / 100

    choices_fit = choices_fit.astype(int)
    choices_test = choices_test.astype(int)

    # Reshape data
    choices_fit_flat = np.reshape(choices_fit, (N_SUBS * N_BLOCKS, N_TRIALS))
    y_fit_flat = np.reshape(y_fit, (N_SUBS * N_BLOCKS, N_TRIALS))
    choices_test_flat = np.reshape(choices_test, (N_SUBS, N_TRIALS))
    y_test_flat = np.reshape(y_test, (N_SUBS, N_TRIALS))

    # Deal with missing data
    missing_fit = np.isnan(y_fit_flat)
    y_fit_flat[np.isnan(y_fit_flat)] = 0
    missing_test = np.isnan(y_test_flat)
    y_test_flat[np.isnan(y_test_flat)] = 0

    # if model is ucb_lcb_n then create novelty bonus

    if decision_model.lower() == "ucb_lcb_n":
        novelty = np.zeros((N_SUBS * N_BLOCKS, N_TRIALS, GRID_SIZE**2))
        novelty += 1
        for i in range(N_SUBS * N_BLOCKS):
            for j in range(1,N_TRIALS):
                novelty[i,j,:] = novelty[i,j-1,:]# copy novelty from previous trial (bc what was not novel on previous trial is not novel on this trial)
                novelty[i,j,choices_fit_flat[i,j-1]] = 0 # obs from this trial is no longer novel

    # Convert to JAX arrays (allows for faster computation on the GPU)
    choices_fit_flat = jax.device_put(choices_fit_flat.astype(int))
    y_fit_flat = jax.device_put(y_fit_flat)
    choices_test_flat = jax.device_put(choices_test_flat.astype(int))
    y_test_flat = jax.device_put(y_test_flat)

    # One dimensional column vectors of inputs
    x1 = np.arange(GRID_SIZE)[:, None]
    x2 = np.arange(GRID_SIZE)[:, None]

    # Make cartesian grid
    X = jnp.dstack(np.meshgrid(x1.squeeze(), x2.squeeze())).reshape(-1, 2)

    # Get distances
    distances = jnp.sqrt(square_dist(X, 1))

    # Simulate and run models
    if decision_model.lower() == "ucb_lcb":

        model_output = sample(
            grid_model_UCB_LCB,
            choices_fit_flat,
            choices_fit_flat,
            y_fit_flat,
            missing_fit,
            distances,
            N_SUBS,
            N_BLOCKS,
            N_TRIALS,
            n_samples=n_samples,
            n_warmup=n_warmup,
            seed=runID
        )

        likelihood_model = grid_model_UCB_LCB

    elif decision_model.lower() == "ucb_lcb_n":# novelty bonus model

        model_output = sample_n(
            grid_model_UCB_LCB_n,
            choices_fit_flat,
            choices_fit_flat,
            y_fit_flat,
            missing_fit,
            distances,
            novelty,
            N_SUBS,
            N_BLOCKS,
            N_TRIALS,
            n_samples=n_samples,
            n_warmup=n_warmup,
            seed=runID,
        )

        likelihood_model = grid_model_UCB_LCB_n

    elif decision_model.lower() == "ucb_b0":

        model_output = sample(
            grid_model_UCB_b0,
            choices_fit_flat,
            choices_fit_flat,
            y_fit_flat,
            missing_fit,
            distances,
            N_SUBS,
            N_BLOCKS,
            N_TRIALS,
            n_samples=n_samples,
            n_warmup=n_warmup,
            seed=runID,
        )

        likelihood_model = grid_model_UCB_b0

    elif decision_model.lower() == "ucb_l0":

        model_output = sample(
            grid_model_UCB_l0,
            choices_fit_flat,
            choices_fit_flat,
            y_fit_flat,
            missing_fit,
            distances,
            N_SUBS,
            N_BLOCKS,
            N_TRIALS,
            n_samples=n_samples,
            n_warmup=n_warmup,
            seed=runID,
        )

        likelihood_model = grid_model_UCB_l0

    elif decision_model.lower() == "pos":

        model_output = sample(
            grid_model_POS,
            choices_fit_flat,
            choices_fit_flat,
            y_fit_flat,
            missing_fit,
            distances,
            N_SUBS,
            N_BLOCKS,
            N_TRIALS,
            n_samples=n_samples,
            n_warmup=n_warmup,
            seed=runID,
        )

        likelihood_model = grid_model_POS

    posterior_samples = model_output.get_samples()

    # Save results
    output_dir = "results/model_fitting_CV/{0}".format(decision_model.lower())

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Convert to ArViz
    az_data = az.from_numpyro(model_output)

    # Save ArViz NetCDF
    print(
        "Saving to NetCDF, location = {0}".format(
            os.path.join(
                output_dir,
                "model_fit__kraken-{0}_fold-{1}_run-{2}.nc".format(
                    kraken_present, cv_fold, runID
                ),
            )
        )
    )

    az_data.to_netcdf(
        os.path.join(
            output_dir,
            "model_fit__kraken-{0}_fold-{1}_run-{2}.nc".format(
                kraken_present, cv_fold, runID
            ),
        )
    )
