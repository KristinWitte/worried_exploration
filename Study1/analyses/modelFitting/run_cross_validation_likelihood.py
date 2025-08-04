"""
This script calculates log likelihoods on held out data for each model
"""

from decision_strategies import *
from gp_model_functions import *
from model_fit import *
import jax
import jax.numpy as jnp
from jax import random
import numpy as np
import pandas as pd
import arviz as az
import os
import pickle
import argparse
from numpyro.infer import Predictive
#from sklearn.metrics import accuracy_score

if __name__ == "__main__":

    rng_key_count = 0

    logp_df = {
        "model": [],
        "ID": [],
        "kraken_present": [],
        "cv_fold": [],
        "logp": [],
    }

    for kraken_present in [0, 1]:

        for cv_fold in range(5):

            print("CV fold = {0}".format(cv_fold))
            print("Kraken present = {0}".format(kraken_present))

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
                    dict(
                        zip(
                            data.loc[data["ID"] == sub, "blocknr"].unique(), range(1, 7)
                        )
                    )
                )

            # Select data for CV fitting
            fit_data = data[data["blocknr"] != cv_fold + 1]
            test_data = data[data["blocknr"] == cv_fold + 1]
            
            # get the subs that are present in fit_data AND test_data
            subs = set(fit_data["ID"].unique()) & set(test_data["ID"].unique())

            # Get important numbers
            N_SUBS = len(subs)
            N_BLOCKS = len(fit_data["blocknr"].unique())
            N_TRIALS = len(fit_data["click"].unique())

            # Initialise arrays
            choices_fit = np.zeros((N_SUBS, N_BLOCKS, N_TRIALS))
            y_fit = np.zeros((N_SUBS, N_BLOCKS, N_TRIALS))
            choices_test = np.zeros((N_SUBS, 1, N_TRIALS))
            y_test = np.zeros((N_SUBS, 1, N_TRIALS))

            GRID_SIZE = 11

            # Extract and transform data
            for n_sub, sub in enumerate(subs):

                # Fit data
                sub_df = fit_data[fit_data["ID"] == sub]
                for n_block, block in enumerate(sub_df["blocknr"].unique()):
                    sub_block_df = sub_df[sub_df["blocknr"] == block]
                    sub_block_df = sub_block_df.fillna(
                        {"x": 0, "y": 0}
                    )  # Set missing X and Y to zero
                    choices_fit[n_sub, n_block, :] = np.ravel_multi_index(
                        sub_block_df[["x", "y"]].values.T.astype(int),
                        (GRID_SIZE, GRID_SIZE),
                    )
                    y_fit[n_sub, n_block, :] = (sub_block_df["z"] - 50) / 100

                # Test data
                sub_df = test_data[test_data["ID"] == sub]
                sub_df = sub_df.fillna(
                        {"x": 0, "y": 0}
                    )  # Set missing X and Y to zero
                choices_test[n_sub, 0, :] = np.ravel_multi_index(
                    sub_df[["x", "y"]].values.T.astype(int),
                    (GRID_SIZE, GRID_SIZE),
                )
                y_test[n_sub, 0, :] = (sub_df["z"] - 50) / 100

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

             # create novelty bonus
            novelty = np.zeros((N_SUBS, N_TRIALS, GRID_SIZE**2))
            novelty += 1
            for i in range(N_SUBS):
                for j in range(1,N_TRIALS):
                    novelty[i,j,:] = novelty[i,j-1,:]# copy novelty from previous trial (bc what was not novel on previous trial is not novel on this trial)
                    novelty[i,j,choices_test_flat[i,j-1]] = 0 # obs from this trial is no longer novel


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

            #for decision_model in ["ucb_lcb", "ucb_b0", "ucb_l0", "pos", "post", "poi", "ucb_lcb_l", "ucb_b0_l", "pos_l"]:
            for decision_model in ["ucb_lcb_n", "ucb_lcb", "ucb_b0", "pos"]:

                print("Decision model = {0}".format(decision_model))

                az_data = []
                for runID in [0, 1]:
                     # I messed up again so sometimes runID = [1,2] and sometimes [0,1]
                    az_data.append(
                        az.from_netcdf(
                            "results/model_fitting_CV/{3}/model_fit__kraken-{0}_fold-{1}_run-{2}.nc".format(
                                kraken_present, cv_fold, runID, decision_model
                            )
                        )
                    )
                    
                        
                dataset = az.concat(*az_data, dim="draw")

                posterior_samples = {}

                for k in dataset.posterior.keys():
                    posterior_samples[k] = dataset.posterior[k].to_numpy().squeeze()
                    
                
                # get rid of estimates from subjects that are NOT in the subs
                incl = np.isin(fit_data["ID"].unique(), list(subs))


                # Simulate and run models
                if decision_model.lower() in ["ucb", "lcb", "ucb_lcb"]:

                    p = model_func_UCB_jit(
                        jnp.repeat(
                            posterior_samples["ls_subject_transformed"][:,incl].mean(axis=0), 1
                        ),
                        jnp.repeat(
                            posterior_samples["tau_subject_transformed"][:,incl].mean(axis=0), 1
                        ),
                        jnp.repeat(
                            posterior_samples["beta_subject_transformed"][:,incl].mean(axis=0),
                            1,
                        ),
                        0,
                        choices_test_flat,
                        y_test_flat,
                        distances,
                        N_TRIALS,
                    ).squeeze()

                elif decision_model.lower() == "ucb_lcb_n":

                    p = model_func_UCB_n_jit(
                        jnp.repeat(
                            posterior_samples["ls_subject_transformed"][:,incl].mean(axis=0), 1
                        ),
                        jnp.repeat(
                            posterior_samples["tau_subject_transformed"][:,incl].mean(axis=0), 1
                        ),
                        jnp.repeat(
                            posterior_samples["beta_subject_transformed"][:,incl].mean(axis=0),
                            1,
                        ),
                        0,
                        choices_test_flat,
                        y_test_flat,
                        distances,
                        novelty,
                        N_TRIALS,
                    ).squeeze()

                elif decision_model.lower() == "ucb_b0":

                    p = model_func_UCB_jit(
                        jnp.repeat(
                            posterior_samples["ls_subject_transformed"][:,incl].mean(axis=0), 1
                        ),
                        jnp.repeat(
                            posterior_samples["tau_subject_transformed"][:,incl].mean(axis=0), 1
                        ),
                        jnp.zeros_like(
                            jnp.repeat(
                                posterior_samples["tau_subject_transformed"][:,incl].mean(
                                    axis=0
                                ),
                                1,
                            )
                        ),
                        0,
                        choices_test_flat,
                        y_test_flat,
                        distances,
                        N_TRIALS,
                    ).squeeze()

                elif decision_model.lower() == "ucb_l0":

                    p = model_func_UCB_jit(
                        jnp.zeros_like(
                            jnp.repeat(
                                posterior_samples["tau_subject_transformed"][:,incl].mean(
                                    axis=0
                                ),
                                1,
                            )
                        ) + 0.000001,
                        jnp.repeat(
                            posterior_samples["tau_subject_transformed"][:,incl].mean(axis=0), 1
                        ),
                        jnp.repeat(
                            posterior_samples["beta_subject_transformed"][:,incl].mean(axis=0),
                            1,
                        ),
                        0,
                        choices_test_flat,
                        y_test_flat,
                        distances,
                        N_TRIALS,
                    ).squeeze()

                elif decision_model.lower() == "pos":

                    p = model_func_POS_jit(
                        jnp.repeat(
                            posterior_samples["ls_subject_transformed"][:,incl].mean(axis=0), 1
                        ),
                        jnp.repeat(
                            posterior_samples["tau_subject_transformed"][:,incl].mean(axis=0), 1
                        ),
                        jnp.zeros_like(
                            jnp.repeat(
                                posterior_samples["tau_subject_transformed"][:,incl].mean(
                                    axis=0
                                ),
                                1,
                            )
                        ),
                        0,
                        choices_test_flat,
                        y_test_flat,
                        distances,
                        N_TRIALS,
                    ).squeeze()

                rng_key = random.PRNGKey(rng_key_count)
                rng_key_count += 1

                logp = (
                    dist.Categorical(
                        p.squeeze().transpose((1, 0, 2))[~missing_test[:, 1:]]
                    )
                    .log_prob(choices_test_flat[:, 1:][~missing_test[:, 1:]])
                    
                )
                
                # in risky condition the length of logp is super weird bc of different trial lengths
                # can't just reshape it
                
                temp = np.zeros((len(subs), 10))
                temp = np.reshape(temp, len(subs)*10)
                missing_test_flat = np.reshape(missing_test[:, 1:], len(subs)*10)
                temp[~missing_test_flat] = logp
                print(temp.shape)
                print(missing_test_flat)
                
                logp = temp
                
                logp = np.reshape(logp, (len(subs), 10))
                print(logp.shape)
                logp = logp.sum(1)
                print(logp.shape)
                print(logp)
                
                logp_df["model"] = np.concatenate([logp_df["model"], np.repeat(decision_model, len(subs))])
                logp_df["ID"] = np.concatenate([logp_df["ID"], np.array(list(subs))])
                logp_df["cv_fold"] = np.concatenate([logp_df["cv_fold"], np.repeat(cv_fold, len(subs))])
                logp_df["kraken_present"] = np.concatenate([logp_df["kraken_present"], np.repeat(kraken_present, len(subs))])
                logp_df["logp"] = np.concatenate([logp_df["logp"], logp])


                for k, v in logp_df.items():
                    print(k, len(v))


            # Random model
            print("random")
            p = np.ones_like(p) / p.shape[-1]

            rng_key = random.PRNGKey(rng_key_count)
            rng_key_count += 1

            logp = (
                dist.Categorical(
                    p.squeeze().transpose((1, 0, 2))[~missing_test[:, 1:]]
                )
                .log_prob(choices_test_flat[:, 1:][~missing_test[:, 1:]])
               
            )

            
            temp = np.zeros((len(subs), 10))
            temp = np.reshape(temp, len(subs)*10)
            missing_test_flat = np.reshape(missing_test[:, 1:], len(subs)*10)
            temp[~missing_test_flat] = logp
            print(temp.shape)
            print(missing_test_flat)
            
            logp = temp


            logp = np.reshape(logp, (len(subs), 10))
            print(logp.shape)
            logp = logp.sum(1)
            print(logp.shape)
            print(logp)
               
            logp_df["model"] = np.concatenate([logp_df["model"], np.repeat("random", len(subs))])
            logp_df["ID"] = np.concatenate([logp_df["ID"], np.array(list(subs))])
            logp_df["cv_fold"] = np.concatenate([logp_df["cv_fold"], np.repeat(cv_fold, len(subs))])
            logp_df["kraken_present"] = np.concatenate([logp_df["kraken_present"], np.repeat(kraken_present, len(subs))])
            logp_df["logp"] = np.concatenate([logp_df["logp"], logp])



            for k, v in logp_df.items():
                print(k, len(v))


    logp_df = pd.DataFrame(logp_df)
    logp_df.to_csv("results/model_fitting_CV/logp_df.csv", index=False)
