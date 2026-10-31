############### get exceedance probabilities based on log likelihoods using the groupBMC package by sichao
# run from the repo root

import numpy as np
import pandas as pd
#from groupBMC.groupBMC import GroupBMC
from groupBMC2 import GroupBMC

# study: (cross-validated log likelihoods, condition columns, output file)
studies = {
    "Study1": ("Study1/analyses/modelFitting/logp_df.csv", ["kraken_present"], "Study1/analyses/groupBMC_results_incl_cb_n.csv"),
    "Study2": ("Study2/analyses/modelling/logp_df.csv", ["cond", "tp"], "Study2/analyses/groupBMC_results_incl_cb_n.csv"),
    "Study3": ("replication_study/analyses/model_fitting/results/logp_df.csv", ["kraken_present"], "replication_study/analyses/groupBMC_results_incl_cb_n.csv"),
}

# the Study 1 log likelihoods include participants that were excluded from all other analyses
study1_ids = pd.read_csv("Study1/data/master.csv")["ID"].unique()

for study, (path, keys, out) in studies.items():
    logp = pd.read_csv(path)
    if study == "Study1":
        logp = logp[logp["ID"].isin(study1_ids)]

    # a log likelihood of 0 marks a fold without any trials for that participant (same for all models)
    logp = logp[logp["logp"] != 0]

    # log evidence per participant: summed held-out log likelihood across folds
    L_df = logp.pivot_table(index=keys + ["ID"], columns="model", values="logp", aggfunc="sum")
    assert not L_df.isna().any().any(), "a model is missing folds"

    # mean held-out log likelihood per fold, for plotting
    fold_mean = logp.groupby(["model"] + keys + ["ID"])["logp"].mean().reset_index()
    df = fold_mean.groupby(["model"] + keys).agg(logp=("logp", "mean"), se=("logp", "sem"), N=("ID", "nunique")).reset_index()

    results = []
    for cell, L_cell in L_df.groupby(level=keys):
        cell = cell if isinstance(cell, tuple) else (cell,)
        # group BMC needs a K x N matrix: one row per model, one column per participant
        result = GroupBMC(L_cell.to_numpy().T).get_result()
        results.append(pd.DataFrame({"model": L_cell.columns, **dict(zip(keys, cell)),
                                     "exceedance_probability": result.exceedance_probability,
                                     "protected_exceedance_probability": result.protected_exceedance_probability}))

    df = df.merge(pd.concat(results), on=["model"] + keys)
    print(study)
    print(df)

    df.to_csv(out, index=False)
