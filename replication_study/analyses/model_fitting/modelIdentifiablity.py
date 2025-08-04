import numpy as np
import arviz as az
import pandas as pd
import pickle

# load likelihoods and get waics

waics = {'data_model': [],
         'fit_model': [],
        'kraken': [],
        'waic': [],
        'se': []
        }

posteriors = {}


for data_model in ["CB", "CB_b0", "CB_n"]:
    posteriors[data_model] = {}
    for fit_model in ["ucb_lcb", "ucb_b0", "ucb_lcb_n"]:

       posteriors[data_model][fit_model] = {} 

       for kraken in [0,1]:
            outputs = []

            # Iterate over chains and append them to a list
            for runID in [1,2]:
                #outputs.append(az.from_netcdf("/Users/kwitte/Library/CloudStorage/OneDrive-Personal/CPI/safeExploration/dataAnalysis/modelFitting/results/model_fitting/{2}_master/model_fit__kraken-{0}_run-{1}.nc".format(kraken, runID, model)))
                outputs.append(az.from_netcdf("analysis/model_fitting/results/model_fitting/{2}_simdat{3}/model_fit__kraken-{0}_run-{1}.nc".format(kraken, runID, fit_model, data_model)))
            # Combine the data for the two chains
            dataset = az.concat(*outputs, dim='chain')

            waic = az.waic(dataset)# there were warnings here but doing a full cross validation takes forever

            
            # Add necessary data to the WAIC dictionary - this is in effect adding a row to what will become a dataframe
            waics['fit_model'].append(fit_model)
            waics['data_model'].append(data_model)
            waics['kraken'].append(kraken)
            waics['waic'].append(waic.elpd_waic)
            waics['se'].append(waic.se)

            # Store the posterior
            posteriors[data_model][fit_model][kraken] = dataset


print(waics)

waics_df = pd.DataFrame(waics)
print(waics_df)

# add column called best which is 1 if the model is the best fit for that given data_model and kraken value and 0 otherwise
waics_df['best'] = 0
waics_df.loc[waics_df.groupby(['data_model', 'kraken'])['waic'].idxmax(), 'best'] = 1
waics_df

# save waics as csv

waics_df.to_csv('analysis/model_fitting/results/identifiability_waics.csv', index=False)
