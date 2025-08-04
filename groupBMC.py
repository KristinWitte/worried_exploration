############### get exceedance probabilities based on log likelihoods using the groupBMC package by sichao

import numpy as np
import pandas as pd
#from groupBMC.groupBMC import GroupBMC
from groupBMC2 import GroupBMC

## load log likelihoods
logp = pd.read_csv('model_fitting/results/logp_df.csv')

logp['logp'].replace(0, float('nan'), inplace=True)

# First ddply
df = logp.groupby(['model', 'kraken_present', 'ID']).agg(se=('logp', 'std'), logp=('logp', 'mean')).reset_index()

# Second ddply
df = df.groupby(['model', 'kraken_present']).agg(se=('se', 'std'), logp=('logp', 'mean')).reset_index()




# Filter out unwanted 'model' values
unwanted_models = ["pos_l", "ucb_b0_l", "ucb_lcb_l"]
#df = df[~df['model'].isin(unwanted_models)]

df_safe = df[df['kraken_present'] == 0]
df_risky = df[df['kraken_present'] == 1]

L = np.array(df_safe['logp'])

L = L.reshape(4,1)

result = GroupBMC(L).get_result()
print(result)

df_safe['exceedance_probability'] = result.exceedance_probability

L = np.array(df_risky['logp'])

L = L.reshape(4,1)

result = GroupBMC(L).get_result()

df_risky['exceedance_probability'] = result.exceedance_probability

print(df_safe)
print(df_risky)

# concatinate df_safe and df_risky into one dataframe
df = pd.concat([df_safe, df_risky], ignore_index=True)

# save df as csv
df.to_csv('groupBMC_results_incl_cb_n.csv', index=False)

