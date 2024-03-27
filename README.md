# Repository for the paper: Exploring the Unexplored: Worry as a Catalyst for Exploratory Behavior in Anxiety and Depression by Kristin Witte, Toby Wise, Quentin Huys*, Eric Schulz*


## General structure

Files that are not in any folder are relevant for both studies.

### Plots
contains all plots used in the paper.

### Study 1
Contains all analysis and task code from Study 1. Computational modelling and task code are in their separate sub-folders.

### Study 2
Contains all analysis and task code from Study 2. Computational modelling and task code are in their separate sub-folders.

 ## Specific files

### Study 1
#### master.Rda /master.csv
csv and R data versions of the main task data. 11 blocks of 10 clicks per participant, "krakenPres" indicates the condition in that round (0 = safe, 1 = risky), "krakenCaught" indicates whether that participant in that round clicked a square that made them loose all rewards on that round. "Distance" indicates the euclidian distance to the previous square. x and y indicate the coordinates of the square clicked on that trial. z indicates the reward obtained.

#### estimatesCB_n.csv
Parameter estimates obtained from the novelty bonus parameter. Beta indicates the parameter value for the eta (novelty bonus) parameter. kraken_present indicates the conditon (0 = safe, 1 = risky). Ls indicates the generalisation parameter lambda, tau the softmax temperature.

#### groupBMC_results_incl_cb_n.csv
Output from the estimation of the exceedance probabilities. The exceedance probabilities were estimated as described here: https://github.com/cpilab/group-bayesian-model-comparison

#### lsQs.Rda/etaQs.Rda/tauQs.Rda/NUOQs.Rda
Output from bayesian mixed effects regressions predicting length scale parameter lambda/ novelty seeking parameter eta / softmax temperature tau / probability of selecting a novel option, respectively, from the questionnaires.

### Study 2
#### estimatesCB_n.csv
Parameter estimates obtained from the novelty bonus parameter. Beta indicates the parameter value for the eta (novelty bonus) parameter. cond indicates the conditon that participant was assigned to (0 = control, 1 = intervention). tp indicates the timepoint (0 = before intervention, 1 = after intervention). Ls indicates the generalisation parameter lambda, tau the softmax temperature.

#### groupBMC_results_incl_cb_n.csv
Output from the estimation of the exceedance probabilities. The exceedance probabilities were estimated as described here: https://github.com/cpilab/group-bayesian-model-comparison

#### Master.Rda /Master.csv
csv and R data versions of the main task data. 11 blocks of 25 clicks (trials) per participant, "cond" indicates the condition that participant was assigned to (0 = control, 1 = intervention), "tp" indicates whether this round was before the intervention (0) or after (1).  "krakenFound" indicates whether that participant in that round clicked a square that made them loose all rewards on that round. "unique" indicates whether the square clicked on that trial had been clicked before within that round. x and y indicate the coordinates of the square clicked on that trial. z indicates the reward obtained. "time" indicates the time in ms that that participant took for that block. "env" indicates which of the reward environments (underlying reward grids) the participant had seen on that block. All reward environments can be found in Study2/task/assets/sample_grid.json and in Study2/modelling/sample_grid.json.

#### nervous.Rda /nervous.csv

csv and R data versions of the self-reported nervousness ratings. "nervous" indicates the ratings ranging from 0 (not nervous at all) to 100 (extremely nervous). Participants gave these ratings after 3 of the rounds before the intervention and after 3 of the rounds after the intervention. 



 
