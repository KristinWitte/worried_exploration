####### distance between subsequent clicks as a function of factor scores #########

library(brms)
library(tidyverse)

###### importing and formatting data #########
# run from the repo root
version <- commandArgs(TRUE)[2]
factors <- commandArgs(TRUE)[3]# we tested both 3 and 4 factors, 4 factors was a better fit but the regression results did not differ
load(paste0("replication_study/data/Master_",version,".Rda"))

factor_scores <- read.csv(paste0("replication_study/data/fa_scores", factors, "_",version, ".csv"))

# simple coding instead of dummy coding (recoding the condition (krakenPres) from 0 and 1 (safe and risky) to -0.5, 0.5)
Master$krakenPresent <- Master$krakenPresent - 0.5
unique(Master$krakenPresent)

# euclidean distance between the square clicked on this trial and the square clicked on the previous trial of the same round
# (na on the first click of a round and on trials where the participant had already lost all rewards and moved on to the next round)
prev <- match(paste(Master$ID, Master$block, Master$trial - 1), paste(Master$ID, Master$block, Master$trial))
Master$dist <- sqrt((Master$x - Master$x[prev])^2 + (Master$y - Master$y[prev])^2)

# to take the reward on the current click into account we use the distance to the square clicked on the next trial
nxt <- match(paste(Master$ID, Master$block, Master$trial + 1), paste(Master$ID, Master$block, Master$trial))
Master$distFROMz <- Master$dist[nxt]

# mean-center the dependent variable
Master$distFROMz <- scale(Master$distFROMz, center = T, scale = F)

# add in factor scores

Master <- Master %>% left_join(factor_scores, by = "ID")

# z scale all numeric predictors first

Master[,c(2,3,6,11:19, 28:ncol(Master))] <- sapply(Master[,c(2,3,6,11:19, 28:ncol(Master))], function(df) scale(as.numeric(df), center = T, scale = T))
head(Master)


############# actual analyses #####################


# create a directory to save the results if it doesn't exist yet

if (!file.exists("replication_study/analyses/distance")){

  dir.create(file.path("replication_study/analyses/distance"))}


## get SLURM array ID
# this script is optimised to work on a high performance cluster such that
# all regressions run in parallel

task_id <- as.numeric(commandArgs(TRUE)[1])

formula <- as.formula(paste0("distFROMz~ MR", task_id, " *krakenPresent + z * MR", task_id,
                            " + trial + block + age + as.factor(Sex_0) + edu + (trial + block + krakenPresent + z| ID)"))

# fit the model we want at this array job
model <- brm(formula,
             data = Master,
             family = "gaussian",
             iter = 4000,
             cores = 4,
             chains = 4,
             control = list(adapt_delta = 0.90, max_treedepth = 15))

# give the output the name of the questionnaire we are using here
assign(paste0("MR",task_id), summary(model))

# save the whole thing so we can make the visualisations locally
save.image(paste("replication_study/analyses/distance/",factors, "factors_MR",task_id, version,".Rdata", sep = ""))

