library(tidyverse)
library(brms)

# run from the repo root
version <- commandArgs(TRUE)[1] # strict for min 3 correct attention checks, loose for min 2 correct attention checks

load(paste0("replication_study/data/Master_", version,".Rda"))


# euclidean distance between the square clicked on this trial and the square clicked on the previous trial of the same round
# (na on the first click of a round and on trials where the participant had already lost all rewards and moved on to the next round)
prev <- match(paste(Master$ID, Master$block, Master$trial - 1), paste(Master$ID, Master$block, Master$trial))
Master$dist <- sqrt((Master$x - Master$x[prev])^2 + (Master$y - Master$y[prev])^2)
Master$prev_z <- Master$z[match(paste(Master$ID, Master$block, Master$trial-1),
                                paste(Master$ID, Master$block, Master$trial))]

Master <- Master %>%
  mutate(dist = scale(dist, center = T, scale = F), # mean-center the dependent variable
         krakenPresent = krakenPresent -0.5,
         nervous = scale(as.numeric(nervous)),
         prev_z = scale(as.numeric(prev_z)),
         prevNerv = nervous[match(paste(ID, block-1), paste(ID, block))],
         trial = scale(trial),
         block = scale(block),
         age = scale(as.numeric(age)),
         edu = scale(as.numeric(edu)))


########## regression of nervousness on distance ########

model <- brm(dist ~ krakenPresent * nervous + nervous * prev_z  + trial + block + as.factor(Sex_0) + age + edu +
               (trial + block + krakenPresent * nervous + nervous*prev_z | ID),
             Master,
             chains = 2,
             cores = 2,
             family = "gaussian",
             iter = 5000)

summary(model)

nerv_dist <- summary(model)
save(nerv_dist, file = paste0("replication_study/analyses/distance_nerv_", version,".Rda"))


################# direction of relationship between exploration and nervousness ##############

### nervousness predicting distance in next round

model <- brm(dist ~ krakenPresent *prevNerv + prevNerv *prev_z + trial + block + as.factor(Sex_0) + age + edu +
               (trial + block + krakenPresent*prevNerv + prevNerv*prev_z | ID),
             Master,
             chains = 2,
             cores = 2,
             family = "gaussian",
             iter = 4000)

summary(model)

prevNerv_dist <- summary(model)
save(prevNerv_dist, file = paste0("replication_study/analyses/distance_prevNerv_", version,".Rda"))


########### distance in risky vs safe condition #################

model <- brm(dist ~ krakenPresent * prev_z + trial + block + as.factor(Sex_0) + age + edu +
               (trial + block + krakenPresent *prev_z | ID),
             Master,
             chains = 2,
             cores = 2,
             family = "gaussian",
             iter = 2000)


summary(model)

cond_dist <- summary(model)
save(cond_dist, file = paste0("replication_study/analyses/distance_cond_", version,".Rda"))
