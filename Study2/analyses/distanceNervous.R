################ distance between subsequent clicks by nervousness and intervention (Study 2) ########################
rm(list = ls())
library(brms)
library(tidyverse)
# run from the repo root
load("Study2/data/Master.Rda")
load("Study2/data/nervous.Rda")

# block 1 is extra practice
Master <- subset(Master, block > 1)

# all binary predictors are recoded to -0.5 and 0.5 to use simple coding instead of dummy coding in the regressions
Master$cond <- ifelse(Master$cond == "control", -0.5, 0.5)
Master$tp <- ifelse(Master$tp == "pre", -0.5, 0.5)

nervous$nervous <- scale(as.numeric(nervous$nervous), center = T, scale = F)
nervous$tp <- ifelse(nervous$tp == "Pre", -0.5, 0.5)
nervous$cond <- ifelse(nervous$cond == 0, -0.5, 0.5)
nervous$block <- rep(c(2,4,6,7,9,11), length(unique(Master$ID)))


## adding age and gender to the Master data.frame

load("Study2/data/questionnairesPre.Rda")

Master <- Master %>%
  left_join(questionnairesPre %>% select(ID, age, Sex_0), by = "ID") %>%
  left_join(nervous %>% select(ID, block, nervous), by = c("ID", "block")) %>%
  mutate(age = scale(age),
         nervous = scale(nervous))


Master$prev_z <- scale(Master$z[match(paste(Master$ID, Master$block, Master$trial-1),
                                paste(Master$ID, Master$block, Master$trial))])

Master$prev_nerv <- scale(nervous$nervous[match(paste(Master$ID, Master$block-1),
                                           paste(nervous$ID, nervous$block))])

# euclidean distance between the square clicked on this trial and the square clicked on the previous trial of the same round
# (na on the first click of a round and on trials where the participant had already lost all rewards and moved on to the next round)
prev <- match(paste(Master$ID, Master$block, Master$trial - 1), paste(Master$ID, Master$block, Master$trial))
Master$dist <- sqrt((Master$x - Master$x[prev])^2 + (Master$y - Master$y[prev])^2)

# mean-center the dependent variable
Master$dist <- scale(Master$dist, center = T, scale = F)


## scaling trial and block

Master <- Master %>%
  mutate(trial = scale(trial),
         block = scale(block))


############### distance by nervoussness and intervention ###############

model <- brm(dist ~ nervous * cond * tp+ nervous * prev_z  + cond*tp *prev_z+ trial + block +age + Sex_0 +
               (nervous*tp + nervous * prev_z + tp * prev_z+ trial + block | ID),
             data = Master,
             family = "gaussian",
             cores = 2,
             chains = 2,
             iter = 4000)

save(model, file ="Study2/analyses/distance_nervousByInterv.Rda")
summary(model)


######## test directionality of relationship between exploration and nervousness ###############


model <- brm(dist ~ prev_nerv * cond * tp+ prev_nerv * prev_z  + cond*tp *prev_z+ trial + block +age + Sex_0 + (prev_nerv*tp + prev_nerv * prev_z + tp * prev_z+ trial + block | ID),
             data = Master,
             family = "gaussian",
             cores = 2,
             chains = 2,
             iter = 2000)

summary(model)

dist_prevNerv <- summary(model)
save(dist_prevNerv, file = "Study2/analyses/distance_prevNerv.Rda")
