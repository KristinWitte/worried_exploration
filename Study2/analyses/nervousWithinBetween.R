################ P(novel) by within- and between-person nervousness and intervention (Study 2) ########################
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

nervous$tp <- ifelse(nervous$tp == "Pre", -0.5, 0.5)
nervous$cond <- ifelse(nervous$cond == 0, -0.5, 0.5)
nervous$block <- rep(c(2,4,6,7,9,11), length(unique(Master$ID)))


## split nervousness into the participant's mean over all six rated rounds (between-person, nerv_b)
## and the deviation of each rated round from that mean (within-person, nerv_w)

nervous <- nervous %>%
  mutate(nervous = as.numeric(nervous)) %>%
  group_by(ID) %>%
  mutate(nerv_b = mean(nervous),
         nerv_w = nervous - nerv_b) %>%
  ungroup()

# nerv_b is z-scored across participants, nerv_w is divided by the pooled within-person SD
sd_b <- sd(nervous$nerv_b[!duplicated(nervous$ID)])
sd_w <- sd(nervous$nerv_w)

nervous <- nervous %>%
  mutate(nerv_b = (nerv_b - mean(nerv_b[!duplicated(ID)])) / sd_b,
         nerv_w = nerv_w / sd_w)


## adding age and gender to the Master data.frame

load("Study2/data/questionnairesPre.Rda")

Master <- Master %>%
  left_join(questionnairesPre %>% select(ID, age, Sex_0), by = "ID") %>%
  left_join(nervous %>% select(ID, block, nerv_w, nerv_b), by = c("ID", "block")) %>%
  mutate(age = scale(age))


Master$prev_z <- scale(Master$z[match(paste(Master$ID, Master$block, Master$trial-1),
                                paste(Master$ID, Master$block, Master$trial))])


## scaling trial and block

Master <- Master %>%
  mutate(trial = scale(trial),
         block = scale(block))


############### unique option selected by within- and between-person nervousness and intervention ###############

# nerv_b is constant within a participant, so only nerv_w gets random slopes
model <- brm(unique ~ nerv_w * cond * tp + nerv_b * cond * tp + nerv_w * prev_z + nerv_b * prev_z + cond*tp *prev_z + trial + block +age + Sex_0 +
               (nerv_w*tp + nerv_w * prev_z + tp * prev_z+ trial + block | ID),
             data = Master,
             family = "bernoulli",
             cores = 2,
             chains = 2,
             iter = 4000)

summary(model)

nerv_wb <- summary(model)
nerv_wb_draws <- fixef(model, summary = FALSE)
save(nerv_wb, nerv_wb_draws, sd_w, sd_b, file = "Study2/analyses/nervous_within_between.Rda")
