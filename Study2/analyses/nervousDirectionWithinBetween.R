################ direction of the relationship between exploration and nervousness, within- and between-person (Study 2) ########################
rm(list = ls())
library(brms)
library(tidyverse)
# run from the repo root
meann <- function(x){mean(x, na.rm = T)}
load("Study2/data/Master.Rda")
load("Study2/data/nervous.Rda")

# block 1 is extra practice
Master <- subset(Master, block > 1)

# all binary predictors are recoded to -0.5 and 0.5 to use simple coding instead of dummy coding in the regressions
Master$cond <- ifelse(Master$cond == "control", -0.5, 0.5)
Master$tp <- ifelse(Master$tp == "pre", -0.5, 0.5)

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
  left_join(nervous %>% distinct(ID, nerv_b), by = "ID") %>%
  mutate(age = scale(age))


Master$prev_z <- scale(Master$z[match(paste(Master$ID, Master$block, Master$trial-1),
                                paste(Master$ID, Master$block, Master$trial))])

# within-person nervousness of the previous round
Master$prev_nerv_w <- nervous$nerv_w[match(paste(Master$ID, Master$block-1),
                                           paste(nervous$ID, nervous$block))]

# nervousness of the next round
Master$next_nerv <- scale(nervous$nervous[match(paste(Master$ID, Master$block+1),
                                          paste(nervous$ID, nervous$block))])


## scaling trial and block

Master <- Master %>%
  mutate(trial = scale(trial),
         block = scale(block))


######## nervousness predicting exploration in the next round ###############

# nerv_b is constant within a participant, so only prev_nerv_w gets random slopes
model <- brm(unique ~ prev_nerv_w * cond * tp + nerv_b * cond * tp + prev_nerv_w * prev_z + nerv_b * prev_z + cond*tp *prev_z + trial + block +age + Sex_0 +
               (prev_nerv_w*tp + prev_nerv_w * prev_z + tp * prev_z+ trial + block | ID),
             data = Master,
             family = "bernoulli",
             cores = 2,
             chains = 2,
             iter = 4000)

summary(model)

prevNerv_wb <- summary(model)
prevNerv_wb_draws <- fixef(model, summary = FALSE)


######## exploration predicting nervousness in the next round ###############

## split exploration into the participant's mean over all rounds (between-person, explore_b)
## and the deviation of each round from that mean (within-person, explore_w)

df <- Master %>%
  group_by(ID, block, age, Sex_0, cond, tp) %>%
  summarise(explore = meann(unique),
            nervous = meann(next_nerv)) %>%
  group_by(ID) %>%
  mutate(explore_b = mean(explore),
         explore_w = explore - explore_b) %>%
  ungroup()

# explore_b is z-scored across participants, explore_w is divided by the pooled within-person SD
sd_explore_b <- sd(df$explore_b[!duplicated(df$ID)])
sd_explore_w <- sd(df$explore_w)

df <- df %>%
  mutate(explore_b = (explore_b - mean(explore_b[!duplicated(ID)])) / sd_explore_b,
         explore_w = explore_w / sd_explore_w)

# explore_b is constant within a participant, so only explore_w gets random slopes
model <- brm(nervous ~ explore_w * cond * tp + explore_b * cond * tp + block +age + Sex_0+ (explore_w*tp + block | ID),
             data = df,
             cores = 2,
             chains = 2,
             iter = 4000,
             control = list(adapt_delta = 0.95)) # 1 divergent transition with the default adapt_delta = 0.8

summary(model)

explore_wb <- summary(model)
explore_wb_draws <- fixef(model, summary = FALSE)

save(prevNerv_wb, prevNerv_wb_draws, sd_w, sd_b, explore_wb, explore_wb_draws, sd_explore_w, sd_explore_b,
     file = "Study2/analyses/nervous_direction_within_between.Rda")
