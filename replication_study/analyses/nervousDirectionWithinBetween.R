library(tidyverse)
library(brms)

# run from the repo root
version <- commandArgs(TRUE)[1] # strict for min 3 correct attention checks, loose for min 2 correct attention checks

load(paste0("replication_study/data/Master_", version,".Rda"))

Master$unique<-ave(paste(Master$x, Master$y), paste(Master$ID, 'x', Master$block), FUN=duplicated)
Master$unique<-ifelse(Master$unique==TRUE, 0, 1)
Master$unique[is.na(Master$z)] <- NA
Master$prev_z <- Master$z[match(paste(Master$ID, Master$block, Master$trial-1),
                                paste(Master$ID, Master$block, Master$trial))]


## split nervousness into the participant's mean over all rated rounds (between-person, nerv_b)
## and the deviation of each rated round from that mean (within-person, nerv_w)

nerv <- Master %>%
  distinct(ID, block, nervous) %>%
  subset(!is.na(nervous)) %>%
  mutate(nervous = as.numeric(nervous)) %>%
  group_by(ID) %>%
  mutate(nerv_b = mean(nervous),
         nerv_w = nervous - nerv_b) %>%
  ungroup()

# one nervousness rating per round
stopifnot(!any(duplicated(paste(nerv$ID, nerv$block))))

# nerv_b is z-scored across participants, nerv_w is divided by the pooled within-person SD
sd_b <- sd(nerv$nerv_b[!duplicated(nerv$ID)])
sd_w <- sd(nerv$nerv_w)

nerv <- nerv %>%
  mutate(nerv_b = (nerv_b - mean(nerv_b[!duplicated(ID)])) / sd_b,
         nerv_w = nerv_w / sd_w)

Master <- Master %>%
  left_join(nerv %>% distinct(ID, nerv_b), by = "ID") %>%
  mutate(prev_nerv_w = nerv$nerv_w[match(paste(ID, block-1), paste(nerv$ID, nerv$block))], # within-person nervousness of the previous round
         krakenPresent = krakenPresent -0.5,
         nervous = scale(as.numeric(nervous)),
         prev_z = scale(as.numeric(prev_z)),
         nextNerv = nervous[match(paste( ID, block+1), paste(ID, block))],
         trial = scale(trial),
         block = scale(block),
         age = scale(as.numeric(age)),
         edu = scale(as.numeric(edu)))


################# nervousness predicting exploration in the next round ##############

# nerv_b is constant within a participant, so only prev_nerv_w gets random slopes
model <- brm(unique ~ krakenPresent * prev_nerv_w + krakenPresent * nerv_b + prev_nerv_w * prev_z + nerv_b * prev_z + trial + block + as.factor(Sex_0) + age + edu +
               (trial + block + krakenPresent * prev_nerv_w + prev_nerv_w * prev_z | ID),
             Master,
             chains = 2,
             cores = 2,
             family = "bernoulli",
             iter = 20000) # cor(prev_nerv_w, prev_z) has Rhat > 1.01 with iter = 10000

summary(model)

prevNerv_wb <- summary(model)
prevNerv_wb_draws <- fixef(model, summary = FALSE)


################# exploration predicting nervousness in the next round ##############

## split exploration into the participant's mean over all rounds (between-person, explore_b)
## and the deviation of each round from that mean (within-person, explore_w)

# aggregate bc we only have 1 nervousness per round anyway
df <- Master %>%
  group_by(ID, block, krakenPresent, Sex_0, age, edu, krakenFound) %>%
  summarise(nervous = mean(as.numeric(nextNerv), na.rm = T),
            explore = mean(unique, na.rm = T)) %>%
  group_by(ID) %>%
  mutate(explore_b = mean(explore),
         explore_w = explore - explore_b) %>%
  ungroup()

# explore_b is z-scored across participants, explore_w is divided by the pooled within-person SD
sd_explore_b <- sd(df$explore_b[!duplicated(df$ID)])
sd_explore_w <- sd(df$explore_w)

df <- df %>%
  mutate(explore_b = (explore_b - mean(explore_b[!duplicated(ID)])) / sd_explore_b,
         explore_w = explore_w / sd_explore_w) %>%
  subset(!is.na(nervous)) %>%
  mutate(krakenFound = krakenFound -0.5,
         nervous = scale(nervous))

# explore_b is constant within a participant, so only explore_w gets random slopes
model <- brm(nervous ~ explore_w * krakenPresent + explore_b * krakenPresent + krakenFound + block + as.factor(Sex_0) + age + edu +
               (block + explore_w * krakenPresent + krakenFound | ID),
             df,
             chains = 2,
             cores = 2,
             iter = 10000) # sd(block) and cor(explore_w, krakenFound) have Rhat > 1.01 with iter = 4000

summary(model)

explore_wb <- summary(model)
explore_wb_draws <- fixef(model, summary = FALSE)

save(prevNerv_wb, prevNerv_wb_draws, sd_w, sd_b, explore_wb, explore_wb_draws, sd_explore_w, sd_explore_b,
     file = paste0("replication_study/analyses/nervous_direction_within_between_", version,".Rda"))
