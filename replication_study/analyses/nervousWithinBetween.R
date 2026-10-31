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
  left_join(nerv %>% select(ID, block, nerv_w, nerv_b), by = c("ID", "block")) %>%
  mutate(krakenPresent = krakenPresent -0.5,
         prev_z = scale(as.numeric(prev_z)),
         trial = scale(trial),
         block = scale(block),
         age = scale(as.numeric(age)),
         edu = scale(as.numeric(edu)))


########## regression of within- and between-person nervousness on P(novel) ########

# nerv_b is constant within a participant, so only nerv_w gets random slopes
model <- brm(unique ~ krakenPresent * nerv_w + krakenPresent * nerv_b + nerv_w * prev_z + nerv_b * prev_z + trial + block + as.factor(Sex_0) + age + edu +
               (trial + block + krakenPresent * nerv_w + nerv_w*prev_z | ID),
             Master,
             chains = 2,
             cores = 2,
             family = "bernoulli",
             iter = 10000) # sd(nerv_w) has Rhat > 1.01 with iter = 5000

summary(model)

nerv_wb <- summary(model)
nerv_wb_draws <- fixef(model, summary = FALSE)
save(nerv_wb, nerv_wb_draws, sd_w, sd_b, file = paste0("replication_study/analyses/nervous_within_between_", version,".Rda"))
