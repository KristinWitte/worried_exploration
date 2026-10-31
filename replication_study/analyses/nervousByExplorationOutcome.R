################ exploration predicting nervousness in the next round, moderated by the best reward found (Study 3) ########################
library(tidyverse)
library(brms)

# run from the repo root
version <- commandArgs(TRUE)[1] # strict for min 3 correct attention checks, loose for min 2 correct attention checks

load(paste0("replication_study/data/Master_", version,".Rda"))

Master$unique<-ave(paste(Master$x, Master$y), paste(Master$ID, 'x', Master$block), FUN=duplicated)
Master$unique<-ifelse(Master$unique==TRUE, 0, 1)
Master$unique[is.na(Master$z)] <- NA

Master <- Master %>%
  mutate(krakenPresent = krakenPresent -0.5,
         nervous = scale(as.numeric(nervous)),
         nextNerv = nervous[match(paste( ID, block+1), paste(ID, block))],
         block = scale(block),
         age = scale(as.numeric(age)),
         edu = scale(as.numeric(edu)))

# aggregate bc we only have 1 nervousness per round anyway
# maxZ is the highest reward the participant observed in the round
df <- Master %>%
  group_by(ID, block, krakenPresent, Sex_0, age, edu, krakenFound) %>%
  summarise(nervous = mean(as.numeric(nextNerv), na.rm = T),
            explore = mean(unique, na.rm = T),
            maxZ = max(z, na.rm = T)) %>%
  subset(!is.na(nervous)) %>%
  ungroup() %>%
  mutate(explore = scale(explore),
         maxZ = scale(maxZ),
         krakenFound = krakenFound -0.5,
         nervous = scale(nervous))

print(cor.test(df$explore, df$maxZ))

model <- brm(nervous ~ explore * maxZ + explore * krakenPresent + krakenFound + block + as.factor(Sex_0) + age + edu +
               (block + explore * maxZ + explore * krakenPresent + krakenFound | ID),
             df,
             chains = 2,
             cores = 2,
             iter = 4000)

summary(model)

nextNerv_maxZ <- summary(model)
nextNerv_maxZ_draws <- fixef(model, summary = FALSE)

save(nextNerv_maxZ, nextNerv_maxZ_draws,
     file = paste0("replication_study/analyses/nervous_by_exploration_outcome_", version,".Rda"))
