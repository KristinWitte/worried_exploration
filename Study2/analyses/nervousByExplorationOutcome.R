################ exploration predicting nervousness in the next round, moderated by the best reward found (Study 2) ########################
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

load("Study2/data/questionnairesPre.Rda")

Master <- Master %>%
  left_join(questionnairesPre %>% select(ID, age, Sex_0), by = "ID") %>%
  mutate(age = scale(age))

Master$next_nerv <- scale(as.numeric(nervous$nervous[match(paste(Master$ID, Master$block+1),
                                                           paste(nervous$ID, nervous$block))]))

Master$block <- scale(Master$block)

# maxZ is the highest reward the participant observed in the round
df <- Master %>%
  group_by(ID, block, age, Sex_0, cond, tp) %>%
  summarise(explore = meann(unique),
            nervous = meann(next_nerv),
            maxZ = max(z, na.rm = T)) %>%
  subset(!is.na(nervous)) %>%
  ungroup() %>%
  mutate(explore = scale(explore),
         maxZ = scale(maxZ))

print(cor.test(df$explore, df$maxZ))

model <- brm(nervous ~ explore * cond * tp + explore * maxZ + block + age + Sex_0 + (explore * tp + explore * maxZ + block | ID),
             data = df,
             cores = 2,
             chains = 2,
             iter = 4000)

summary(model)

nextNerv_maxZ <- summary(model)
nextNerv_maxZ_draws <- fixef(model, summary = FALSE)

save(nextNerv_maxZ, nextNerv_maxZ_draws, file = "Study2/analyses/nervous_by_exploration_outcome.Rda")
