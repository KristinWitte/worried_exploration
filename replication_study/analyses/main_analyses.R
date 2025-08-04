library(tidyverse)
library(brms)

version <- "strict" # strict for min 3 correct attention checks, loose for min 2 correct attention checks

load(paste0("replication_study/data/questionnaires_", version,".Rda"))
load(paste0("replication_study/data/Master_", version,".Rda"))



######## cor of nervousness and PSWQ #########


questionnaires <- left_join(questionnaires, Master[Master$krakenPresent == 0, ] %>% 
                          group_by(ID) %>% 
                          summarise(nervous = mean(as.numeric(nervous), na.rm = T)), by = "ID")

cor.test(questionnaires$PSWQ, questionnaires$nervous)
cor(questionnaires[ ,2:9], use = "pairwise.complete.obs")


########## regression of nervousness on P(novel) ########


Master$unique<-ave(paste(Master$x, Master$y), paste(Master$ID, 'x', Master$block), FUN=duplicated)
Master$unique<-ifelse(Master$unique==TRUE, 0, 1)
Master$unique[is.na(Master$z)] <- NA
Master$prev_z <- Master$z[match(paste(Master$ID, Master$block, Master$trial-1), 
                                paste(Master$ID, Master$block, Master$trial))]

Master <- Master %>% 
  mutate(krakenPresent = krakenPresent -0.5,
         nervous = scale(as.numeric(nervous)),
         prev_z = scale(as.numeric(prev_z)),
         nextNerv = nervous[match(paste( ID, block+1), paste(ID, block))],
         prevNerv = nervous[match(paste(ID, block-1), paste(ID, block))],
         trial = scale(trial),
         block = scale(block),
         age = scale(as.numeric(age)),
         edu = scale(as.numeric(edu)))

model <- brm(unique ~ krakenPresent * nervous + nervous * prev_z  + trial + block + as.factor(Sex_0) + age + edu + 
               (trial + block + krakenPresent * nervous + nervous*prev_z | ID),
             Master,
             chains = 2,
             cores = 2,
             family = "bernoulli",
             iter = 5000)

summary(model)

nerv_pnovel <- summary(model)
save(nerv_pnovel, file = paste0("replication_study/analysis/nerv_pnovel_", version,".Rda"))

################# direction of relationship between exploration and nervousness ##############

## nervousness predicted from exploration in previous round

# aggregate bc we only have 1 nervousness per round anyway

df <- Master %>% 
  group_by(ID, block, krakenPresent, Sex_0, age, edu, krakenFound) %>% 
  summarise(nervous = mean(as.numeric(nextNerv), na.rm = T),
            explore = mean(unique, na.rm = T)) %>% 
  subset(!is.na(nervous)) %>% 
  ungroup() %>% 
  mutate(explore = scale(explore),
         krakenFound = krakenFound -0.5,
         nervous = scale(nervous))

model <- brm(nervous ~ explore* krakenPresent + krakenFound + block + as.factor(Sex_0) + age + edu + 
               (block + explore* krakenPresent + krakenFound | ID),
             df,
             chains = 2,
             cores = 2,
             iter = 2000)

summary(model)

## this is significant so what if I don't do the next round but the current one?

df <- Master %>% 
  group_by(ID, block, krakenPresent, Sex_0, age, edu, krakenFound) %>% 
  summarise(nervous = mean(as.numeric(nervous), na.rm = T),
            explore = mean(unique, na.rm = T)) %>% 
  subset(!is.na(nervous)) %>% 
  ungroup() %>% 
  mutate(explore = scale(explore),
         krakenFound = krakenFound -0.5,
         nervous = scale(nervous))

model <- brm(nervous ~ explore* krakenPresent + krakenFound + block + as.factor(Sex_0) + age + edu + 
               (block + explore* krakenPresent + krakenFound | ID),
             df,
             chains = 2,
             cores = 2,
             iter = 2000)

summary(model)

### now nervousness predicting exploration in next round

model <- brm(unique ~ krakenPresent *prevNerv + prevNerv *prev_z + trial + block + as.factor(Sex_0) + age + edu + 
               (trial + block + krakenPresent*prevNerv + prevNerv*prev_z | ID),
             Master,
             chains = 2,
             cores = 2,
             family = "bernoulli",
             iter = 4000)

summary(model)

# is the null effect here bc the other regression was using aggregate data?
df <- Master %>% 
  group_by(ID, block, krakenPresent, Sex_0, age, edu) %>% 
  summarise(nervous = mean(as.numeric(prevNerv), na.rm = T),
            explore = mean(unique, na.rm = T)) %>% 
  subset(!is.na(nervous)) %>% 
  ungroup() %>% 
  mutate(explore = scale(explore))

model <- brm(explore ~ krakenPresent * nervous + block + as.factor(Sex_0) + age + edu + 
               (block + krakenPresent * nervous | ID),
             df,
             chains = 2,
             cores = 2,
             iter = 4000)

summary(model) # nope still no effect


##########  regression of nervousness on model parameters ###########


estims <- read.csv(paste0("replication_study/analysis/estimatesCB_n_",version,".csv")) %>% 
  mutate(krakenPresent = kraken_present) %>% 
  left_join( Master %>% 
                      group_by(ID, krakenPresent, age, Sex_0, edu) %>% 
                      summarise(nervous = mean(as.numeric(nervous), na.rm = T)), by = c("ID", "krakenPresent")) %>% 
  mutate(krakenPresent = krakenPresent -0.5,
         ls = scale(ls),
         beta = scale(beta),
         tau = scale(tau),
         nervous = scale(nervous),
         age = scale(as.numeric(age)),
         edu = scale(as.numeric(edu)))

model <- brm(beta ~ krakenPresent *nervous +age + Sex_0 + edu + (1 | ID),
             estims,
             chains = 2,
             cores = 2,
             iter = 3000)

summary(model)
eta <- summary(model)

model <- brm(ls ~ krakenPresent *nervous +age + Sex_0 + edu + (1 | ID),
             estims,
             chains = 2,
             cores = 2,
             iter = 3000)

summary(model)

ls <- summary(model)

model <- brm(tau ~ krakenPresent *nervous +age + Sex_0 + edu + (1 | ID),
             estims,
             chains = 2,
             cores = 2,
             iter = 3000)

summary(model)

tau <- summary(model)

save(eta, ls, tau, file = paste0("estims_nervous_",version,".Rda"))

########### P(novel) in risky vs safe condition #################

load(paste0("replication_study/data/Master_", version,".Rda"))
Master$unique<-ave(paste(Master$x, Master$y), paste(Master$ID, 'x', Master$block), FUN=duplicated)
Master$unique<-ifelse(Master$unique==TRUE, 0, 1)
Master$unique[is.na(Master$z)] <- NA
Master$prev_z <- Master$z[match(paste(Master$ID, Master$block, Master$trial-1), 
                                paste(Master$ID, Master$block, Master$trial))]

Master <- Master %>% 
  mutate(krakenPresent = krakenPresent -0.5,
         nervous = scale(as.numeric(nervous)),
         prev_z = scale(as.numeric(prev_z)),
         trial = scale(trial),
         block = scale(block),
         age = scale(as.numeric(age)),
         edu = scale(as.numeric(edu)))


model <- brm(unique ~ krakenPresent * prev_z + trial + block + as.factor(Sex_0) + age + edu + 
               (trial + block + krakenPresent *prev_z | ID),
             Master,
             chains = 2,
             cores = 2,
             family = "bernoulli",
             iter = 2000)


summary(model)

