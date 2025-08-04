################ regressions for Study 2 ########################
rm(list = ls())
library(plyr)
library(brms)
library(ggpubr)
library(gghalves)
library(here)
library(tidyverse)
se<-function(x){sd(x, na.rm = T)/sqrt(length(na.omit(x)))}
meann <- function(x){mean(x, na.rm = T)}
load("Study2/data/Master.Rda")
load("Study2/data/nervous.Rda")

# block 1 is extra practice
Master <- subset(Master, block > 1)
Master$row <- 1:nrow(Master)

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

nervous <- nervous %>% 
  left_join(questionnairesPre %>% select(ID, age, Sex_0), by = "ID")


Master$prev_z <- scale(Master$z[match(paste(Master$ID, Master$block, Master$trial-1), 
                                paste(Master$ID, Master$block, Master$trial))])

Master$next_nerv <- scale(nervous$nervous[match(paste(Master$ID, Master$block+1),
                                          paste(nervous$ID, nervous$block))])

Master$prev_nerv <- scale(nervous$nervous[match(paste(Master$ID, Master$block-1),
                                           paste(nervous$ID, nervous$block))])


## scaling trial and block

Master <- Master %>% 
  mutate(trial = scale(trial),
         block = scale(block))



########## nervous by intervention ###########



brm(nervous ~ cond * tp + age + Sex_0 + (tp | ID),
    data = nervous,
    cores = 2,
    chains = 2,
    iter = 4000)

# Regression Coefficients:
#   Estimate Est.Error l-95% CI u-95% CI Rhat Bulk_ESS Tail_ESS
# Intercept     9.21      8.02    -6.39    25.04 1.01      810     1438
# cond         -1.85      5.38   -12.06     9.04 1.00      904     1455
# tp          -12.47      2.92   -18.34    -6.79 1.00     2160     2564
# age          -0.55      0.23    -0.98    -0.10 1.00      930     1590
# Sex_0        14.78      5.23     4.70    25.29 1.00      720     1366
# cond:tp     -18.95      5.86   -30.21    -7.67 1.00     2331     2695




####### lost all rewards by intervention ###########

df <- ddply(Master, ~ID+cond+tp+block+age+Sex_0, summarise, krakenFound = mean(krakenFound))

brm(krakenFound ~ cond*tp  + age + Sex_0+ (tp | ID),
    data = df,
    family = "bernoulli",
    cores = 2,
    chains = 2,
    iter = 4000)

# nothing

############### unique option selected by nervoussness and intervention ###############

model <- brm(unique ~ nervous * cond * tp+ nervous * prev_z  + cond*tp *prev_z+ trial + block +age + Sex_0 + 
               (nervous*tp + nervous * prev_z + tp * prev_z+ trial + block | ID),
             data = Master,
             family = "bernoulli",
             cores = 2,
             chains = 2,
             iter = 4000)

save(model, file ="Study2/nervousByInterv.Rda")
summary(model)

# Regression Coefficients:
#   Estimate Est.Error l-95% CI u-95% CI Rhat Bulk_ESS Tail_ESS
# Intercept           0.15      0.78    -1.44     1.69 1.00     1882     2177
# nervous             1.16      0.46     0.31     2.10 1.00     1663     2262
# cond               -0.33      1.07    -2.47     1.69 1.00     1606     2219
# tp                 -0.29      0.67    -1.59     1.01 1.00     1708     2386
# prev_z             -2.10      0.34    -2.77    -1.46 1.00     2997     3248
# trial              -3.26      0.39    -4.06    -2.55 1.00     2362     2639
# block              -0.42      0.29    -0.99     0.15 1.00     3094     3007
# age                -0.69      0.52    -1.74     0.31 1.00     2106     2590
# Sex_0               0.37      1.04    -1.62     2.52 1.00     1670     2243
# nervous:cond       -0.22      0.82    -1.82     1.38 1.00     1973     2733
# nervous:tp         -0.52      0.73    -2.00     0.86 1.00     2553     2848
# cond:tp            -0.57      1.11    -2.78     1.66 1.00     2420     2608
# nervous:prev_z      0.54      0.29    -0.04     1.09 1.00     2317     3227
# cond:prev_z         0.85      0.61    -0.30     2.07 1.00     3037     3126
# tp:prev_z          -0.26      0.37    -0.96     0.48 1.00     3264     3174
# nervous:cond:tp    -0.23      1.33    -2.88     2.39 1.00     3504     2766
# cond:tp:prev_z     -0.95      0.69    -2.31     0.41 1.00     3767     2896

## account for sticsa

Master$STICSAcog <- scale(Master$STICSAcog)

model <- brm(unique ~ nervous * cond * tp+ nervous * prev_z  + cond*tp *prev_z+ trial + block +age + Sex_0 + STICSAcog+ (nervous*tp + nervous * prev_z + tp * prev_z+ trial + block | ID),
             data = Master,
             family = "bernoulli",
             cores = 2,
             chains = 2,
             iter = 2000)

summary(model)

stargazer::stargazer(summary(model)$fixed, summary = F)

# does not make much of a difference

### include only baseline
df <- subset(Master, tp == -0.5)

model <- brm(unique ~ nervous * cond+ nervous * prev_z  + cond *prev_z+ trial + block +age + Sex_0 + 
               (nervous + nervous * prev_z + trial + block | ID),
             data = df,
             family = "bernoulli",
             cores = 2,
             chains = 2,
             iter = 4000)

model


######## test directionality of relationship between exploration and nervousness ###############


model <- brm(unique ~ prev_nerv * cond * tp+ prev_nerv * prev_z  + cond*tp *prev_z+ trial + block +age + Sex_0 + (prev_nerv*tp + prev_nerv * prev_z + tp * prev_z+ trial + block | ID),
             data = Master,
             family = "bernoulli",
             cores = 2,
             chains = 2,
             iter = 2000)

summary(model)

# Regression Coefficients:
#   Estimate Est.Error l-95% CI u-95% CI Rhat Bulk_ESS Tail_ESS
# Intercept             0.39      0.78    -1.08     1.92 1.01      884      954
# prev_nerv            -0.03      0.24    -0.49     0.46 1.00     1417     1415
# cond                 -0.19      1.08    -2.20     1.96 1.00      693     1103
# tp                   -0.81      0.68    -2.09     0.63 1.00     1184     1245
# prev_z               -2.34      0.35    -3.06    -1.68 1.00     1229     1444
# trial                -2.77      0.32    -3.47    -2.19 1.00     1425     1306
# block                -0.20      0.40    -1.01     0.57 1.00     1546     1359
# age                  -0.62      0.54    -1.71     0.39 1.00      885     1071
# Sex_0                -0.59      1.03    -2.53     1.42 1.00      862     1144
# prev_nerv:cond       -0.32      0.45    -1.25     0.59 1.00     1212     1079
# prev_nerv:tp          0.61      0.42    -0.22     1.46 1.00     1497     1363
# cond:tp              -1.71      1.04    -3.78     0.30 1.00     1276     1073
# prev_nerv:prev_z      0.05      0.19    -0.35     0.40 1.00     2098     1640
# cond:prev_z          -0.30      0.62    -1.55     0.89 1.00     1243     1356
# tp:prev_z             0.00      0.48    -0.86     1.03 1.00      866     1255
# prev_nerv:cond:tp     1.49      0.83    -0.10     3.13 1.00     1422     1335
# cond:tp:prev_z        0.18      0.83    -1.45     1.83 1.00     1692     1474

df <- Master %>% 
  group_by(ID, block, age, Sex_0, cond, tp) %>% 
  summarise(explore = meann(unique),
            nervous = meann(next_nerv)) %>% 
  ungroup() %>% 
  mutate(explore = scale(explore))


model <- brm(nervous ~ explore * cond * tp + block +age + Sex_0+ (explore*tp  + tp + block | ID),
             data = df,
             cores = 2,
             chains = 2,
             iter = 4000)

summary(model)

# Regression Coefficients:
#   Estimate Est.Error l-95% CI u-95% CI Rhat Bulk_ESS Tail_ESS
# Intercept          -0.24      0.12    -0.47     0.01 1.00     1226     2170
# explore             0.07      0.05    -0.02     0.17 1.00     3799     3124
# cond               -0.18      0.17    -0.52     0.16 1.00     1288     1820
# tp                  0.10      0.15    -0.19     0.41 1.00     3564     3113
# block              -0.17      0.09    -0.35     0.01 1.00     3726     3121
# age                -0.18      0.09    -0.35    -0.00 1.00     1559     1864
# Sex_0               0.47      0.16     0.14     0.80 1.00     1367     2247
# explore:cond        0.08      0.10    -0.12     0.28 1.00     2841     2996
# explore:tp          0.10      0.09    -0.06     0.27 1.00     4535     2906
# cond:tp            -0.29      0.17    -0.64     0.05 1.00     4585     3105
# explore:cond:tp     0.02      0.17    -0.32     0.36 1.00     4844     2915

######## mediation analysis ###########
# source: https://en.wikipedia.org/wiki/Mediation_(statistics)
# nuo stands for number of unique (novel) options
NUO <- ddply(Master, ~ID+tp+cond+age+Sex_0+STICSAcog, unique = meann(Master$unique))

step1 <- brm(unique ~ tp * cond + age + Sex_0 + STICSAcog,
             data = NUO,
             cores = 2,
             chains = 2,
             iter = 4000)


df <- Master[Master$block != 1 & Master$block != 3 & Master$block != 5 & Master$block != 8 & Master$block != 10, ]
df$trial <- scale(df$trial)
df$block <- scale(df$block)
## add nervous

df$nervous <- rep(scale(as.numeric(nervous$nervous)), each = 26)

step2 <- brm(nervous ~  cond * tp + trial + block + age + Sex_0 + STICSAcog + (tp + trial + block | ID),
             data = df,
             cores = 2,
             chains = 2,
             iter = 4000)

step3 <-  brm(unique ~ nervous * cond * tp + trial + block + age + Sex_0 + STICSAcog +(nervous*tp + trial + block | ID),
          data = df,
          family = "bernoulli",
          cores = 2,
          chains = 2,
          iter = 4000)


summary(step1)
summary(step2)
summary(step3)


########## same but for eta parameter from Novelty bonus model

estims <- read.csv("Study2/estimatesCB_n.csv")
estims$tp <- estims$tp -0.5
estims$cond <- estims$cond - 0.5

estims$ls <- scale(estims$ls, center = T, scale = F)
estims$tau <- scale(estims$tau, center = T, scale = F)
estims$beta <- scale(estims$beta, center = T, scale = F)

nerv <- ddply(nervous, ~ID+cond+tp, summarise, nervous = meann(as.numeric(nervous)))

estims$nervous <- scale(nerv$nervous[match(paste(estims$ID, estims$tp), paste(nerv$ID, nerv$tp))])

estims <- estims %>% left_join(questionnairesPre %>% select(ID, age, Sex_0), by = "ID") %>% 
  mutate(age = scale(age))

step1 <- brm(beta ~ tp * cond + age + Sex_0 + (1|ID),
             data = estims,
             cores = 2,
             chains = 2,
             iter = 4000)

step2 <- brm(nervous ~  cond * tp + age + Sex_0+ (1 | ID),
             data = estims,
             cores = 2,
             chains = 2,
             iter = 4000)

step3 <-  brm(beta ~ nervous * cond * tp + age + Sex_0+ (1 | ID),
              data = estims,
              cores = 2,
              chains = 2,
              iter = 4000)


summary(step1)
summary(step2)
summary(step3)


########### reward by intervention ###########

z<- ddply(Master, ~ID+tp+cond, z = meann(Master$z))

brm(z ~ tp * cond,
    data = z,
    cores = 2,
    chains = 2,
    iter = 4000)

# Population-Level Effects: 
#   Estimate Est.Error l-95% CI u-95% CI Rhat Bulk_ESS Tail_ESS
# Intercept    74.81      0.27    74.26    75.34 1.00     5560     3145
# tp           -3.77      0.54    -4.80    -2.73 1.00     4618     3231
# cond          3.00      0.54     1.95     4.05 1.00     5566     3053
# tp:cond      -2.04      1.04    -4.06     0.02 1.00     4480     3043

####################### parameter estimates by intervention #################


estims <- read.csv("Study2/estimatesCB_n.csv")
estims$tp <- estims$tp -0.5
estims$cond <- estims$cond - 0.5

# the parameter eta is called beta throughout this script for convenience of recycling code

estims$ls <- scale(estims$ls, center = T, scale = F)
estims$tau <- scale(estims$tau, center = T, scale = F)
estims$beta <- scale(estims$beta, center = T, scale = F)

nerv <- nervous %>% 
  group_by(ID, tp, age,Sex_0) %>% 
  summarise(nervous = meann(nervous))

estims <- estims %>% left_join(nerv, by = c("ID", "tp")) %>% 
  mutate(nervous = scale(nervous),# it was just mean-centered before
         age = scale(age))

estims <- estims %>% left_join(Master %>% group_by(ID) %>% summarise(STICSAcog = mean(STICSAcog)), 
                               by = "ID") %>% 
  ungroup() %>% 
  mutate(STICSAcog = scale(as.numeric(STICSAcog)))

brm(ls ~ nervous * cond * tp +age + Sex_0 + STICSAcog + (1|ID),# unable to make it converge with tp random slope
    data = estims,
    cores = 2,
    chains = 2,
    iter = 8000)

# Regression Coefficients:
#   Estimate Est.Error l-95% CI u-95% CI Rhat Bulk_ESS Tail_ESS
# Intercept           0.15      0.12    -0.09     0.38 1.00     3525     4873
# nervous            -0.00      0.00    -0.00     0.00 1.00     4925     6304
# cond               -0.04      0.07    -0.18     0.10 1.00     3399     5247
# tp                  0.08      0.04    -0.00     0.16 1.00    10070     6571
# age                -0.00      0.00    -0.01     0.00 1.00     3597     4862
# Sex_0               0.01      0.07    -0.13     0.16 1.00     3133     4498
# STICSAcog          -0.06      0.04    -0.13     0.02 1.00     3400     5291
# nervous:cond        0.00      0.00    -0.00     0.01 1.00     4071     4961
# nervous:tp          0.00      0.00    -0.00     0.01 1.00    11028     6618
# cond:tp             0.09      0.08    -0.07     0.25 1.00     9391     5681
# nervous:cond:tp     0.01      0.00     0.00     0.01 1.00     8267     6594


model <-brm(tau ~ nervous * cond * tp + age + Sex_0 + STICSAcog + (1|ID),
            data = estims,
            cores = 2,
            chains = 2,
            iter = 8000)

model

# Regression Coefficients:
#   Estimate Est.Error l-95% CI u-95% CI Rhat Bulk_ESS Tail_ESS
# Intercept          -0.00      0.01    -0.03     0.02 1.00     7125     6346
# nervous             0.00      0.00    -0.00     0.00 1.00     7174     6712
# cond               -0.00      0.01    -0.02     0.01 1.00     5796     5596
# tp                 -0.00      0.01    -0.01     0.01 1.00    13191     5861
# age                -0.00      0.00    -0.00     0.00 1.00     7472     6377
# Sex_0               0.02      0.01     0.00     0.03 1.00     5534     6087
# STICSAcog          -0.01      0.00    -0.02     0.00 1.00     6008     6110
# nervous:cond        0.00      0.00    -0.00     0.00 1.00     7235     6480
# nervous:tp          0.00      0.00    -0.00     0.00 1.00     8292     5721
# cond:tp             0.02      0.01    -0.01     0.04 1.00    11193     5423
# nervous:cond:tp     0.00      0.00    -0.00     0.00 1.00     8878     5510

model <-brm(beta ~ nervous * cond * tp + age + Sex_0 + STICSAcog + (1|ID), # the results are qualitatively the same if we take out nervousness
            data = estims,
            cores = 2,
            chains = 2,
            iter = 8000)

model
beta <- summary(model)

save(beta, file = "eta.Rda")


# Regression Coefficients:
#   Estimate Est.Error l-95% CI u-95% CI Rhat Bulk_ESS Tail_ESS
# Intercept          -0.00      0.15    -0.29     0.29 1.00     2936     4773
# nervous             0.00      0.00    -0.00     0.00 1.00     4621     5750
# cond               -0.04      0.09    -0.22     0.14 1.00     2830     4113
# tp                 -0.17      0.05    -0.27    -0.07 1.00     8482     5618
# age                -0.00      0.00    -0.01     0.01 1.00     2747     4715
# Sex_0               0.12      0.09    -0.07     0.30 1.00     3074     4929
# STICSAcog           0.03      0.05    -0.07     0.13 1.00     2890     4179
# nervous:cond       -0.00      0.00    -0.01     0.00 1.00     4231     5645
# nervous:tp         -0.00      0.00    -0.00     0.00 1.00     9177     6623
# cond:tp            -0.34      0.10    -0.53    -0.14 1.00     8156     6134
# nervous:cond:tp    -0.00      0.00    -0.01     0.01 1.00     7637     6320


######### difference in estimates before the intervention (just to make sure)


df <- subset(estims, tp == -0.5)

brm(tau ~ cond,
    data = df,
    cores = 2,
    chains = 2,
    iter = 4000)
## no difference!

brm(ls ~ cond,
    data = df,
    cores = 2,
    chains = 2,
    iter = 4000)
# no difference!

brm(beta ~ cond,
    data = df,
    cores = 2,
    chains = 2,
    iter = 4000)
# no difference

t.test(df$beta[df$cond == -0.5], df$beta[df$cond == 0.5]) # still no difference

############# question during intervention about whether reclicking or clicking as many as possible ######

load("Study2/data/intervention.Rda")

# Wilcoxon signed rank test with continuity correction

library(MASS)
wilcox.test(intervention$reclicking, intervention$reclicking2, paired=TRUE) 

##################### test for inattention bias ############################


load("Study2/data/nervous.Rda")
load("Study2/data/questionnairesPre.Rda")
load("Study2/data/questionnairesPost.Rda")

df <- nervous %>% 
  left_join(questionnairesPre %>% select(ID, attention), by = "ID") %>% 
  left_join(questionnairesPost %>% select(ID, attention2), by = "ID") %>% 
  mutate(attention_agg = attention + attention2) %>% 
  group_by(ID) %>% 
  summarise(nervous = meann(as.numeric(nervous)),
            attention = mean(attention_agg))


## did nervousness differ depending on whether participants missed 1 attention check?

t.test(df$nervous[df$attention == 3], df$nervous[df$attention == 4], paired = F)

# nope

# data:  df$nervous[df$attention == 3] and df$nervous[df$attention == 4]
# t = -0.91704, df = 23.129, p-value = 0.3686
# alternative hypothesis: true difference in means is not equal to 0
# 95 percent confidence interval:
#   -17.648621   6.804948
# sample estimates:
#   mean of x mean of y 
# 26.38462  31.80645 




