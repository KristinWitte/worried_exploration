################ regressions for Study 2 ########################

library(plyr)
library(brms)
library(ggpubr)
library(gghalves)
setwd("/Users/kristinwitte/Documents/exploration_worries")
se<-function(x){sd(x, na.rm = T)/sqrt(length(na.omit(x)))}
meann <- function(x){mean(x, na.rm = T)}
load("Study2/Master.Rda")
load("Study2/nervous.Rda")

# block 1 is extra practice
Master <- subset(Master, block > 1)
Master$row <- 1:nrow(Master)

# all binary predictors are recoded to -0.5 and 0.5 to use simple coding instead of dummy coding in the regressions
Master$cond <- ifelse(Master$cond == "control", -0.5, 0.5)
Master$tp <- ifelse(Master$tp == "pre", -0.5, 0.5)

nervous$nervous <- scale(as.numeric(nervous$nervous), center = T, scale = F)
nervous$tp <- ifelse(nervous$tp == "Pre", -0.5, 0.5)
nervous$cond <- ifelse(nervous$cond == 0, -0.5, 0.5)



########## nervous by intervention ###########



brm(nervous ~ cond * tp + (tp | ID),
    data = nervous,
    cores = 2,
    chains = 2,
    iter = 4000)

# Population-Level Effects: 
#   Estimate Est.Error l-95% CI u-95% CI Rhat Bulk_ESS Tail_ESS
# Intercept    -0.11      2.84    -5.75     5.40 1.00      721     1352
# cond         -2.07      5.76   -13.50     9.33 1.00      725     1418
# tp          -12.46      2.91   -18.18    -6.84 1.00     2285     2856
# cond:tp     -18.81      5.88   -30.63    -7.32 1.00     2222     3051




####### lost all rewards by intervention ###########

df <- ddply(Master, ~ID+cond+tp+block, summarise, krakenFound = mean(krakenFound))

brm(krakenFound ~ cond*tp  + (tp | ID),
    data = df,
    family = "bernoulli",
    cores = 2,
    chains = 2,
    iter = 4000)

# nothing

############### unique option selected by nervoussness and intervention ###############

### toss out blocks for which I didn't ask about nervousness

df <- Master[Master$block != 1 & Master$block != 3 & Master$block != 5 & Master$block != 8 & Master$block != 10, ]
df$trial <- scale(df$trial)
df$block <- scale(df$block)
## add nervous

df$nervous <- rep(scale(as.numeric(nervous$nervous)), each = 26)

model <- brm(unique ~ nervous * cond * tp + trial + block + (nervous*tp + trial + block | ID),
    data = df,
    family = "bernoulli",
    cores = 2,
    chains = 2,
    iter = 4000)

 model


# Population-Level Effects: 
#   Estimate Est.Error l-95% CI u-95% CI Rhat Bulk_ESS Tail_ESS
# Intercept          -1.64      0.67    -2.99    -0.33 1.00      677     1120
# nervous             2.09      0.52     1.13     3.20 1.00      752     1731
# cond               -0.30      0.94    -2.10     1.62 1.00      961     1462
# tp                 -0.35      0.72    -1.73     1.09 1.00     1111     1978
# trial              -5.31      0.57    -6.49    -4.29 1.00     1164     1897
# block              -0.07      0.32    -0.70     0.55 1.00     1947     2577
# nervous:cond       -0.25      0.82    -1.87     1.38 1.00     1252     2104
# nervous:tp         -0.26      0.87    -1.97     1.46 1.00     1602     2355
# cond:tp            -0.92      1.22    -3.27     1.50 1.00     1581     2028
# nervous:cond:tp    -0.94      1.66    -4.31     2.32 1.00     2117     2339

save(model, file ="Study2/nervousByInterv.Rda")



######## mediation analysis ###########
# source: https://en.wikipedia.org/wiki/Mediation_(statistics)
# nuo stands for number of unique options
NUO <- ddply(Master, ~ID+tp+cond, unique = meann(Master$unique))

step1 <- brm(unique ~ tp * cond,
             data = NUO,
             cores = 2,
             chains = 2,
             iter = 4000)


df <- Master[Master$block != 1 & Master$block != 3 & Master$block != 5 & Master$block != 8 & Master$block != 10, ]
df$trial <- scale(df$trial)
df$block <- scale(df$block)
## add nervous

df$nervous <- rep(scale(as.numeric(nervous$nervous)), each = 26)

step2 <- brm(nervous ~  cond * tp + trial + block + (tp + trial + block | ID),
             data = df,
             cores = 2,
             chains = 2,
             iter = 4000)

step3 <-  brm(unique ~ nervous * cond * tp + trial + block + (nervous*tp + trial + block | ID),
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


nervous$nervous <- scale(as.numeric(nervous$nervous), center = T, scale = F)
nervous$tp <- ifelse(nervous$tp == "Pre", -0.5, 0.5)
nervous$cond <- ifelse(nervous$cond == 0, -0.5, 0.5)

nerv <- ddply(nervous, ~ID+cond+tp, summarise, nervous = meann(as.numeric(nervous)))

estims$nervous <- nerv$nervous[match(paste(estims$ID, estims$tp), paste(nerv$ID, nerv$tp))]

step1 <- brm(beta ~ tp * cond + (1|ID),
             data = estims,
             cores = 2,
             chains = 2,
             iter = 4000)

step2 <- brm(nervous ~  cond * tp + (1 | ID),
             data = estims,
             cores = 2,
             chains = 2,
             iter = 4000)

step3 <-  brm(beta ~ nervous * cond * tp + (1 | ID),
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

nerv <- ddply(nervous, ~ID+cond+tp, summarise, nervous = meann(as.numeric(nervous)))

estims$nervous <- nerv$nervous[match(paste(estims$ID, estims$tp), paste(nerv$ID, nerv$tp))]

brm(ls ~ nervous * cond * tp + (1|ID),# unable to make it converge with tp random slope
    data = estims,
    cores = 2,
    chains = 2,
    iter = 4000)

# Population-Level Effects: 
#   Estimate Est.Error l-95% CI u-95% CI Rhat Bulk_ESS Tail_ESS
# Intercept           0.02      0.04    -0.06     0.09 1.00     1174     2096
# nervous            -0.00      0.00    -0.00     0.00 1.00     2270     2771
# cond               -0.05      0.07    -0.19     0.09 1.00     1117     2091
# tp                  0.07      0.04    -0.01     0.15 1.00     3920     2788
# nervous:cond        0.00      0.00    -0.00     0.01 1.00     1497     2451
# nervous:tp          0.00      0.00    -0.00     0.01 1.00     4631     3171
# cond:tp             0.09      0.08    -0.06     0.24 1.00     4039     3239
# nervous:cond:tp     0.01      0.00     0.00     0.01 1.00     3406     2805


model <-brm(tau ~ nervous * cond * tp + (1|ID),
            data = estims,
            cores = 2,
            chains = 2,
            iter = 4000)

model

# Population-Level Effects: 
#   Estimate Est.Error l-95% CI u-95% CI Rhat Bulk_ESS Tail_ESS
# Intercept           0.00      0.00    -0.01     0.01 1.00     7655     3168
# nervous             0.00      0.00    -0.00     0.01 1.00     6356     3193
# cond               -0.00      0.01    -0.02     0.01 1.00     6310     2999
# tp                 -0.00      0.01    -0.02     0.01 1.00     6115     3035
# nervous:cond        0.00      0.01    -0.01     0.02 1.00     5719     3336
# nervous:tp          0.01      0.01    -0.01     0.02 1.00     7010     3158
# cond:tp             0.01      0.01    -0.01     0.04 1.00     5025     2948
# nervous:cond:tp     0.00      0.01    -0.03     0.03 1.00     5758     3077

model <-brm(beta ~ nervous * cond * tp + (1|ID), # the results are qualitatively the same if we take out nervousness
            data = estims,
            cores = 2,
            chains = 2,
            iter = 4000)

model
beta <- summary(model)

save(beta, file = "eta.Rda")


# Population-Level Effects: 
#   Estimate Est.Error l-95% CI u-95% CI Rhat Bulk_ESS Tail_ESS
# Intercept          -0.01      0.04    -0.10     0.08 1.00     1782     1922
# nervous             0.00      0.00    -0.00     0.00 1.00     3096     3163
# cond               -0.02      0.09    -0.19     0.15 1.00     1837     2410
# tp                 -0.16      0.05    -0.26    -0.07 1.00     4766     3149  !!
# nervous:cond       -0.00      0.00    -0.01     0.00 1.00     2720     2653
# nervous:tp         -0.00      0.00    -0.00     0.00 1.00     5273     3564
# cond:tp            -0.33      0.10    -0.52    -0.14 1.00     4380     3407  !!
# nervous:cond:tp    -0.00      0.00    -0.01     0.00 1.00     4483     3079  


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

load("Study2/intervention.Rda")

# Wilcoxon signed rank test with continuity correction

library(MASS)
wilcox.test(intervention$reclicking, intervention$reclicking2, paired=TRUE) 


