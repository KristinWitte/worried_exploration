# predicting parameter estimates of novelty bonus model from questionnaire scores
library(plyr)
library(brms)



########## loading data and prepping it ########
setwd("/Users/kristinwitte/Documents/GitHub/worried_exploration")
load("Study1/master.Rda")

df <- read.csv(paste("Study1/estimatesCB_n.csv", sep = ""))

# add questionnaire scores and age and gender into df

df$STICSAcog <- scale(Master$STICSAcog[match(df$ID, Master$ID)])
df$STICSAsoma <- scale(Master$STICSAsoma[match(df$ID, Master$ID)])
df$CAPE <- scale(Master$CAPE_depressed[match(df$ID, Master$ID)])
df$IUS <- scale(Master$IUS[match(df$ID, Master$ID)])
df$RRQ <- scale(Master$RRQ[match(df$ID, Master$ID)])
df$PID5 <- scale(Master$PID5_negativeAffect[match(df$ID, Master$ID)])
df$age <- scale(Master$age[match(df$ID, Master$ID)])
df$gender <- Master$gender[match(df$ID, Master$ID)]
df$gender <- factor(df$gender, levels = df$gender, labels = df$gender)
df$kraken_present <- df$kraken_present-0.5 # effect coding

###### actual analyses #####################


# mean-center the parameter estimates

df$ls <- scale(df$ls, center = T, scale = F)
df$tau <- scale(df$tau, center = T, scale = F)
df$beta <- scale(df$beta, center = T, scale = F)
parameters <- c("ls", "tau", "beta")


task_id <- as.numeric(commandArgs(TRUE)[1])
# this script was made to be run on a high-performance cluster such that all regressions
# can be run in parallel. The task_id is being passed to the script and ensures that each combination of 
# questionnaire and model parameter is being evaluated

questionnaires <- c("STICSAcog", "STICSAsoma", "CAPE", "IUS", "RRQ", "PID5")

combs <- data.frame(q = rep(questionnaires, length(parameters)),
                    p = rep(parameters, each = length(questionnaires)))

equation <- as.formula(paste(combs$p[task_id], "~", combs$q[task_id], "* kraken_present + (1|ID)"))

model <- brm(equation,
             data = df,
             iter = 40000,
             cores = 4,
             chains = 4, 
             control = list(adapt_delta = 0.99))

assign(paste(combs$p[task_id], combs$q[task_id], sep = "_"), summary(model))

save.image(paste("~/parameterEstimatesCB_n/",combs$p[task_id],combs$q[task_id] ,".Rdata", sep = ""), safe = F)

