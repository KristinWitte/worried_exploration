# predicting parameter estimates of novelty bonus model from questionnaire scores
library(plyr)
library(brms)



########## loading data and prepping it ########
setwd("/Users/kristinwitte/Documents/GitHub/worried_exploration")
load("Study1/data/master.Rda")
load("Study1/data/factorScores.Rda")

df <- read.csv(paste("Study1/data/estimatesCB_n.csv", sep = ""))

# add factor scores and age and gender into df

df$anx <- factorScores$AnxDepr[match(df$ID, factorScores$ID)]
df$ext <- factorScores$ext[match(df$ID, factorScores$ID)]
df$neuro <- factorScores$neuroDev[match(df$ID, factorScores$ID)]
df$withdraw <- factorScores$withdraw[match(df$ID, factorScores$ID)]
df$age <- scale(Master$age[match(df$ID, Master$ID)])
df$gender <- Master$gender[match(df$ID, Master$ID)]
df$gender <- factor(df$gender, levels = df$gender, labels = df$gender)
df$edu <- scale(Master$edu[match(df$ID, Master$ID)])
df$kraken_present <- df$kraken_present-0.5 # effect coding

###### actual analyses #####################


# mean-center the parameter estimates

df$ls <- scale(df$ls, center = T, scale = F)
df$tau <- scale(df$tau, center = T, scale = F)
df$beta <- scale(df$beta, center = T, scale = F)
parameters <- c("ls", "tau", "beta")


# create a directory to save the results if it doesn't exist yet

if (!file.exists("~/safe_exploration/parameterEstimatesCB_n")){
  
  dir.create(file.path("~/safe_exploration/parameterEstimatesCB_n"))}


task_id <- as.numeric(commandArgs(TRUE)[1])
# this script was made to be run on a high-performance cluster such that all regressions
# can be run in parallel. The task_id is being passed to the script and ensures that each combination of 
# questionnaire and model parameter is being evaluated

questionnaires <- c("anx", "ext", "neuro", "withdraw")

combs <- data.frame(q = rep(questionnaires, length(parameters)),
                    p = rep(parameters, each = length(questionnaires)))

equation <- as.formula(paste(combs$p[task_id], "~", combs$q[task_id], "* kraken_present + age + gender + edu+ (1|ID)"))

model <- brm(equation,
             data = df,
             iter = 40000,
             cores = 4,
             chains = 4, 
             control = list(adapt_delta = 0.99))

assign(paste(combs$p[task_id], combs$q[task_id], sep = "_"), summary(model))

save.image(paste("~/safe_exploration/parameterEstimatesCB_n/",combs$p[task_id],combs$q[task_id] ,".Rdata", sep = ""), safe = F)

