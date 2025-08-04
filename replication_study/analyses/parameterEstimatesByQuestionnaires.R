# predicting parameter estimates of novelty bonus model from questionnaire scores
library(plyr)
library(brms)



########## loading data and prepping it ########
setwd("~/replication_study")
version <- commandArgs(TRUE)[2]
load(paste0("Master_",version,".Rda"))

df <- read.csv(paste("model_fitting/estimatesCB_n.csv", sep = ""))

# add questionnaire scores and age and gender into df

df$STICSAcog <- scale(Master$STICA_T_c[match(df$ID, Master$ID)])
df$STICSAsoma <- scale(Master$STICA_T_s[match(df$ID, Master$ID)])
df$CAPE <- scale(Master$CAPE[match(df$ID, Master$ID)])
df$IUS <- scale(Master$IUS[match(df$ID, Master$ID)])
df$RRQ <- scale(Master$RRQ[match(df$ID, Master$ID)])
df$PID5 <- scale(Master$PID[match(df$ID, Master$ID)])
df$PSWQ <- scale(Master$PSWQ[match(df$ID, Master$ID)])
df$age <- scale(as.numeric(Master$age[match(df$ID, Master$ID)]))
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

if (!file.exists("~/replication_study/parameterEstimatesCB_n")){
  
  dir.create(file.path("~/replication_study/parameterEstimatesCB_n"))}


task_id <- as.numeric(commandArgs(TRUE)[1])
# this script was made to be run on a high-performance cluster such that all regressions
# can be run in parallel. The task_id is being passed to the script and ensures that each combination of 
# questionnaire and model parameter is being evaluated

questionnaires <- c("STICSAcog", "STICSAsoma", "CAPE", "IUS", "RRQ", "PID5", "PSWQ")

combs <- data.frame(q = rep(questionnaires, length(parameters)),
                    p = rep(parameters, each = length(questionnaires)))

equation <- as.formula(paste(combs$p[task_id], "~", combs$q[task_id], "* kraken_present + age + gender + edu+ (1|ID)"))

model <- brm(equation,
             data = df,
             iter = 4000,
             cores = 4,
             chains = 4, 
             control = list(adapt_delta = 0.90))

assign(paste(combs$p[task_id], combs$q[task_id], sep = "_"), summary(model))

save.image(paste("~/replication_study/parameterEstimatesCB_n/",combs$p[task_id],combs$q[task_id] ,version,".Rdata", sep = ""), safe = F)

