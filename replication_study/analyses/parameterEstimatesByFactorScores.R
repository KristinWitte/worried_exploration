# predicting parameter estimates of novelty bonus model from questionnaire scores
library(plyr)
library(brms)



########## loading data and prepping it ########
setwd("~/replication_study")
version <- commandArgs(TRUE)[2]
load(paste0("Master_", version,".Rda"))


df <- read.csv(paste("model_fitting/estimatesCB_n_",version,".csv", sep = ""))

factor_scores <- read.csv(paste0("fa_scores4_", version,".csv"))


df$MR1 <- scale(factor_scores$MR1[match(df$ID, factor_scores$ID)])
df$MR2 <- scale(factor_scores$MR2[match(df$ID, factor_scores$ID)])
df$MR3 <- scale(factor_scores$MR3[match(df$ID, factor_scores$ID)])
df$MR4 <- scale(factor_scores$MR4[match(df$ID, factor_scores$ID)])
df$age <- scale(as.numeric(Master$age[match(df$ID, Master$ID)]))
df$gender <- Master$Sex_0[match(df$ID, Master$ID)]
df$gender <- factor(df$gender, levels = df$gender, labels = df$gender)
df$edu <- scale(Master$edu[match(df$ID, Master$ID)])

df$kraken_present <- df$kraken_present-0.5

df$ls <- scale(df$ls, center = T, scale = F)
df$tau <- scale(df$tau, center = T, scale = F)
df$beta <- scale(df$beta, center = T, scale = F)
parameters <- c("ls", "tau", "beta")


if (!file.exists("~/replication_study/parameterEstimatesCB_n")){
  
  dir.create(file.path("~/replication_study/parameterEstimatesCB_n"))}


task_id <- as.numeric(commandArgs(TRUE)[1])
# this script was made to be run on a high-performance cluster such that all regressions
# can be run in parallel. The task_id is being passed to the script and ensures that each combination of 
# questionnaire and model parameter is being evaluated

questionnaires <- c("MR1", "MR2", "MR3", "MR4")

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

save.image(paste("~/replication_study/parameterEstimatesCB_n/",combs$p[task_id],combs$q[task_id] ,"_", version,".Rdata", sep = ""), safe = F)

