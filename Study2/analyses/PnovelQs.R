####### probability of selecting a novel option as a function of questionnaire scores #########

library(brms)

###### importing and formatting data #########

load("Study2/data/Master.Rda")
load("Study2/data/questionnairesPre.Rda")

# simple coding instead of dummy coding (recoding the condition (krakenPres) from 0 and 1 (safe and risky) to -0.5, 0.5)
Master$cond <- ifelse(Master$cond == "control", -0.5, 0.5)
unique(Master$cond)

Master$tp <- ifelse(Master$tp == "pre", -0.5, 0.5)
unique(Master$tp)

# create variable that encodes whether a unique (never before selected) option was selected
Master$unique<-ave(paste(Master$x, Master$y), paste(Master$ID, 'x', Master$block), FUN=duplicated)
Master$unique<-ifelse(Master$unique==TRUE, 0, 1)

# set unique to na on trials where the participant had already lost all rewards and moved on to the next round 
# (and therefore rewards (z) are na)
Master$unique[is.na(Master$z)] <- NA

## remove first round which was extra practice

Master <- subset(Master, block > 1)

Master$prev_z <- Master$z[match(paste(Master$ID, Master$block, Master$trial-1), 
                                paste(Master$ID, Master$block, Master$trial))]

# get age and sex
Master$age <- questionnairesPre$age[match(Master$ID, questionnairesPre$ID)]
Master$gender <- questionnairesPre$Sex_0[match(Master$ID, questionnairesPre$ID)]

# z scale all numeric predictors first

Master[,c(2,3,6,12:25,27:28)] <- sapply(Master[,c(2,3,6,12:25,27:28)], function(df) scale(df, center = T, scale = T))
head(Master)


############# actual analyses #####################


# create a directory to save the results if it doesn't exist yet

if (!file.exists("Study2/analyses/NUOz")){
  
  dir.create(file.path("Study2/analyses/NUOz"))}


questionnaires <- c("STICSAcog", "STICSAsoma", "MCQ", "MCQpos", "MCQneg", "MCQconf",
                    "MCQcontrol", "MCQselfcons", "MWQbelief", "MWQbelief", "CASpre", "CASpost", "CASchange", "PHQ")

## get SLURM array ID
# this script is optimised to work on a high performance cluster such that 
# all regressions run in parallel

task_id <- as.numeric(commandArgs(TRUE)[1])


formula <- as.formula(paste("unique~", questionnaires[task_id], "*cond *tp + prev_z *", questionnaires[task_id],
                            "+cond*tp *prev_z + trial + block + age + gender + (tp * prev_z+ trial + block | ID)"))

# fit the model we want at this array job
model <- brm(formula, 
             data = Master,
             family = "bernoulli",
             iter = 4000,
             cores = 4,
             chains = 4, 
             control = list(adapt_delta = 0.99, max_treedepth = 15))

# give the output the name of the questionnaire we are using here
assign(paste(questionnaires[task_id]), summary(model))

# save the whole thing so we can make the visualisations locally
save.image(paste("Study2/analyses/NUOz/",questionnaires[task_id],".Rdata", sep = ""))


