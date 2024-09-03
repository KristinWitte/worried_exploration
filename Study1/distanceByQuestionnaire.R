######### Distance by questionnaires #############


####### probability of selecting a novel option as a function of questionnaire scores #########

library(brms)

###### importing and formatting data #########
setwd("/Users/kristinwitte/Documents/GitHub/worried_exploration")
load("Study1/master.Rda")


# simple coding instead of dummy coding (recoding the condition (krakenPres) from 0 and 1 (safe and risky) to -0.5, 0.5)
Master$krakenPres <- Master$krakenPres - 0.5
unique(Master$krakenPres)


## remove block number 6 because that was a bonus round where participants could not always choose freely

Master <- subset(Master, blocknr != 6)


# to take the reward on the previous click into account we shift unique up one row
Master$distanceFROMz <- c(Master$distance[2:nrow(Master)], NA)
Master$distanceFROMz[Master$click == 11] <- NA


# z scale all numeric predictors first

Master[,c(2,3,6,11:20)] <- sapply(Master[,c(2,3,6,11:20)], function(df) scale(df, center = T, scale = T))
head(Master)


############# actual analyses #####################


# create a directory to save the results if it doesn't exist yet

if (!file.exists("~/safe_exploration/distance")){
  
  dir.create(file.path("~/safe_exploration/distance"))}


questionnaires <- c("STICSAcog", "STICSAsoma", "RRQ", "IUS", "CAPE_depressed", 
                    "PID5_negativeAffect")

## get SLURM array ID
# this script is optimised to work on a high performance cluster such that 
# all regressions run in parallel

task_id <- as.numeric(commandArgs(TRUE)[1])

formula <- as.formula(paste("distanceFROMz~", questionnaires[task_id], "*krakenPres + z *", questionnaires[task_id], "+ click + blocknr + (click + blocknr + krakenPres + z| ID)"))

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
save.image(paste("~/safe_exploration/distance/",questionnaires[task_id],".Rdata", sep = ""))


