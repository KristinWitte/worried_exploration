####### distance between subsequent clicks as a function of factor scores #########

library(brms)

###### importing and formatting data #########
# run from the repo root
load("Study1/data/master.Rda")
load("Study1/data/factorScores.Rda")

# simple coding instead of dummy coding (recoding the condition (krakenPres) from 0 and 1 (safe and risky) to -0.5, 0.5)
Master$krakenPres <- Master$krakenPres - 0.5
unique(Master$krakenPres)

## remove block number 6 because that was a bonus round where participants could not always choose freely

Master <- subset(Master, blocknr != 6)

# euclidean distance between the square clicked on this trial and the square clicked on the previous trial of the same round
# (na on the first click of a round and on trials where the participant had already lost all rewards and moved on to the next round)
prev <- match(paste(Master$ID, Master$blocknr, Master$click - 1), paste(Master$ID, Master$blocknr, Master$click))
Master$dist <- sqrt((Master$x - Master$x[prev])^2 + (Master$y - Master$y[prev])^2)

# to take the reward on the current click into account we use the distance to the square clicked on the next trial
nxt <- match(paste(Master$ID, Master$blocknr, Master$click + 1), paste(Master$ID, Master$blocknr, Master$click))
Master$distFROMz <- Master$dist[nxt]

# mean-center the dependent variable
Master$distFROMz <- scale(Master$distFROMz, center = T, scale = F)

Master$anx <- factorScores$AnxDepr[match(Master$ID, factorScores$ID)]
Master$ext <- factorScores$ext[match(Master$ID, factorScores$ID)]
Master$neuro <- factorScores$neuroDev[match(Master$ID, factorScores$ID)]
Master$withdraw <- factorScores$withdraw[match(Master$ID, factorScores$ID)]

# z scale all numeric predictors first

Master[,c(2,3,6,11:20,22:24, 26:29)] <- sapply(Master[,c(2,3,6,11:20,22:24, 26:29)], function(df) scale(df, center = T, scale = T))
head(Master)


############# actual analyses #####################


# create a directory to save the results if it doesn't exist yet

if (!file.exists("Study1/analyses/distance_factor")){

  dir.create(file.path("Study1/analyses/distance_factor"))}


questionnaires <- c("anx", "ext", "neuro", "withdraw")

## get SLURM array ID
# this script is optimised to work on a high performance cluster such that
# all regressions run in parallel

task_id <- as.numeric(commandArgs(TRUE)[1])

formula <- as.formula(paste("distFROMz~", questionnaires[task_id], "*krakenPres + z *", questionnaires[task_id], " + click + blocknr + age + gender + edu + (click + blocknr + krakenPres + z| ID)"))

# fit the model we want at this array job
model <- brm(formula,
             data = Master,
             family = "gaussian",
             iter = 4000,
             cores = 2,
             chains = 2,
             control = list(adapt_delta = 0.8, max_treedepth = 15))

# give the output the name of the questionnaire we are using here
assign(paste(questionnaires[task_id]), summary(model))

# save the whole thing so we can make the visualisations locally
save.image(paste("Study1/analyses/distance_factor/",questionnaires[task_id],".Rdata", sep = ""))

