####### probability of selecting a novel option as a function of questionnaire scores #########

library(brms)

###### importing and formatting data #########

# this was optimised to run on a HPC where the version 'strict' vs "loose" is supplied as a command argument
version <- commandArgs(TRUE)[2]
load(paste0("replication_study/data/Master_",version,".Rda"))


# simple coding instead of dummy coding (recoding the condition (krakenPres) from 0 and 1 (safe and risky) to -0.5, 0.5)
Master$krakenPresent <- Master$krakenPresent - 0.5
unique(Master$krakenPresent)

# create variable that encodes whether a unique (never before selected) option was selected
Master$unique<-ave(paste(Master$x, Master$y), paste(Master$ID, 'x', Master$block), FUN=duplicated)
Master$unique<-ifelse(Master$unique==TRUE, 0, 1)

# set unique to na on trials where the participant had already lost all rewards and moved on to the next round 
# (and therefore rewards (z) are na)
Master$unique[is.na(Master$z)] <- NA


# to take the reward on the previous click into account we shift unique up one row
Master$uniqueFROMz <- c(Master$unique[2:nrow(Master)], NA)
Master$uniqueFROMz[Master$trial == 11] <- NA


# z scale all numeric predictors first

Master[,c(2,3,6,11:19)] <- sapply(Master[,c(2,3,6,11:19)], function(df) scale(as.numeric(df), center = T, scale = T))
head(Master)


############# actual analyses #####################


# create a directory to save the results if it doesn't exist yet

if (!file.exists("replication_study/analyses/NUOz")){
  
  dir.create(file.path("replication_study/analyses/NUOz"))}


questionnaires <- c("CAPE", "IUS", "RRQ", "STICA_T_c", 
                    "STICA_T_s", "PID", "PSWQ")

## get SLURM array ID
# this script is optimised to work on a high performance cluster such that 
# all regressions run in parallel

task_id <- as.numeric(commandArgs(TRUE)[1])

formula <- as.formula(paste("uniqueFROMz~", questionnaires[task_id], "*krakenPresent + z *", questionnaires[task_id], 
                            "+ trial + block + age + as.factor(Sex_0) + edu + (trial + block + krakenPresent + z| ID)"))

# fit the model we want at this array job
model <- brm(formula, 
             data = Master,
             family = "bernoulli",
             iter = 4000,
             cores = 4,
             chains = 4, 
             control = list(adapt_delta = 0.90, max_treedepth = 15))

# give the output the name of the questionnaire we are using here
assign(paste(questionnaires[task_id]), summary(model))

# save the whole thing so we can make the visualisations locally
save.image(paste("replication_study/analyses/NUOz/",questionnaires[task_id], version,".Rdata", sep = ""))


