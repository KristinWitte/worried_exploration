############ Comparing the distance between subsequent clicks in the safe vs the risky condition ##########

library(brms)
# run from the repo root
load("Study1/data/master.Rda")


########## getting the data ready ######

# simple coding instead of dummy coding (recoding the condition (krakenPres) from 0 and 1 (safe and risky) to -0.5, 0.5)
Master$krakenPres <- Master$krakenPres - 0.5
unique(Master$krakenPres)

## remove block number 6 because that was a bonus round where participants could not always choose freely

Master <- subset(Master, blocknr != 6)

# euclidean distance between the square clicked on this trial and the square clicked on the previous trial of the same round
# (na on the first click of a round and on trials where the participant had already lost all rewards and moved on to the next round)
prev <- match(paste(Master$ID, Master$blocknr, Master$click - 1), paste(Master$ID, Master$blocknr, Master$click))
Master$dist <- sqrt((Master$x - Master$x[prev])^2 + (Master$y - Master$y[prev])^2)

# mean-center the dependent variable
Master$dist <- scale(Master$dist, center = T, scale = F)

########### regression #############
model <- brm(dist ~ krakenPres,
             data = Master,
             family = "gaussian",
             cores = 2,
             chains = 2,
             iter = 2000)

summary(model)

dist_cond <- summary(model)
save(dist_cond, file = "Study1/analyses/distanceByCond.Rda")
