############ Comparing the probability of selecting a novel option in the safe vs the risky condition ##########

library(brms)
setwd("/Users/kristinwitte/Documents/GitHub/worried_exploration")
load("Study1/master.Rda")


########## getting the data ready ######

# simple coding instead of dummy coding (recoding the condition (krakenPres) from 0 and 1 (safe and risky) to -0.5, 0.5)
Master$krakenPres <- Master$krakenPres - 0.5
unique(Master$krakenPres)

# create variable that encodes whether a unique (never before selected) option was selected
Master$unique<-ave(paste(Master$x, Master$y), paste(Master$ID, 'x', Master$blocknr), FUN=duplicated)
Master$unique<-ifelse(Master$unique==TRUE, 0, 1)

# set unique to na on trials where the participant had already lost all rewards and moved on to the next round 
# (and therefore rewards (z) are na)
Master$unique[is.na(Master$z)] <- NA

## remove block number 6 because that was a bonus round where participants could not always choose freely

Master <- subset(Master, blocknr != 6)

########### regression #############
brm(unique ~ krakenPres, 
    data = Master, 
    family = "bernoulli",
    cores = 2,
    chains = 2,
    iter = 2000)
