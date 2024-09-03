############# save output of regressions as one Rda file to upload to github and use for plotting #########

library(here)

files <- list.files("distance")

for (file in files){
  load(paste0("distance/", file))
  }

save(CAPE_depressed, IUS, PID5_negativeAffect, RRQ, STICSAcog, STICSAsoma, file = "Study1/distanceQs.Rda")
