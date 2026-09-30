############## power analysis ################

library(brms)
library(dplyr)

which_study <- "Study1"

################# load data #############

load(paste0(which_study,"/data/master.Rda"))

# 1. Define the "Effect Size" you want to test
# Since it's logistic, this is in Log-Odds. 
# 0.4 is roughly a "small-to-medium" effect (Odds Ratio ≈ 1.5)
target_effect_q <- 0.4
target_effect_kr <- 0.3
target_effect_inter <- 0.2

##### data preprocessing ########

if (which_study == "Study1"){
  Master$krakenPres <- Master$krakenPres - 0.5
  
  # create variable that encodes whether a unique (never before selected) option was selected
  Master$unique<-ave(paste(Master$x, Master$y), paste(Master$ID, 'x', Master$blocknr), FUN=duplicated)
  Master$unique<-ifelse(Master$unique==TRUE, 0, 1)
  
  # set unique to na on trials where the participant had already lost all rewards and moved on to the next round 
  # (and therefore rewards (z) are na)
  Master$unique[is.na(Master$z)] <- NA
  
  ## remove block number 6 because that was a bonus round where participants could not always choose freely
  
  Master <- subset(Master, blocknr != 6)
  
  Master$unique[is.na(Master$z)] <- NA
  
  # to take the reward on the previous click into account we shift unique up one row
  Master$uniqueFROMz <- c(Master$unique[2:nrow(Master)], NA)
  Master$uniqueFROMz[Master$click == 11] <- NA
  
  # z scale all numeric predictors first
  
  Master[,c(2,3,6,11:20,22)] <- sapply(Master[,c(2,3,6,11:20,22)], function(df) scale(df, center = T, scale = T))
  
  # unify variable naming at least a bit
  
  Master$dv <- Master$uniqueFROMz
  
} else if (which_study == "Study2"){
  
  
  
  # simple coding instead of dummy coding (recoding the condition (krakenPres) from 0 and 1 (safe and risky) to -0.5, 0.5)
  Master$cond <- ifelse(Master$cond == "control", -0.5, 0.5)
  
  Master$tp <- ifelse(Master$tp == "pre", -0.5, 0.5)
  
  # set unique to na on trials where the participant had already lost all rewards and moved on to the next round 
  # (and therefore rewards (z) are na)
  Master$unique[is.na(Master$z)] <- NA
  
  ## remove first round which was extra practice
  
  Master <- subset(Master, block > 1)
  
  Master$prev_z <- Master$z[match(paste(Master$ID, Master$block, Master$trial-1), 
                                  paste(Master$ID, Master$block, Master$trial))]
  
  # get age and sex
  load("Study2/data/questionnairesPre.Rda")
  Master$age <- questionnairesPre$age[match(Master$ID, questionnairesPre$ID)]
  Master$gender <- questionnairesPre$Sex_0[match(Master$ID, questionnairesPre$ID)]
  
  # z scale all numeric predictors first
  
  Master[,c(2,3,6,12:25,27:28)] <- sapply(Master[,c(2,3,6,12:25,27:28)], function(df) scale(df, center = T, scale = T))
  
  Master$dv <- Master$unique
  
} else if (which_study == "replication_study") {
  
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
  
  
  
}




############## main simulation ###################
run_sim <- function(i, original_model, original_data, target_effects) {
  
  sim_data <- original_data
  if(which_study == "Study1"){
    vars_needed <- c("dv", "STICSAcog", "krakenPres", "z", "click", "blocknr", "ID", "uniqueFROMz")
    formula <- as.formula("uniqueFROMz ~ STICSAcog * krakenPres + z * STICSAcog + click + blocknr")
    target_cols <- c("STICSAcog", "STICSAcog:krakenPres", "krakenPres")
  } else if (which_study == "Study2"){
    vars_needed <- c("dv", "STICSAcog", "tp", "cond", "prev_z", "trial", "block", "ID", "age", "gender", "unique")
    formula <- as.formula("unique ~ STICSAcog * cond * tp + prev_z * STICSAcog + cond * tp * prev_z + trial + block + age + gender")
    target_cols <- c("STICSAcog", "STICSAcog:cond", "STICSAcog:tp", "STICSAcog:cond:tp", "cond:tp", "cond", "tp")
  } else if (which_study == "replication_study") {
    vars_needed <- c("uniqueFROMz", "STICA_T_c", "krakenPresent", "z", "trial", "block", "ID", "age", "Sex_0", "edu")
    formula <- as.formula("uniqueFROMz ~ STICA_T_c * krakenPresent + z * STICA_T_c + trial + block + age + as.factor(Sex_0) + edu ")
    target_cols <- c("STICA_T_c", "STICA_T_c:krakenPresent", "krakenPresent")
    
  }
  
  sim_data <- na.omit(sim_data[, vars_needed])
  
  # uses STICSAcog as a stand-in for any questionnaire bc the required power is the same no matter the actual questionnaire
  
  # 1. Get the Design Matrix (X)
  
  X <- model.matrix(formula, data = sim_data)
  
  # 2. Get the original coefficients (betas)
  betas <- fixef(original_model)[,1]
  
  # 3. MANUALLY SET THE EFFECT SIZE
  for (effect in target_cols){
    if (counts <- nchar(effect) - nchar(gsub(":", "", effect)) == 2){# 3-way interaction
      betas[effect] <- 0.1
    } else if (grepl(":", effect)){# 2-way interaction
      betas[effect] <- 0.2
    } else if (grepl("kraken", effect) | grepl("cond", effect) | grepl("tp", effect)){ # binary variable
      betas[effect] <- 0.3
    } else { # continuous variable
      betas[effect] <- 0.6
    }
  }
  print(betas)
  # target_col <- "STICSAcog" 
  # betas[target_col] <- target_effects[1]
  # target_col <- "STICSAcog:krakenPres" 
  # betas[target_col] <- target_effects[3]
  # target_col <- "krakenPres" 
  # betas[target_col] <- target_effects[2]
  
  
  # 4. Calculate the Linear Predictor (eta)
  # This combines your forced effect size with the other original estimates
  # 1. Generate one random value per unique ID
  ranef_sd <- VarCorr(original_model)$ID$sd["Intercept", "Estimate"]
  unique_ids <- unique(sim_data$ID)
  id_noise <- rnorm(length(unique_ids), mean = 0, sd = ranef_sd)
  # This creates a vector the same length as sim_data
  mapped_noise <- id_noise[match(sim_data$ID, unique_ids)]
  eta <- as.numeric(X %*% betas) + mapped_noise
  
  # 6. Convert to Probabilities and Generate Outcomes
  probs <- plogis(eta)
  sim_data$uniqueFROMz <- rbinom(nrow(sim_data), 1, probs)
  
  # 7. Refit and check if the CI for that specific term excludes zero
  sim_fit <- update(original_model, newdata = sim_data, iter = 4000, chains = 4, cores = 4)
  success <- c()
  for (target_col in target_cols){
    ci <- posterior_interval(sim_fit, variable = paste0("b_", target_col))
    success <- c(success, ci[1,1] > 0 | ci[1,2] < 0)
  }
  
  
  return(success)
}


power_results <- sapply(1:50, run_sim, original_model = model, original_data = Master, target_effect)
final_power <- mean(power_results)



############## import and analyse results from that was running on the HPC #################
which_study = "replication_study"
n_jobs = 100

target_cols <- c("STICA_T_c", "STICA_T_c:krakenPresent", "krakenPresent")
effect_sizes <- c(0.2,0.3,0.4)

powers <- data.frame(effect_size = effect_sizes,
                     STICSA = NA,
                     STICSAxCondition = NA,
                     Condition = NA)

for (effect_size in effect_sizes){
  files <- list.files(paste0(which_study, "/analyses/power_",effect_size))
  print(effect_size)
  
  if(exists("outcomes")) {rm(outcomes)}
  
  for (i in 1:n_jobs){
    file <- paste0("NUO_Q_",i,".Rda") 
    
    if (file %in% files) {
      load(paste0(which_study,"/analyses/power_", effect_size,"/",file))
      
      if (!exists("outcomes")) {
        outcomes <- data.frame(matrix(nrow = n_jobs, ncol = (length(target_cols))))
        colnames(outcomes) <- c(target_cols)
        outcomes[i,1:length(target_cols)] <- power_results
        
      } else {
        
        outcomes[i,1:length(target_cols)] <- power_results
        
      }
      
      
      
    } else {# missing file
      print(i)
      
    }
    
  }
  powers[powers$effect_size == effect_size,2:ncol(powers)] <- colMeans(outcomes, na.rm = T)
  
  
  
}

library(stargazer)

stargazer(powers, summary = F, rownames = F, digits = 2)
######### for model parameter regressions ############

n_jobs = 500

which_study = "replication_study"

target_cols <- c("STICSAcog", "STICSAcog:kraken_present", "kraken_present")

effect_sizes <- c(0.1,0.15,0.2)

powers <- data.frame(parameter = rep(c("ls", "tau", "beta"), each = length(effect_sizes)),
                     effect_size = rep(effect_sizes, 3),
                     STICSA = NA,
                     STICSAxCondition = NA,
                     Condition = NA) 

for (effect_size in effect_sizes){
  files <- list.files(paste0(which_study, "/analyses/power_param", effect_size))
  
  if(exists("outcomes")) {rm(outcomes)}
  
  for (i in 1:n_jobs) {
    file <- paste0("Q_", i, ".Rda")
    
    if (file %in% files) {
      load(paste0(which_study, "/analyses/power_param",effect_size,"/", file))
      
      if (!exists("outcomes")) {
        outcomes <- list()
        for (param in c("ls", "tau", "beta")) {
          outcomes[[param]] <- data.frame(matrix(nrow = n_jobs, ncol = length(target_cols)))
          colnames(outcomes[[param]]) <- target_cols
          
        }
        
      }
      for (param in c("ls", "tau", "beta")) {
        outcomes[[param]][i, ] <- power_results[[param]]
        
      }
      
    } else {  # missing file
      print(i)
      
    }
    
  }
  
  for (param in c("ls", "tau", "beta")) {
    print(colMeans(outcomes[[param]], na.rm = T))
    powers[powers$parameter == param & powers$effect_size == effect_size, 3:ncol(powers)] <- colMeans(outcomes[[param]], na.rm = T)
    
  } 
  
  
  
}

print(powers)

powers %>% 
  mutate(parameter = recode(parameter, "ls" = "$lambda$",
                            "tau" = "$tau$",
                            "beta" = "$eta$")) %>% 
  stargazer(summary = F, rownames = F, digits = 2)

