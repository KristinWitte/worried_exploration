############## power analysis ################

library(brms)
library(dplyr)

########## loading data and prepping it ########

load("~/safe_exploration/master_perfect_attention.Rda")
df <- read.csv("~/safe_exploration/estimatesCB_n.csv")
df <- subset(df, is.element(df$ID, Master$ID))
# add questionnaire scores and age and gender into df

df$STICSAcog <- scale(Master$STICSAcog[match(df$ID, Master$ID)])
df$STICSAsoma <- scale(Master$STICSAsoma[match(df$ID, Master$ID)])
df$CAPE <- scale(Master$CAPE_depressed[match(df$ID, Master$ID)])
df$IUS <- scale(Master$IUS[match(df$ID, Master$ID)])
df$RRQ <- scale(Master$RRQ[match(df$ID, Master$ID)])
df$PID5 <- scale(Master$PID5_negativeAffect[match(df$ID, Master$ID)])
df$age <- scale(Master$age[match(df$ID, Master$ID)])
df$gender <- Master$gender[match(df$ID, Master$ID)]
df$edu <- scale(Master$edu[match(df$ID, Master$ID)])
df$kraken_present <- df$kraken_present-0.5 # effect coding

###### actual analyses #####################


# mean-center the parameter estimates

df$ls <- scale(df$ls, center = T, scale = T)
df$tau <- scale(df$tau, center = T, scale = T)
df$beta <- scale(df$beta, center = T, scale = T)
parameters <- c("ls", "tau", "beta")

print(head(df))

# create a directory to save the results if it doesn't exist yet

if (!file.exists("~/safe_exploration/power_param")){
  
  dir.create(file.path("~/safe_exploration/power_param"))}


task_id <- as.numeric(commandArgs(TRUE)[1])

# 2. Function to generate data and refit
run_sim <- function(i, original_model, original_data, parameter) {
  
  sim_data <- original_data
  vars_needed <- c(parameter, "STICSAcog", "kraken_present", "ID", "age", "gender", "edu")
  sim_data <- na.omit(sim_data[, vars_needed])
  target_cols <- c("STICSAcog", "STICSAcog:kraken_present", "kraken_present")
  
  # uses STICSAcog as a stand-in for any questionnaire bc the required power is the same no matter the actual questionnaire
  
  # 1. Get the Design Matrix (X)
  formula <- as.formula(paste0(parameter," ~ STICSAcog * kraken_present + age + gender + edu"))
  X <- model.matrix(formula, data = sim_data)
  
  # 2. Get the original coefficients (betas)
  betas <- fixef(original_model)[,1]
  
  # 3. MANUALLY SET THE EFFECT SIZE
  for (effect in target_cols){
    if (counts <- nchar(effect) - nchar(gsub(":", "", effect)) == 2){# 3-way interaction
      betas[effect] <- 0.02
    } else if (grepl(":", effect)){# 2-way interaction
      betas[effect] <- 0.05
    } else { # continuous variable
      betas[effect] <- 0.1
    }
  }
  print(betas)
  
  
  # 4. Calculate the Linear Predictor (eta)
  # This combines your forced effect size with the other original estimates
  # 1. Generate one random value per unique ID
  ranef_sd <- VarCorr(original_model)$ID$sd["Intercept", "Estimate"]
  unique_ids <- unique(sim_data$ID)
  id_noise <- rnorm(length(unique_ids), mean = 0, sd = ranef_sd)
  # This creates a vector the same length as sim_data
  mapped_noise <- id_noise[match(sim_data$ID, unique_ids)]
  eta <- as.numeric(X %*% betas) + mapped_noise
  
  sigma_val <- summary(original_model)$spec_pars["sigma", "Estimate"]
  

  sim_data[ ,grep(parameter, colnames(sim_data))] <- rnorm(nrow(sim_data), mean = eta, sd = sigma_val)
  
  # 7. Refit and check if the CI for that specific term excludes zero
  sim_fit <- update(original_model, newdata = sim_data, 
                    iter = 4000, 
                    chains = 4, 
                    cores = 4,
                    control = list(adapt_delta = 0.95))
  success <- c()
  for (target_col in c("STICSAcog", "STICSAcog:kraken_present", "kraken_present")){
    ci <- posterior_interval(sim_fit, variable = paste0("b_", target_col))
    success <- c(success, ci[1,1] > 0 | ci[1,2] < 0)
  }
  print(sim_fit)
  
  return(success)
}


for (param in parameters){
  load(paste0("~/safe_exploration/brm_", param,"_Q.Rda"))
  power_results <- run_sim(i = task_id, original_model = model, original_data  = df, parameter = param) 
  
}



save(power_results, file = paste0("~/safe_exploration/power_param/Q_", toString(task_id), ".Rda"))