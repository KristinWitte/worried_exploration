################ SI plots and tables ###################
rm(list = ls())

library(ggplot2)
theme_set(theme_classic(base_size = 15))
library(ggpubr)
library(RColorBrewer)
library(plyr)
library(gghalves)
library(knitr)
library(docstring)
library(here)
library(tidyverse)

here()

############## get data ##################
load("Study1/data/master.Rda")
Master1 <- Master
Master1$krakenPres<- factor(Master1$krakenPres, levels = c(0,1), labels = c("safe", "risky"))

load("Study2/data/Master.Rda")
load("Study2/data/nervous.Rda")

Master$row <- 1:nrow(Master)
nervous$round <- rep(c(2, 4, 6, 7, 9, 11), nrow(nervous)/6)
nervous$nervous <- as.numeric(nervous$nervous)
nervous$cond <- factor(nervous$cond, levels = c(0,1), labels = c("control", "intervention"))

Master <- subset(Master, block > 1)

# get NUO variable
Master$unique<-ave(paste(Master$x, Master$y), paste(Master$ID, 'x', Master$block), FUN=duplicated)
Master$unique<-ifelse(Master$unique==TRUE, 0, 1)
Master$unique[is.na(Master$z)] <- NA

Master1$unique<-ave(paste(Master1$x, Master1$y), paste(Master1$ID, 'x', Master1$blocknr), FUN=duplicated)
Master1$unique<-ifelse(Master1$unique==TRUE, 0, 1)
Master1$unique[is.na(Master1$z)] <- NA

## set colours
red <- brewer.pal(12,"Paired")[6]
darkBlue <-  brewer.pal(12,"Paired")[2]
control <- "#E39189"

################ functions ##############

errorBarPlot <- function(df, title = waiver(), xlabel = expression(beta~"-Coefficients with 95%HDI"), ylabel = element_blank()){
  #' visualising a range of effects as errorbars
  #' @param dataset data.frame, has to have specific format and variable naming
  #' @param title str (optional)
  #' @param xlabel str, label of x axis (optional), defaults to "beta coefficients with 95%HDI
  #' @param ylabel str (optional), defaults to element_blank
  #' @return ggplot object
  
  df$var = factor(df$var, levels = df$var, labels = df$var)
  # make the ones that don't overlap with 0 heavier
  df$size <- ifelse(sign(df$upper) == sign(df$lower), 1, 0)
  df$size <- factor(df$size, levels = df$size, labels = df$size)
  
  
  p2 <- ggplot(df, aes(x = Estimate, y = var), color = "black") + 
    geom_point(size = 3) + 
    geom_errorbar(aes(xmin = lower, xmax = upper, size = size), width = 0.5) +
    theme(legend.position = "none") +
    geom_vline(xintercept = 0) + 
    labs(title = title, 
         x = xlabel, 
         y = ylabel)+
    scale_size_manual(breaks = c(0,1), values = c(1, 1.8))
  
  return(p2)
  
}

se<-function(x){sd(x, na.rm = T)/sqrt(length(na.omit(x)))}
meann <- function(x){mean(x, na.rm = T)}

heatmap <- function(df, x = x, y = y, limits = c(-1,1)){
  
  ggplot(df, aes(x = x, y = y, fill = cor)) + geom_raster() + 
    scale_fill_gradient2(high = darkBlue, low = red, mid = "white", limits = limits)+
    geom_label(aes(label = round(cor, digits = 2)), fill = "white") +
    scale_x_discrete(expand = c(0.01, 0)) +
    scale_y_discrete(expand = c(0.01, 0))

}

self_cor <- function(df){
  cors <- df %>% 
    select(-ID) %>% 
    cor(use = "pairwise.complete.obs") %>% 
    as.data.frame() %>% 
    mutate(x = rownames(.)) %>% 
    pivot_longer(cols = -x, names_to = "y", values_to = "cor")
  
  return(cors)
  
}



#################### Tables for SI ################

#### this code only provides a general formatting and the data for the latex tables, 
# but we still had to do some manual prettifying


######### St1 NUO ~Q
library(stargazer)
load("Study1/analyses/NUOQs.Rda")

main <- rbind(STICSAcog$fixed[rownames(STICSAcog$fixed) == "STICSAcog", ],
              STICSAsoma$fixed[rownames(STICSAsoma$fixed) == "STICSAsoma", ], 
              CAPE_depressed$fixed[rownames(CAPE_depressed$fixed) == "CAPE_depressed", ], 
              IUS$fixed[rownames(IUS$fixed) == "IUS", ], 
              RRQ$fixed[rownames(RRQ$fixed) == "RRQ", ], 
              PID5_negativeAffect$fixed[rownames(PID5_negativeAffect$fixed) == "PID5_negativeAffect", ])

df <- data.frame(var = c("cognitive anxiety", "somatic anxiety", "depressivity", "intolerance to uncertainty", "rumination", "negative affect"), Estimate = main[ ,1], lower = main[ ,3], upper = main[ ,4])
df$var = factor(df$var, levels = df$var, labels = df$var)

stargazer(df, type = "latex", summary = F, rownames = F, column.labels  = c("Predictor","beta", 
                                                                            "95HDI lower bound", "95HDI upper bound"))

interact <- rbind(STICSAcog$fixed[rownames(STICSAcog$fixed) == "STICSAcog:krakenPres", ],
                  STICSAsoma$fixed[rownames(STICSAsoma$fixed) == "STICSAsoma:krakenPres", ], 
                  CAPE_depressed$fixed[rownames(CAPE_depressed$fixed) == "CAPE_depressed:krakenPres", ], 
                  IUS$fixed[rownames(IUS$fixed) == "IUS:krakenPres", ], 
                  RRQ$fixed[rownames(RRQ$fixed) == "RRQ:krakenPres", ], 
                  PID5_negativeAffect$fixed[rownames(PID5_negativeAffect$fixed) == "PID5_negativeAffect:krakenPres", ])
df <- data.frame(var = c("cognitive anxiety", "somatic anxiety", "depressivity", "intolerance to uncertainty", "rumination", "negative affect"), 
                 Estimate = interact[ ,1], lower = interact[ ,3], upper = interact[ ,4])
df$var = factor(df$var, levels = df$var, labels = df$var)

stargazer(df, type = "latex", summary = F, rownames = F, column.labels  = c("Predictor","beta", 
                                                                            "95HDI lower bound", "95HDI upper bound"))


############## Study 1 eta ~ Qs 

load("Study1/analyses/etaQs.Rda")

View(beta_PID5$fixed)

main <- rbind(beta_STICSAcog$fixed[rownames(beta_STICSAcog$fixed) == "STICSAcog", ],
              beta_STICSAsoma$fixed[rownames(beta_STICSAsoma$fixed) == "STICSAsoma", ], 
              beta_CAPE$fixed[rownames(beta_CAPE$fixed) == "CAPE", ], 
              beta_IUS$fixed[rownames(beta_IUS$fixed) == "IUS", ], 
              beta_RRQ$fixed[rownames(beta_RRQ$fixed) == "RRQ", ], 
              beta_PID5$fixed[rownames(beta_PID5$fixed) == "PID5", ])
df <- data.frame(var = c("cognitive anxiety", "somatic anxiety", "depressivity", "intolerance to uncertainty", "rumination", "negative affect"), Estimate = main[ ,1], lower = main[ ,3], upper = main[ ,4])
df$var = factor(df$var, levels = df$var, labels = df$var)

stargazer(df, type = "latex", summary = F, rownames = F, column.labels  = c("Predictor","beta", 
                                                                            "95HDI lower bound", "95HDI upper bound"))


interact <- rbind(beta_STICSAcog$fixed[rownames(beta_STICSAcog$fixed) == "STICSAcog:kraken_present", ],
                  beta_STICSAsoma$fixed[rownames(beta_STICSAsoma$fixed) == "STICSAsoma:kraken_present", ], 
                  beta_CAPE$fixed[rownames(beta_CAPE$fixed) == "CAPE:kraken_present", ], 
                  beta_IUS$fixed[rownames(beta_IUS$fixed) == "IUS:kraken_present", ], 
                  beta_RRQ$fixed[rownames(beta_RRQ$fixed) == "RRQ:kraken_present", ], 
                  beta_PID5$fixed[rownames(beta_PID5$fixed) == "PID5:kraken_present", ])
df <- data.frame(var = c("cognitive anxiety", "somatic anxiety", "depressivity", "intolerance to uncertainty", "rumination", "negative affect"), Estimate = interact[ ,1], lower = interact[ ,3], upper = interact[ ,4])
df$var = factor(df$var, levels = df$var, labels = df$var)

stargazer(df, type = "latex", summary = F, rownames = F, column.labels  = c("Predictor","beta", 
                                                                            "95HDI lower bound", "95HDI upper bound"))


############# St1 tau ~Qs

load("Study1/analyses/tauQs.Rda")
View(tau_IUS$fixed)

main <- rbind(tau_STICSAcog$fixed[rownames(tau_STICSAcog$fixed) == "STICSAcog", ],
              tau_STICSAsoma$fixed[rownames(tau_STICSAsoma$fixed) == "STICSAsoma", ], 
              tau_CAPE$fixed[rownames(tau_CAPE$fixed) == "CAPE", ], 
              tau_IUS$fixed[rownames(tau_IUS$fixed) == "IUS", ], 
              tau_RRQ$fixed[rownames(tau_RRQ$fixed) == "RRQ", ], 
              tau_PID5$fixed[rownames(tau_PID5$fixed) == "PID5", ])
df <- data.frame(var = c("cognitive anxiety", "somatic anxiety", "depressivity", "intolerance to uncertainty", "rumination", "negative affect"), Estimate = main[ ,1], lower = main[ ,3], upper = main[ ,4])
df$var = factor(df$var, levels = df$var, labels = df$var)

stargazer(df, type = "latex", summary = F, rownames = F, column.labels  = c("Predictor","beta", 
                                                                            "95HDI lower bound", "95HDI upper bound"))

p1 <- errorBarPlot(df)

# extract interaction with condition

interact <- rbind(tau_STICSAcog$fixed[rownames(tau_STICSAcog$fixed) == "STICSAcog:kraken_present", ],
                  tau_STICSAsoma$fixed[rownames(tau_STICSAsoma$fixed) == "STICSAsoma:kraken_present", ], 
                  tau_CAPE$fixed[rownames(tau_CAPE$fixed) == "CAPE:kraken_present", ], 
                  tau_IUS$fixed[rownames(tau_IUS$fixed) == "IUS:kraken_present", ], 
                  tau_RRQ$fixed[rownames(tau_RRQ$fixed) == "RRQ:kraken_present", ], 
                  tau_PID5$fixed[rownames(tau_PID5$fixed) == "PID5:kraken_present", ])
df <- data.frame(var = c("cognitive anxiety", "somatic anxiety", "depressivity", "intolerance to uncertainty", "rumination", "negative affect"), Estimate = interact[ ,1], lower = interact[ ,3], upper = interact[ ,4])
df$var = factor(df$var, levels = df$var, labels = df$var)

stargazer(df, type = "latex", summary = F, rownames = F, column.labels  = c("Predictor","beta", 
                                                                            "95HDI lower bound", "95HDI upper bound"))

p2 <- errorBarPlot(df)

ggarrange(p1, p2)

########## St1 ls ~Qs

load("Study1/analyses/lsQs.Rda")

main <- rbind(ls_STICSAcog$fixed[rownames(ls_STICSAcog$fixed) == "STICSAcog", ],
              ls_STICSAsoma$fixed[rownames(ls_STICSAsoma$fixed) == "STICSAsoma", ], 
              ls_CAPE$fixed[rownames(ls_CAPE$fixed) == "CAPE", ], 
              ls_IUS$fixed[rownames(ls_IUS$fixed) == "IUS", ], 
              ls_RRQ$fixed[rownames(ls_RRQ$fixed) == "RRQ", ], 
              ls_PID5$fixed[rownames(ls_PID5$fixed) == "PID5", ])
df <- data.frame(var = c("cognitive anxiety", "somatic anxiety", "depressivity", "intolerance to uncertainty", "rumination", "negative affect"), 
                 Estimate = main[ ,1], lower = main[ ,3], upper = main[ ,4])
df$var = factor(df$var, levels = df$var, labels = df$var)

stargazer(df, type = "latex", summary = F, rownames = F, column.labels  = c("Predictor","beta", 
                                                                            "95HDI lower bound", "95HDI upper bound"))

p1 <- errorBarPlot(df)
# extract interaction with condition

interact <- rbind(ls_STICSAcog$fixed[rownames(ls_STICSAcog$fixed) == "STICSAcog:kraken_present", ],
                  ls_STICSAsoma$fixed[rownames(ls_STICSAsoma$fixed) == "STICSAsoma:kraken_present", ], 
                  ls_CAPE$fixed[rownames(ls_CAPE$fixed) == "CAPE:kraken_present", ], 
                  ls_IUS$fixed[rownames(ls_IUS$fixed) == "IUS:kraken_present", ], 
                  ls_RRQ$fixed[rownames(ls_RRQ$fixed) == "RRQ:kraken_present", ], 
                  ls_PID5$fixed[rownames(ls_PID5$fixed) == "PID5:kraken_present", ])
df <- data.frame(var = c("cognitive anxiety", "somatic anxiety", "depressivity", "intolerance to uncertainty", "rumination", "negative affect"), 
                 Estimate = interact[ ,1], lower = interact[ ,3], upper = interact[ ,4])
df$var = factor(df$var, levels = df$var, labels = df$var)

stargazer(df, type = "latex", summary = F, rownames = F, column.labels  = c("Predictor","beta", 
                                                                            "95HDI lower bound", "95HDI upper bound"))
p2 <- errorBarPlot(df)

ggarrange(p1, p2)


############# Study 1 factor score regression results ##########

######### P(novel)
load("Study1/analyses/NUO_factor.Rda")


main <- rbind(anx$fixed, ext$fixed, neuro$fixed, withdraw$fixed) %>% 
  subset(rownames(.) %in% c("anx", "ext","neuro","withdraw")) %>% 
  mutate(Predictor = recode(rownames(.), "anx" = "internalising",
                            "ext" = "externalising", 
                            "neuro" = "neurodevelopmental",
                            "withdraw" = "social withdrawal")) %>% 
  select(Predictor,Estimate, `l-95% CI`, `u-95% CI`)

stargazer(main, type = "latex", 
          summary = F, 
          rownames = F, 
          column.labels  = c("Predictor","beta", "95HDI lower bound", "95HDI upper bound"),
          title = paste("Main effects of factor scores on P(novel)"),
          table.placement = "H",
          label = paste("tab:St1MainFactorNUO"))

interact <- rbind(anx$fixed, ext$fixed, neuro$fixed, withdraw$fixed) %>% 
  subset(rownames(.) %in% c("anx:krakenPres", "ext:krakenPres","neuro:krakenPres","withdraw:krakenPres")) %>% 
  mutate(Predictor = recode(rownames(.), "anx:krakenPres" = "internalising*condition",
                            "ext:krakenPres" = "externalising*condition", 
                            "neuro:krakenPres" = "neurodevelopmental*condition",
                            "withdraw:krakenPres" = "social withdrawal*condition")) %>% 
  select(Predictor,Estimate, `l-95% CI`, `u-95% CI`)

stargazer(interact, type = "latex", 
          summary = F, 
          rownames = F, 
          column.labels  = c("Predictor","beta", "95HDI lower bound", "95HDI upper bound"),
          title = paste("Interaction effects of factor scores on P(novel)"),
          table.placement = "H",
          label = paste("tab:St1InteractFactorNUO"))


############ model parameters

for (param in c("eta", "ls", "tau")){
  p <- ifelse(param == "eta", "beta", param)
  
  for (i in c("anx", "withdraw", "ext", "neuro")){
    load(paste0("Study1/analyses/parameterEstimatesCB_n_perfect/", p, i, ".Rdata"))
    # these files can be obtained by executing Study1/analyses/parameterEstimatesByFactor.R

  }
  
  print(paste0("\\subsubsection*{Predicting the ", param, " parameter from factor scores in Study 1}"))
  
  anx <- get(paste(p,"anx", sep = "_"), envir = .GlobalEnv)
  ext <- get(paste(p,"ext", sep = "_"), envir = .GlobalEnv)
  neuro <- get(paste(p,"neuro", sep = "_"), envir = .GlobalEnv)
  withdraw <- get(paste(p,"withdraw", sep = "_"), envir = .GlobalEnv)
  
  
  main <- rbind(anx$fixed, ext$fixed, neuro$fixed, withdraw$fixed) %>% 
    subset(rownames(.) %in% c("anx", "ext","neuro","withdraw")) %>% 
    mutate(Predictor = recode(rownames(.), "anx" = "internalising",
                              "ext" = "externalising", 
                              "neuro" = "neurodevelopmental",
                              "withdraw" = "social withdrawal"),
           beta = Estimate,
           `95HDI lower bound` = `l-95% CI`,
           `95HDI upper bound` = `u-95% CI`) %>% 
    select(Predictor,beta, `95HDI lower bound`, `95HDI upper bound`)
  
  stargazer(main, type = "latex", 
            summary = F, 
            rownames = F, 
            column.labels  = c("Predictor","beta", "95HDI lower bound", "95HDI upper bound"),
            title = paste("Main effects of factor scores on", param),
            table.placement = "H",
            label = paste0("tab:St1MainFactor", param))
  
  interact <- rbind(anx$fixed, ext$fixed, neuro$fixed, withdraw$fixed) %>% 
    subset(rownames(.) %in% c("anx:kraken_present", "ext:kraken_present",
                              "neuro:kraken_present","withdraw:kraken_present")) %>% 
    mutate(Predictor = recode(rownames(.), "anx:kraken_present" = "internalising*condition",
                              "ext:kraken_present" = "externalising*condition", 
                              "neuro:kraken_present" = "neurodevelopmental*condition",
                              "withdraw:kraken_present" = "social withdrawal*condition"),
           beta = Estimate,
           `95HDI lower bound` = `l-95% CI`,
           `95HDI upper bound` = `u-95% CI`) %>% 
    select(Predictor,beta, `95HDI lower bound`, `95HDI upper bound`)
  
  stargazer(interact, type = "latex", 
            summary = F, 
            rownames = F, 
            column.labels  = c("Predictor","beta", "95HDI lower bound", "95HDI upper bound"),
            title = paste("Interaction effects of factor scores on", param),
            table.placement = "H",
            label = paste0("tab:St1InteractFactor", param))
  
  
}



########### Study 2 P(novel) ~Q ##########

files <- list.files("Study2/NUOz")
for (i in files){load(paste0("Study2/NUOz/",i))}
# these files can be obtained by executing Study2/analyses/PnovelQs.R

qs <- c("STICSAcog", "STICSAsoma", "MCQ", "MCQpos", "MCQneg", "MCQconf",
        "MCQcontrol", "MCQselfcons", "MWQbelief", "CASpre", "CASpost", "CASchange", "PHQ")

# Dynamically retrieve variables
selected_vars <- lapply(qs,function(x) get(x)$fixed)
names(selected_vars) <- qs
df <- selected_vars %>% 
  bind_rows(.id = "Predictor") %>% 
  mutate(Predictor = apply(as.array(rownames(.)), 1, function(x) strsplit(x, "\\.\\.\\.")[[1]][1]))

dat <- df %>% subset(is.element(Predictor, qs)) %>% select(Predictor, Estimate, `l-95% CI`, `u-95% CI`)
stargazer(dat, type = "latex", summary = F, rownames = F, 
          title = paste("Main effects of Questionnaires on P(novel)"),
          table.placement = "H",
          label = paste("tab:St2MainNUO"))

dat <- df %>% subset(grepl(":cond", Predictor)&!grepl("tp",Predictor)) %>% select(Predictor, Estimate, `l-95% CI`, `u-95% CI`)
stargazer(dat, type = "latex", summary = F, rownames = F, 
          title = paste("Interaction effects of Questionnaires with condition on P(novel)"),
          table.placement = "H",
          label = paste("tab:St2IntercondNUO"))

dat <- df %>% subset(!grepl("cond", Predictor)&grepl(":tp",Predictor)) %>% select(Predictor, Estimate, `l-95% CI`, `u-95% CI`)
stargazer(dat, type = "latex", summary = F, rownames = F, 
          title = paste("Interaction effects of Questionnaires with timpoint on P(novel)"),
          table.placement = "H",
          label = paste("tab:St2IntertpNUO"))

dat <- dat <- df %>% subset(grepl(":cond:tp", Predictor)) %>% select(Predictor, Estimate, `l-95% CI`, `u-95% CI`)
stargazer(dat, type = "latex", summary = F, rownames = F, 
          title = paste("Interaction effects of Questionnaires with intervention effect on P(novel)"),
          table.placement = "H",
          label = paste("tab:St2InterIntervNUO"))

############# Study 2 estimate ~ Q ##############

files <- list.files("Study2/CB_nQ")
for (i in files){load(paste0("Study2/CB_nQ/",i))}

# the model outputs are calculated in estimsQs.R and not supplied in the repository for file size reasons

for (param in c("beta", "ls", "tau")){
  
  print(paste0("\\subsection*{Predicting ",param," parameter from questionnaires in Study 2}"))
  
  # View the result
  qs <- c("STICSAcog", "STICSAsoma", "MCQ", "MCQpos", "MCQneg", "MCQconf",
          "MCQcontrol", "MCQselfcons", "MWQbelief","MWQfreq", "CASpre", "CASpost", "CASchange", "PHQ")
  
  var_names <- lapply(qs, function(x) paste(param,x, sep = "_"))
  
  
  # Dynamically retrieve variables
  selected_vars <- lapply(var_names,function(x) get(x)$fixed)
  names(selected_vars) <- var_names
  df <- selected_vars %>% 
    bind_rows(.id = "Predictor") %>% 
    mutate(pred = apply(as.array(rownames(.)), 1, function(x) strsplit(x, "\\.\\.\\.")[[1]][1]),
           Predictor = apply(as.array(Predictor), 1, function(x) strsplit(x, "_")[[1]][2]),
           Predictor = ifelse(grepl("Q:", pred), paste0(Predictor,":", substr(pred, 3,nchar(pred))),Predictor))
  
  dat <- df %>% subset(pred == "Q") %>% select(Predictor, Estimate, `l-95% CI`, `u-95% CI`)
  stargazer(dat, type = "latex", summary = F, rownames = F, 
            title = paste("Main effects of Questionnaires on", param, "parameter"),
            table.placement = "H",
            label = paste("tab:St2Main",param))
  
  dat <- df %>% subset(pred == "Q:cond") %>% select(Predictor, Estimate, `l-95% CI`, `u-95% CI`)
  stargazer(dat, type = "latex", summary = F, rownames = F, 
            title = paste("Interaction effects of Questionnaires with condition on", param, "parameter"),
            table.placement = "H",
            label = paste("tab:St2Intercond",param))
  
  dat <- df %>% subset(pred == "Q:tp") %>% select(Predictor, Estimate, `l-95% CI`, `u-95% CI`)
  stargazer(dat, type = "latex", summary = F, rownames = F, 
            title = paste("Interaction effects of Questionnaires with timpoint on", param, "parameter"),
            table.placement = "H",
            label = paste("tab:St2Intertp",param))
  
  dat <- dat <- df %>% subset(pred == "Q:cond:tp") %>% select(Predictor, Estimate, `l-95% CI`, `u-95% CI`)
  stargazer(dat, type = "latex", summary = F, rownames = F, 
            title = paste("Interaction effects of Questionnaires with intervention effect on", param, "parameter"),
            table.placement = "H",
            label = paste("tab:St2InterInterv",param))
  
}


################# Study 3 questionnaire result tables ###########

####### P(novel)

version <- "strict" # strict vs loose inclusion criteria (default is strict)


load(paste0("replication_study/analysis/QsNUO",version,".Rda"))
# the model results in QsNUO[version].Rda can be obtained by executing
# replication_study/analyses/novelOptionsByQuestionnaires.R

main <- rbind(STICA_T_c$fixed[rownames(STICA_T_c$fixed) == "STICA_T_c", ],
              STICA_T_s$fixed[rownames(STICA_T_s$fixed) == "STICA_T_s", ], 
              CAPE$fixed[rownames(CAPE$fixed) == "CAPE", ], 
              IUS$fixed[rownames(IUS$fixed) == "IUS", ], 
              RRQ$fixed[rownames(RRQ$fixed) == "RRQ", ], 
              PID$fixed[rownames(PID$fixed) == "PID", ],
              PSWQ$fixed[rownames(PSWQ$fixed) == "PSWQ", ])

df <- data.frame(var = c("cognitive anxiety", "somatic anxiety", "depressivity", "intolerance to uncertainty", "rumination", "negative affect", "worry"), 
                 Estimate = main[ ,1], lower = main[ ,3], upper = main[ ,4])
df$var = factor(df$var, levels = df$var, labels = df$var)

colnames(df) <- c("Predictor","beta", 
                  "95HDI lower bound", "95HDI upper bound")

stargazer(df, type = "latex", summary = F, rownames = F, 
          title = paste("Main effects using",version,"inclusion criteria"))

interact <- rbind(STICA_T_c$fixed[rownames(STICA_T_c$fixed) == "STICA_T_c:krakenPresent", ],
                  STICA_T_s$fixed[rownames(STICA_T_s$fixed) == "STICA_T_s:krakenPresent", ], 
                  CAPE$fixed[rownames(CAPE$fixed) == "CAPE:krakenPresent", ], 
                  IUS$fixed[rownames(IUS$fixed) == "IUS:krakenPresent", ], 
                  RRQ$fixed[rownames(RRQ$fixed) == "RRQ:krakenPresent", ], 
                  PID$fixed[rownames(PID$fixed) == "PID:krakenPresent", ],
                  PSWQ$fixed[rownames(PSWQ$fixed) == "PSWQ:krakenPresent", ])

df <- data.frame(var = c("cognitive anxiety*condition", "somatic anxiety*condition", 
                         "depressivity*condition", "intolerance to uncertainty*condition", 
                         "rumination*condition", "negative affect*condition", "worry*condition"), 
                 Estimate = main[ ,1], lower = main[ ,3], upper = main[ ,4])

df$var = factor(df$var, levels = df$var, labels = df$var)

colnames(df) <- c("Predictor","beta", 
                  "95HDI lower bound", "95HDI upper bound")

stargazer(df, type = "latex", summary = F, rownames = F, 
          title = paste("Interaction effects using",version,"inclusion criteria"))


#### parameter estimates by questionnaire
version <- "strict"

files <- list.files(path = "replication_study/analysis/parameterEstimatesCB_n")
files <- files[grepl(version, files)]
# these files can be optained by executing
# replication_study/analyses/parameterEstimatesByQuestionnaires.R

for (i in files) {load(paste0("replication_study/analyses/parameterEstimatesCB_n/",i))}

for (param in c("beta", "ls", "tau")){
  print(param)
  
  Sc <- get(paste(param,"STICSAcog", sep = "_"), envir = .GlobalEnv)
  Ss <- get(paste(param,"STICSAsoma", sep = "_"), envir = .GlobalEnv)
  C <- get(paste(param,"CAPE", sep = "_"), envir = .GlobalEnv)
  I <- get(paste(param,"IUS", sep = "_"), envir = .GlobalEnv)
  R <- get(paste(param,"RRQ", sep = "_"), envir = .GlobalEnv)
  Pi <- get(paste(param,"PID5", sep = "_"), envir = .GlobalEnv)
  Ps <- get(paste(param,"PSWQ", sep = "_"), envir = .GlobalEnv)
  
  main <- rbind(Sc$fixed[rownames(Sc$fixed) == "STICSAcog", ],
                Ss$fixed[rownames(Ss$fixed) == "STICSAsoma", ], 
                C$fixed[rownames(C$fixed) == "CAPE", ], 
                I$fixed[rownames(I$fixed) == "IUS", ], 
                R$fixed[rownames(R$fixed) == "RRQ", ], 
                Pi$fixed[rownames(Pi$fixed) == "PID5", ],
                Ps$fixed[rownames(Ps$fixed) == "PSWQ", ])
  
  df <- data.frame(var = c("cognitive anxiety", "somatic anxiety", "depressivity", 
                           "intolerance to uncertainty", "rumination", "negative affect", "worry"), 
                   Estimate = main[ ,1], lower = main[ ,3], upper = main[ ,4])
  df$var = factor(df$var, levels = df$var, labels = df$var)
  
  colnames(df) <- c("Predictor","beta", 
                    "95HDI lower bound", "95HDI upper bound")
  
  stargazer(df, type = "latex", summary = F, rownames = F, 
            title = paste("Main effects using",version,"inclusion criteria"),
            table.placement = "H",
            label = paste0("tab:",param,"main",version))
  
  interact <- rbind(Sc$fixed[rownames(Sc$fixed) == "STICSAcog:kraken_present", ],
                    Ss$fixed[rownames(Ss$fixed) == "STICSAsoma:kraken_present", ], 
                    C$fixed[rownames(C$fixed) == "CAPE:kraken_present", ], 
                    I$fixed[rownames(I$fixed) == "IUS:kraken_present", ], 
                    R$fixed[rownames(R$fixed) == "RRQ:kraken_present", ], 
                    Pi$fixed[rownames(Pi$fixed) == "PID:kraken_present", ],
                    Ps$fixed[rownames(Ps$fixed) == "PSWQ:kraken_present", ])
  
  df <- data.frame(var = c("cognitive anxiety*condition", "somatic anxiety*condition", 
                           "depressivity*condition", "intolerance to uncertainty*condition", 
                           "rumination*condition", "negative affect*condition", "worry*condition"), 
                   Estimate = main[ ,1], lower = main[ ,3], upper = main[ ,4])
  
  df$var = factor(df$var, levels = df$var, labels = df$var)
  
  colnames(df) <- c("Predictor","beta", 
                    "95HDI lower bound", "95HDI upper bound")
  
  stargazer(df, type = "latex", summary = F, rownames = F, 
            title = paste("Interaction effects using",version,"inclusion criteria"),
            table.placement = "H",
            label = paste0("tab:",param,"interact", version))
  
  
}


############ nervousness results for strict and loose dataset

load("replication_study/analyses/nerv_pnovel_strict.Rda")
# this is the result from executing analyses in replication_study/analyses/main_analyses.R
# the file is also provided for your convienience

df <- data.frame(nerv_pnovel$fixed) %>% 
  mutate(Predictor = rownames(.)) %>% 
  select(Predictor, Estimate, `l.95..CI`, `u.95..CI`)


stargazer(df, type = "latex", summary = F, rownames = F, 
          title = paste("Predicting P(novel) in Study 3 using the strict inclusion criteria"),
          table.placement = "H",
          label = "tab:nervSt3strict")


load("replication_study/analysis/nerv_pnovel_loose.Rda")

df <- data.frame(nerv_pnovel$fixed) %>% 
  mutate(Predictor = rownames(.)) %>% 
  select(Predictor, Estimate, `l.95..CI`, `u.95..CI`)


stargazer(df, type = "latex", summary = F, rownames = F, 
          title = paste("Predicting P(novel) in Study 3 using the loose inclusion criteria"),
          table.placement = "H",
          label = "tab:nervSt3loose")


###### parameter estimates

# this uses results from regressions executed in replication_study/analyses/main_analyses.R
for (param in c("eta", "ls", "tau")){
  
  print(paste0("\\subsection*{Predicting ",param," parameter in Study 3}"))
  
  for (version in c("strict", "loose")){
    
    v <- ifelse(version == "strict", "", "_loose")
    load(paste0("replication_study/analysis/estims_nervous",v,".Rda"))
    
    nerv_pnovel <-  get(paste(param), envir = .GlobalEnv)
    df <- data.frame(nerv_pnovel$fixed) %>% 
      mutate(Predictor = rownames(.)) %>% 
      select(Predictor, Estimate, `l.95..CI`, `u.95..CI`)
    
    stargazer(df, type = "latex", summary = F, rownames = F, 
              title = paste("Predicting", param,"in Study 3 using the",version,"inclusion criteria"),
              table.placement = "H",
              label = paste("tab:St3",param,version))
    
  }
  
}

############ P(novel) by factor scores

# these files are not supplied due to their file size but they can be obtained by 
# executing replication_study/analyses/novelOptionsByFactorScores.R
for (version in c("strict", "loose")){
  
  files = list.files("replication_study/analysis/NUOz")
  files = files[grepl(version, files) & grepl("4factors", files)]
  
  for (i in files) {load(paste0("replication_study/analysis/NUOz/",i))}
    
    print(paste0("\\subsubsection*{Predicting P(novel) from factor scores in Study 3, ", version, " inclusion criteria}"))
    
    main <- rbind(MR1$fixed[rownames(MR1$fixed) == "MR1", ],
                  MR2$fixed[rownames(MR2$fixed) == "MR2", ], 
                  MR3$fixed[rownames(MR3$fixed) == "MR3", ], 
                  MR4$fixed[rownames(MR4$fixed) == "MR4", ])
    
    df <- data.frame(var = c("worry/rumination", "depression/cog. anxiety", "somatic anxiety", 
                             "intolerance to uncertainty"), 
                     Estimate = main[ ,1], lower = main[ ,3], upper = main[ ,4])
    df$var = factor(df$var, levels = df$var, labels = df$var)
    
    colnames(df) <- c("Predictor","beta", 
                      "95HDI lower bound", "95HDI upper bound")
    
    stargazer(df, type = "latex", summary = F, rownames = F, 
              title = paste("Main effects using",version,"inclusion criteria"),
              table.placement = "H",
              label = paste0("tab:NUOmain",version, "factor3"))
    
    ## interaction effect
    main <- rbind(MR1$fixed[rownames(MR1$fixed) == "MR1:krakenPresent", ],
                  MR2$fixed[rownames(MR2$fixed) == "MR2:krakenPresent", ], 
                  MR3$fixed[rownames(MR3$fixed) == "MR3:krakenPresent", ], 
                  MR4$fixed[rownames(MR4$fixed) == "MR4:krakenPresent", ])
    
    df <- data.frame(var = c("worry/rumination*condition", "depression/cog. anxiety*condition", "somatic anxiety*condition", 
                             "intolerance to uncertainty*condition"), 
                     Estimate = main[ ,1], lower = main[ ,3], upper = main[ ,4])
    df$var = factor(df$var, levels = df$var, labels = df$var)
    
    colnames(df) <- c("Predictor","beta", 
                      "95HDI lower bound", "95HDI upper bound")
    
    stargazer(df, type = "latex", summary = F, rownames = F, 
              title = paste("Interaction effects using",version,"inclusion criteria"),
              table.placement = "H",
              label = paste0("tab:NUOinteract",version, "factor3"))
    

  
}
########### parameters by factor scores

# these files are not supplied due to their file size but they can be obtained by
# executing replication_study/analyses/parameterEstimatesByFactorScores.R
for (version in c("strict", "loose")){

  files <- list.files(path = "replication_study/analyses/parameterEstimatesCB_n")
  files <- files[grepl(version, files)]
  
  for (i in files) {load(paste0("replication_study/analyses/parameterEstimatesCB_n/",i))}
  
for (param in c("eta", "ls", "tau")){
  p <- ifelse(param == "eta", "beta",param)
  
  print(paste0("\\subsubsection*{Predicting ",param," parameter from factor scores in Study 3, ", version, " inclusion criteria}"))
  
  MR1 <- get(paste(p,"MR1", sep = "_"), envir = .GlobalEnv)
  MR2 <- get(paste(p,"MR2", sep = "_"), envir = .GlobalEnv)
  MR3 <- get(paste(p,"MR3", sep = "_"), envir = .GlobalEnv)
  MR4 <- get(paste(p,"MR4", sep = "_"), envir = .GlobalEnv)
  
  main <- rbind(MR1$fixed[rownames(MR1$fixed) == "MR1", ],
                MR2$fixed[rownames(MR2$fixed) == "MR2", ], 
                MR3$fixed[rownames(MR3$fixed) == "MR3", ], 
                MR4$fixed[rownames(MR4$fixed) == "MR4", ])
  
  df <- data.frame(var = c("worry/rumination", "depression/cog. anxiety", "somatic anxiety", 
                           "intolerance to uncertainty"), 
                   Estimate = main[ ,1], lower = main[ ,3], upper = main[ ,4])
  df$var = factor(df$var, levels = df$var, labels = df$var)
  
  colnames(df) <- c("Predictor","beta", 
                    "95HDI lower bound", "95HDI upper bound")
  
  stargazer(df, type = "latex", summary = F, rownames = F, 
            title = paste("Main effects using",version,"inclusion criteria"),
            table.placement = "H",
            label = paste0("tab:",param,"main",version, "factor3"))
  
  ## interaction effect
  main <- rbind(MR1$fixed[rownames(MR1$fixed) == "MR1:kraken_present", ],
                MR2$fixed[rownames(MR2$fixed) == "MR2:kraken_present", ], 
                MR3$fixed[rownames(MR3$fixed) == "MR3:kraken_present", ], 
                MR4$fixed[rownames(MR4$fixed) == "MR4:kraken_present", ])
  
  df <- data.frame(var = c("worry/rumination*condition", "depression/cog. anxiety*condition", "somatic anxiety*condition", 
                           "intolerance to uncertainty*condition"), 
                   Estimate = main[ ,1], lower = main[ ,3], upper = main[ ,4])
  df$var = factor(df$var, levels = df$var, labels = df$var)
  
  colnames(df) <- c("Predictor","beta", 
                    "95HDI lower bound", "95HDI upper bound")
  
  stargazer(df, type = "latex", summary = F, rownames = F, 
            title = paste("Interaction effects using",version,"inclusion criteria"),
            table.placement = "H",
            label = paste0("tab:",param,"interact",version, "factor3"))
    
}
  
}
  

################### model comparison plots ##############
# to obtain the groupBMC results for all studies can be obtained using the code in
# groupBMC.py


############ model comparison St 2

df <- read.csv("Study2/analyses/groupBMC_results_incl_cb_n.csv")

df$cond <- factor(df$cond, levels = c(0,1), labels = c("control", "intervention"))
df$tp <- factor(df$tp, levels = c(0,1), labels = c("pre", "post"))



p1 <- ggplot(df, aes(y=exceedance_probability, x=model, fill=cond)) +
  #bars
  geom_bar(position= position_dodge(0.8), stat="identity", width=0.75)+
  scale_fill_manual(values = c(control, red), name = "Condition")+
  # scale_color_manual(values = c(gold, red), name = "Condition")+
  #title
  labs(title = "Model Comparison", 
       x = "model", 
       y = "exceedance probability")+
  theme_classic(base_size = 15) +
  #scale_y_continuous(limits = c(0,3), expand = c(0, 0)) +
  #scale_x_discrete(labels = c("safe", "risky"))+
  #adjust text size
  theme(text = element_text(size=16)) +
  #theme(legend.position = "none")+
  scale_x_discrete(labels = c("POS", "Random", expression("CB"~beta~"=0"), "full CB", "novelty bonus"))+
  #scale_x_discrete(labels = c("POS", "POS-T", "random", expression("CB_"~beta~"0"), expression("CB_"~lambda~"0")))+
  #scale_y_continuous(expand = c(0.01, 0))+
  
  #  geom_jitter(data = fit, aes(x = model, y= logp), alpha = 0.1, width = 0.1)+
  facet_grid(cols = vars(tp))

p1


################ St1 model comparison

exc <- read.csv("Study1/analyses/groupBMC_results_incl_cb_n.csv")
exc$kraken_present <- factor(exc$kraken_present, levels = c(0,1), labels = c("safe", "risky"))

ggplot(exc, aes(y=exceedance_probability, x=model, fill=as.factor(kraken_present))) +
  #bars
  geom_bar(position= position_dodge(0.8), stat="identity", width=0.75)+
  scale_fill_manual(values = c(darkBlue, red), name = "Condition")+
  #title
  labs(title = "Model Comparison", 
       x = "model", 
       y = "exceedance probability")+
  theme_classic(base_size = 15) +
  #scale_y_continuous(limits = c(0,3), expand = c(0, 0)) +
  #scale_x_discrete(labels = c("safe", "risky"))+
  #adjust text size
  theme(text = element_text(size=16)) +
  theme(legend.position = "none")+
  scale_x_discrete(labels = c("POS", "random", expression("CB_"~beta~"=0"), "full CB", "novelty bonus"))+
  scale_y_continuous(expand = c(0.01, 0))+
  facet_grid(cols = vars(kraken_present))

######## replication study

exc <- read.csv("replication_study/analyses/groupBMC_results_incl_cb_n.csv")  %>% 
  mutate(kraken_present = factor(kraken_present, levels = c(0,1), labels = c("safe", "risky")))

ggplot(exc, aes(y = exceedance_probability, x=model, fill=kraken_present)) +
  #bars
  geom_bar(position= position_dodge(0.8), stat="identity", width=0.75)+
  scale_fill_manual(values = c(darkBlue, red), name = "Condition")+
  #geom_errorbar(aes(ymin = logp-se, ymax = logp+se), width = 0.2) +
  #title
  labs(title = "Model Comparison", 
       x = "model", 
       y = "exceedance probability")+
  theme_classic(base_size = 15) +
  #scale_y_continuous(limits = c(0,3), expand = c(0, 0)) +
  #scale_x_discrete(labels = c("safe", "risky"))+
  #adjust text size
  theme(text = element_text(size=16)) +
  theme(legend.position = "none")+
  scale_x_discrete(labels = c("random", expression("CB_"~beta~"=0"), "full CB", "novelty bonus"))+
  scale_y_continuous(expand = c(0.01, 0))+
  facet_grid(cols = vars(kraken_present))


############# parameter recovery plots ##########

### scatter plot

trueParameters <- read.csv("Study1/data/estimatesCB_n.csv")%>% 
  pivot_longer(cols = 3:5, names_to = "Parameter", values_to = "generating")
recoveredParameters <- read.csv("Study1/data/recoveredEstimates.csv")%>% 
  pivot_longer(cols = 3:5, names_to = "Parameter", values_to = "recovered")

parameters <- trueParameters %>% 
  left_join(recoveredParameters, by = c("ID", "kraken_present", "Parameter")) %>% 
  mutate(Condition = factor(kraken_present, levels = c(0,1), labels = c("safe", "risky")),
         Parameter = recode(Parameter, "beta" = "eta", "ls" = "lambda"))

library(ggh4x)

lims <- parameters %>% 
  pivot_longer(cols = c(generating, recovered), names_to = "source", values_to = "estimate") %>% 
  group_by(Parameter) %>% 
  summarise(min = min(estimate),
            max = max(estimate))

p1 <- ggplot(parameters, aes(generating, recovered, color = Condition)) + 
  geom_jitter(alpha = 0.3) +
  facet_wrap(vars(Parameter), scales = "free") +
  facetted_pos_scales(
    x = list(
      eta = scale_x_continuous(limits = unlist(lims[lims$Parameter == "eta", 2:3])),
      lambda = scale_x_continuous(limits = unlist(lims[lims$Parameter == "lambda", 2:3])),
      tau = scale_x_continuous(limits = unlist(lims[lims$Parameter == "tau", 2:3]))
    ),
    y = list(
      eta = scale_y_continuous(limits = unlist(lims[lims$Parameter == "eta", 2:3])),
      lambda = scale_y_continuous(limits = unlist(lims[lims$Parameter == "lambda", 2:3])),
      tau = scale_y_continuous(limits = unlist(lims[lims$Parameter == "tau", 2:3]))
    )
  )+
  geom_abline(aes(intercept = 0, slope = 1))+
  scale_color_manual(values = c(darkBlue, red))+
  ggtitle("Parameter recovery in Study 1")

p1

#ggsave("plots/SIParameterRecoveryScatterStudy1.png", plot = p1, width = 9, height = 3)


####### confusion matrix plot

parameters <- read.csv("Study1/data/estimatesCB_n.csv") %>% 
  left_join(read.csv("Study1/data/recoveredEstimates.csv"), by = c("ID", "kraken_present"))

compute_correlations <- function(data) {
  data %>%
    cor(use = "pairwise.complete.obs") %>%
    as.data.frame() %>%
    mutate(row = rownames(.)) %>%
    subset(grepl("y", row), select = !grepl("y", colnames(.))) %>%
    pivot_longer(cols = c(1:3), names_to = "generating", values_to = "cor") %>%
    rename(recovered = row) %>%
    mutate(generating = substr(generating, 1, nchar(generating) - 2),
           recovered = substr(recovered, 1, nchar(recovered) - 2))
}

cors <- parameters %>%
  split(.$kraken_present) %>%
  map(~ .x %>% select(-ID, -kraken_present) %>% compute_correlations()) %>%
  bind_rows(.id = "Condition") %>%
  mutate(Condition = factor(Condition, levels = c(0,1), labels = c("safe", "risky")),
         generating = recode(generating, "beta" = "eta", "ls" = "lambda"),
         recovered = recode(recovered, "beta" = "eta", "ls" = "lambda"))



p2 <- ggplot(cors, aes(generating, recovered, fill = cor)) + 
  geom_raster()+
  scale_fill_gradient2(low = red, mid = "white", high = darkBlue) +
  geom_label(aes(label = round(cor, digits =2)), fill = "white")+ 
  facet_wrap(vars(Condition)) +
  ggtitle("Parameter identifiability in Study 1")
  
p2

recov <- ggarrange(p1, p2, ncol = 1, nrow = 2, labels = "AUTO")

recov

ggsave("plots/SIParameterRecoveryStudy1.png", plot = recov, width = 9, height = 8)



##### same for study 2


### scatter plot

trueParameters <- read.csv("Study2/data/estimatesCB_n.csv")%>% 
  pivot_longer(cols = 4:6, names_to = "Parameter", values_to = "generating")
recoveredParameters <- read.csv("Study2/data/recoveredEstimates.csv")%>% 
  subset(select = -X) %>% 
  pivot_longer(cols = 4:6, names_to = "Parameter", values_to = "recovered")

parameters <- trueParameters %>% 
  left_join(recoveredParameters, by = c("ID", "cond","tp", "Parameter")) %>% 
  mutate(Condition = factor(cond, levels = c(0,1), labels = c("control", "intervention")),
         Timepoint = factor(tp, levels = c(0,1), labels = c("Pre", "Post")),
         Parameter = recode(Parameter, "beta" = "eta", "ls" = "lambda"))

lims <- parameters %>% 
  pivot_longer(cols = c(generating, recovered), names_to = "source", values_to = "estimate") %>% 
  group_by(Parameter) %>% 
  summarise(min = min(estimate),
            max = max(estimate))

pa <- ggplot(parameters[parameters$Timepoint == "Pre", ], aes(generating, recovered, color = Condition)) + 
  geom_jitter() +
  facet_wrap(vars(Parameter), scales = "free") +
  facetted_pos_scales(
    x = list(
      eta = scale_x_continuous(limits = unlist(lims[lims$Parameter == "eta", 2:3])),
      lambda = scale_x_continuous(limits = unlist(lims[lims$Parameter == "lambda", 2:3])),
      tau = scale_x_continuous(limits = unlist(lims[lims$Parameter == "tau", 2:3]))
    ),
    y = list(
      eta = scale_y_continuous(limits = unlist(lims[lims$Parameter == "eta", 2:3])),
      lambda = scale_y_continuous(limits = unlist(lims[lims$Parameter == "lambda", 2:3])),
      tau = scale_y_continuous(limits = unlist(lims[lims$Parameter == "tau", 2:3]))
    )
  )+
  geom_abline(aes(intercept = 0, slope = 1))+
  scale_color_manual(values = c(control, red)) +
  ggtitle("Baseline")

pa

pb <- ggplot(parameters[parameters$Timepoint == "Post", ], aes(generating, recovered, color = Condition)) + 
  geom_jitter() +
  facet_wrap(vars(Parameter), scales = "free") +
  facetted_pos_scales(
    x = list(
      eta = scale_x_continuous(limits = unlist(lims[lims$Parameter == "eta", 2:3])),
      lambda = scale_x_continuous(limits = unlist(lims[lims$Parameter == "lambda", 2:3])),
      tau = scale_x_continuous(limits = unlist(lims[lims$Parameter == "tau", 2:3]))
    ),
    y = list(
      eta = scale_y_continuous(limits = unlist(lims[lims$Parameter == "eta", 2:3])),
      lambda = scale_y_continuous(limits = unlist(lims[lims$Parameter == "lambda", 2:3])),
      tau = scale_y_continuous(limits = unlist(lims[lims$Parameter == "tau", 2:3]))
    )
  )+
  geom_abline(aes(intercept = 0, slope = 1))+
  scale_color_manual(values = c(control, red))+
  ggtitle("After intervention")

pb

p1 <- ggarrange(pa, pb, ncol = 1, nrow = 2,common.legend = T, legend = "right") + ggtitle("Parameter recovery in Study 2")

p1

#ggsave("plots/SIParameterRecoveryScatterStudy2.png", plot = p1, width = 9, height = 6)

####### confusion matrix plot

parameters <- read.csv("Study2/data/estimatesCB_n.csv") %>% 
  left_join(read.csv("Study2/data/recoveredEstimates.csv"), by = c("ID", "cond", "tp"))


# Capture group identifiers
group_info <- parameters %>%
  group_by(cond, tp) %>%
  group_keys()

# Split the dataframe by groups and compute correlations
cors <- parameters %>%
  group_by(cond, tp) %>%
  group_split() %>%
  map(~ .x %>% select(-ID, -cond, -tp) %>% compute_correlations()) %>%
  bind_rows(.id = "CondTp") %>%
  mutate(Condition = factor(rep(group_info$cond, each = 3*3), levels = c(0,1), labels = c("control", "intervention")),
         Timepoint = factor(rep(group_info$tp, each = 3*3), levels = c(0,1), labels = c("baseline", "post")),
         generating = recode(generating, "beta" = "eta", "ls" = "lambda"),
         recovered = recode(recovered, "beta" = "eta", "ls" = "lambda"))



p2 <- ggplot(cors, aes(generating, recovered, fill = cor)) + 
  geom_raster()+
  scale_fill_gradient2(low = red, mid = "white", high = darkBlue) +
  geom_label(aes(label = round(cor, digits =2)), fill = "white")+ 
  facet_grid(cols = vars(Condition), rows = vars(Timepoint)) +
  ggtitle("Parameter identifiability in Study 2")

p2


recov <- ggarrange(p1, p2, ncol = 1, nrow = 2, labels = "AUTO")

recov

ggsave("plots/SIParameterRecoveryStudy2.png", plot = recov, width = 9, height = 9)


############ replication study 

## scatter plot


trueParameters <- read.csv("replication_study/data/estimatesCB_n_strict.csv")%>% 
  pivot_longer(cols = 3:5, names_to = "Parameter", values_to = "generating")
recoveredParameters <- read.csv("replication_study/data/recovered_estimatesCB_n.csv")%>% 
  pivot_longer(cols = 3:5, names_to = "Parameter", values_to = "recovered")

parameters <- trueParameters %>% 
  left_join(recoveredParameters, by = c("ID", "kraken_present", "Parameter")) %>% 
  mutate(Condition = factor(kraken_present, levels = c(0,1), labels = c("safe", "risky")),
         Parameter = recode(Parameter, "beta" = "eta", "ls" = "lambda"))

library(ggh4x)

lims <- parameters %>% 
  pivot_longer(cols = c(generating, recovered), names_to = "source", values_to = "estimate") %>% 
  group_by(Parameter) %>% 
  summarise(min = min(estimate),
            max = max(estimate))

p1 <- ggplot(parameters, aes(generating, recovered, color = Condition)) + 
  geom_jitter(alpha = 0.3) +
  facet_wrap(vars(Parameter), scales = "free") +
  facetted_pos_scales(
    x = list(
      eta = scale_x_continuous(limits = unlist(lims[lims$Parameter == "eta", 2:3])),
      lambda = scale_x_continuous(limits = unlist(lims[lims$Parameter == "lambda", 2:3])),
      tau = scale_x_continuous(limits = unlist(lims[lims$Parameter == "tau", 2:3]))
    ),
    y = list(
      eta = scale_y_continuous(limits = unlist(lims[lims$Parameter == "eta", 2:3])),
      lambda = scale_y_continuous(limits = unlist(lims[lims$Parameter == "lambda", 2:3])),
      tau = scale_y_continuous(limits = unlist(lims[lims$Parameter == "tau", 2:3]))
    )
  )+
  geom_abline(aes(intercept = 0, slope = 1))+
  scale_color_manual(values = c(darkBlue, red))+
  ggtitle("Parameter recovery in replication")

p1


####### confusion matrix plot

parameters <- read.csv("replication_study/data/estimatesCB_n_strict.csv") %>% 
  left_join(read.csv("replication_study/data/recovered_estimatesCB_n.csv"), by = c("ID", "kraken_present"))

compute_correlations <- function(data) {
  data %>%
    cor(use = "pairwise.complete.obs") %>%
    as.data.frame() %>%
    mutate(row = rownames(.)) %>%
    subset(grepl("y", row), select = !grepl("y", colnames(.))) %>%
    pivot_longer(cols = c(1:3), names_to = "generating", values_to = "cor") %>%
    rename(recovered = row) %>%
    mutate(generating = substr(generating, 1, nchar(generating) - 2),
           recovered = substr(recovered, 1, nchar(recovered) - 2))
}

cors <- parameters %>%
  split(.$kraken_present) %>%
  map(~ .x %>% select(-ID, -kraken_present) %>% compute_correlations()) %>%
  bind_rows(.id = "Condition") %>%
  mutate(Condition = factor(Condition, levels = c(0,1), labels = c("safe", "risky")),
         generating = recode(generating, "beta" = "eta", "ls" = "lambda"),
         recovered = recode(recovered, "beta" = "eta", "ls" = "lambda"))



p2 <- ggplot(cors, aes(generating, recovered, fill = cor)) + 
  geom_raster()+
  scale_fill_gradient2(low = red, mid = "white", high = darkBlue) +
  geom_label(aes(label = round(cor, digits =2)), fill = "white")+ 
  facet_wrap(vars(Condition)) +
  ggtitle("Parameter identifiability in replication")

p2


recov <- ggarrange(p1, p2, ncol = 1, nrow = 2, labels = "AUTO")

recov


ggsave("plots/SIParameterRecoveryStudy3.png", plot = recov, width = 9, height = 8)

################ randomly sample answers to intervention questions for SI ################

set.seed(111)
load("Study2/data/intervention.Rda")

# loop through questions of the intervention and randomly sample 2 participants to print the answers from

out_string_interv <- ""
for (Q in c(1:5)) {
  
  pp <- sample(intervention$ID[intervention$cond == 1], 2, replace = F)
  
  texts <- intervention$texts[intervention$ID == pp[1]][[1]]
  A1 <- texts[(Q+5)] # add 5 bc that is the version of the text after they had a chance to change it
  
  texts <- intervention$texts[intervention$ID == pp[2]][[1]]
  A2 <- texts[(Q+5)]
  
  while(is.null(A1) | is.null(A2) ){
    pp <- sample(intervention$ID[intervention$cond == 1], 2, replace = F)
    
    texts <- intervention$texts[intervention$ID == pp[1]][[1]]
    A1 <- texts[(Q+5)] # add 5 bc that is the version of the text after they had a chance to change it
    
    texts <- intervention$texts[intervention$ID == pp[2]][[1]]
    A2 <- texts[(Q+5)]
  }
  print(pp)
  Q_text <- paste0("Q", Q, ":\\Answer 1: ", A1, "\\Answer 2: ", A2, "\\")
  out_string_interv <- paste0(out_string_interv, Q_text)
  
}

out_string_interv


out_string_control <- ""
for (Q in c(1:5)) {
  
  pp <- sample(intervention$ID[intervention$cond == 0], 2, replace = F)
  
  texts <- intervention$texts[intervention$ID == pp[1]][[1]]
  A1 <- texts[(Q+5)] # add 5 bc that is the version of the text after they had a chance to change it
  
  texts <- intervention$texts[intervention$ID == pp[2]][[1]]
  A2 <- texts[(Q+5)]
  
  while(is.null(A1) | is.null(A2) ){
    pp <- sample(intervention$ID[intervention$cond == 0], 2, replace = F)
    
    texts <- intervention$texts[intervention$ID == pp[1]][[1]]
    A1 <- texts[(Q+5)] # add 5 bc that is the version of the text after they had a chance to change it
    
    texts <- intervention$texts[intervention$ID == pp[2]][[1]]
    A2 <- texts[(Q+5)]
  }
  print(pp)
  Q_text <- paste0("Q", Q, ":\\Answer 1: ", A1, "\\Answer 2: ", A2, "\\")
  out_string_control <- paste0(out_string_control, Q_text)
  
}

out_string_control


################ Demographics overview table ###############

# what do we want to plot in here?
# one big table with demographics bc those are the same in both studies
# then separate ones for questionnaire scores


#### general
demographic <- c("age", "Sex_0", "motivation_rating", "kraken_rating", "diagnosis_2", "meds_3")
load("Study1/data/questionnaires_processed.Rda")
load("Study1/data/master.Rda")

questionnaires_1 <- questionnaires %>% 
  subset(is.element(item, demographic) & is.element(ID, Master$ID))

load("Study2/data/questionnairesPre.Rda")
load("Study2/data/questionnairesPost.Rda")

questionnaires_2 <- questionnairesPre %>% 
  left_join(questionnairesPost, by = "ID") %>% 
  mutate(diagnosis_2 = as.numeric(diagnosis_2),
         meds_3 = as.numeric(meds_3),
         feedback = NA) %>% 
  pivot_longer(cols = -ID, names_to = "item", values_to = "score") %>% 
  subset(is.element(item, demographic))

load("replication_study/data/demographics.Rda")
load("replication_study/data/exclusions.Rda")
incl <- exclusions$ID[exclusions$final_strict == 0]
questionnaires_3 <- demographics %>% 
  rename(kraken_rating = kraken_2,
         motivation_rating = motivation_1) %>% 
  mutate(age = ifelse(as.numeric(age) > 100, NA, age)) %>% # I person is 6115 y/o which is likely a typo
  pivot_longer(cols = -c(ID), names_to = "item", values_to = "score") %>% 
  subset(is.element(item, demographic) & is.element(ID, incl)) %>% 
  mutate(score = as.numeric(score)) 
  

demo <-  questionnaires_1 %>% 
  select(ID, item, score) %>% 
  bind_rows(questionnaires_2, questionnaires_3, .id = "Study")

# separate into control and intervention for study 2
load("Study2/data/intervention.Rda")

mean_sd <- function(x){
  out_str <- sprintf("%.1f (%.1f)", mean(x, na.rm = T), sd(x, na.rm = T))
  return(out_str)
}

N_percent <- function(x, value){
  out_str <- sprintf("%i (%.2f)", sum(x == value), mean(x == value)*100)
  return(out_str)
}


load("Study2/data/inclusions.Rda")

demo_tab <- demo %>% 
  left_join(intervention %>% select(ID, cond), by = "ID") %>% 
  mutate(Study = ifelse(Study == 1, "Study 1",
                        ifelse(Study == 3, "Replication", 
                               ifelse(cond == 1, "Study 2 intervention", "Study 2 control"))) )  %>% 
  pivot_wider(id_cols = c(ID, Study), names_from = item, values_from = score) %>% 
  group_by(Study) %>% 
  summarise(`Age_Mean (SD)` = mean_sd(age),
            Age_Range = sprintf("%i - %i", min(age, na.rm = T), max(age, na.rm = T)),
            Gender_Male = N_percent(Sex_0, 0),
            Gender_Female = N_percent(Sex_0, 1),
            Gender_Other = N_percent(Sex_0, 2),
            `Motivation_Mean (SD)` = mean_sd(motivation_rating),
            `Fear of kraken_Mean (SD)` = mean_sd(kraken_rating),
            Diagnosis_Yes = N_percent(diagnosis_2, 0),
            Diagnosis_No = N_percent(diagnosis_2, 1),
            `Diagnosis_Prefer not to say` = N_percent(diagnosis_2, 2),
            Medication_Yes = N_percent(meds_3, 0),
            Medication_No = N_percent(meds_3, 1),
            `Medication_Prefer not to say` = N_percent(meds_3, 2))  %>% # aggregate by study
  pivot_longer(cols = -Study, names_to = "Question", values_to = "Answer") %>%
  # pivot to format where item, study 1, study 2 intervention, study 2 control
  pivot_wider(id_cols = Question, names_from = Study, values_from = Answer) %>% 
  mutate(Topic = sub("_.*", "", Question),
         Question = sub(".*_", "", Question)) %>% 
  select(Topic, everything()) %>% 
# add number of exclusions
rbind(data.frame(Topic = rep("Exclusions", 8),
                 Question = rep(c("Excluded", "Included"), 4),
                 Study = rep(c("Study 1", "Study 2 control", "Study 2 intervention", "Replication"), each = 2),
                 Answer = c(sprintf("%i (%.2f)", 300 - 220, ((300-220)/300)*100),
                               sprintf("%i (%.2f)", 220, (220/300)*100),
                            N_percent(in_ex$included[in_ex$cond == "control"], 0),
                                       N_percent(in_ex$included[in_ex$cond == "control"], 1),
                            N_percent(in_ex$included[in_ex$cond == "intervention"], 0),
                                       N_percent(in_ex$included[in_ex$cond == "intervention"], 1),
                            N_percent(exclusions$final_strict, 1),
                            N_percent(exclusions$final_strict, 0))) %>% 
        pivot_wider(id_cols = c(Question, Topic), names_from = Study, values_from = Answer)) %>% 
  select(Topic, Question, `Study 1`, `Study 2 control`, `Study 2 intervention`, Replication)


stargazer::stargazer(demo_tab, summary = F, rownames = F)

#### Study 1
# edu
# income
# ESI
# IUS
# RRQ
# STICSA cognitve
# STICSA soma
# AQ10
# ASRS
# CAPE (all subscales)
# PID5 (all subscales)
# covid

load("Study1/data/questionnaires_processed.Rda")
load("Study1/data/master.Rda")

questionnaires_1 <- questionnaires %>% 
  subset(!is.element(item, demographic) & is.element(ID, Master$ID))

unique(questionnaires_1$questionnaire)

demo_tab1 <- questionnaires_1 %>% 
  subset(!is.element(questionnaire, c("attention1", "rt.x", "income", "attention2", "rt.y"))) %>% # income is the only discrete variable here so we will process this separately
  group_by(questionnaire, ID) %>% 
  mutate(score = ifelse(questionnaire == "edu", score, score+1)) %>% # questionnaire responses should be coded 1-K not 0-(K-1)
  summarise(sum = sum(score, na.rm = T)) %>% # questionnaire score is usually the sum
  ungroup() %>% 
  group_by(questionnaire) %>% 
  summarise(meanSD = mean_sd(sum)) %>% 
  mutate(questionnaire = recode(questionnaire, "edu" = "Years of education",
                                "AQ" = "AQ 10",
                                "ESI" = "ESI short form")) %>% 
  rbind(data.frame(questionnaire = c('monthly income <$500', 
                                     'monthly income $500 - $1000', 
                                     'monthly income $1000 - $1500', 
                                     'monthly income $1500 - $2000', 
                                     'monthly income $2000 - $2500', 
                                     'monthly income $2500 - $3000', 
                                     'monthly income $3500 - $4000', 
                                     'monthly income >$4000'),
                   meanSD = c(N_percent(questionnaires_1$score[questionnaires_1$questionnaire == "income"], 0),
                              N_percent(questionnaires_1$score[questionnaires_1$questionnaire == "income"], 1),
                              N_percent(questionnaires_1$score[questionnaires_1$questionnaire == "income"], 2),
                              N_percent(questionnaires_1$score[questionnaires_1$questionnaire == "income"], 3),
                              N_percent(questionnaires_1$score[questionnaires_1$questionnaire == "income"], 4),
                              N_percent(questionnaires_1$score[questionnaires_1$questionnaire == "income"], 5),
                              N_percent(questionnaires_1$score[questionnaires_1$questionnaire == "income"], 6),
                              N_percent(questionnaires_1$score[questionnaires_1$questionnaire == "income"], 7))))


stargazer::stargazer(demo_tab1, rownames = F, summary = F)
## Study 2
# MCQ score
# MWQ score
# CAS score pre
# CAS score post
# STICSA cognitive
# STICSA somatic
# PHQ9 score

load("Study2/data/Master.Rda")



demo_tab2 <- Master %>% 
  subset(trial == 1 & block == 2,select = c(ID, MCQpos, MCQneg, MCQconf, MCQcontrol, MCQselfcons,
                                            MWQfreq, MWQbelief, CASpre, CASpost, CASchange, STICSAcog, STICSAsoma, PHQ, cond)) %>% 
  mutate(MWQbelief = MWQbelief / 7) %>% # I created a sum score here just like the other questionnaires but for this particular one a mean is more easily interpretable
  # all questionnaire scores are sum scores of the questionnaires coded from 0 to (K-1) instead of 1 to K 
  # I will fix this for STICSA in the interest of comparability between studies
  mutate(STICSAcog = STICSAcog +10,
         STICSAsoma = STICSAsoma +11) %>% 
  pivot_longer(cols = -c(ID, cond), names_to = "Questionnaire", values_to = "score") %>% 
  group_by(Questionnaire, cond) %>% 
  summarise(Mean_SD = mean_sd(score)) %>% 
  pivot_wider(id_cols = Questionnaire, names_from = cond, values_from = Mean_SD) %>% 
  mutate(Questionnaire = recode(Questionnaire, "MCQconf" = "MCQcognitive confidence",
                                "MCQneg" = "MCQnegative worries", "MCQpos" = "MCQpositive worries",
                                "MCQselfcons" = "MCQcognitive self-consciousness", "MCQcontrol" = "MCQneed for control",
                                "MWQfreq" = "MWQfrequency", "PHQ" = "PHQ-9", 
                                "STICSAcog" = "STICSAcognitive", "STICSAsoma" = "STICSAsomatic"))

stargazer::stargazer(demo_tab2, rownames = F, summary = F)

#### replication study

# edu
# income
# IUS
# RRQ
# STICSA cognitve
# STICSA soma
# CAPE depr
# PID5 negative aff
# PSWQ

load("replication_study/data/qs_item_level_clean_full.Rda")
demographics <- demographics %>% subset(is.element(ID, incl))
questionnaires <- questionnaires %>% 
  subset(is.element(ID, incl)) %>% 
  mutate(Measure = ifelse(Measure == "STICSA_T", paste0(Measure, subscale), Measure),
         value = as.numeric(value)) %>% 
  group_by(ID, Measure) %>% 
  summarise(score = sum(value+1)) %>% # +1 bc questionnaires should be coded 1 to K not 0 to K-1
  rbind(demographics %>% select(ID, edu) %>%
          mutate(edu = as.numeric(edu)) %>% 
          pivot_longer(cols = -ID, names_to = "Measure", values_to = "score"))


income <- data.frame(Measure = c('monthly income <$500', 
                                       'monthly income $500 - $1000', 
                                       'monthly income $1000 - $1500', 
                                       'monthly income $1500 - $2000', 
                                       'monthly income $2000 - $2500', 
                                       'monthly income $2500 - $3000', 
                                       'monthly income $3500 - $4000', 
                                       'monthly income >$4000'),
                     meanSD = c(N_percent(demographics$income_0, 0),
                                N_percent(demographics$income_0, 1),
                                N_percent(demographics$income_0, 2),
                                N_percent(demographics$income_0, 3),
                                N_percent(demographics$income_0, 4),
                                N_percent(demographics$income_0, 5),
                                N_percent(demographics$income_0, 6),
                                N_percent(demographics$income_0, 7)))

demo_tab3 <- questionnaires %>% 
  group_by(Measure) %>% 
  summarise(meanSD = mean_sd(score)) %>% 
  rbind(income) %>% 
  mutate(Measure = recode(Measure,
                          "CAPE" = "CAPE depressivity",
                          "PID" = "PID5 negative affect",
                          "STICSA_Tc" = "STICSA cognitive",
                          "STICSA_Ts" = "STICSA somatic",
                          "edu" = "Years of education"))
  





stargazer::stargazer(demo_tab3, rownames = F, summary = F)

############## distribution of questionnaire scores in all three datasets #########
demographic <- c("age", "Sex_0", "motivation_rating", "kraken_rating", "diagnosis_2", "meds_3",
                 "attention1", "edu", "rt.x", "rt.y", "attention2", "income_0")

load("Study1/data/questionnaires_processed.Rda")
load("Study1/data/master.Rda")

questionnaires_1 <- questionnaires %>% 
  subset(!is.element(item, demographic) & is.element(ID, Master$ID)) %>% 
  group_by(ID, questionnaire) %>% 
  summarise(score = sum(score +1)) # +1 bc values are 0 indexed but shouldn't be

load("Study2/data/Master.Rda")

questionnaires_2 <- Master %>% 
  subset(trial == 1 & block == 2,select = c(ID, MCQpos, MCQneg, MCQconf, MCQcontrol, MCQselfcons,
         MWQfreq, MWQbelief, CASpre, CASpost, CASchange, STICSAcog, STICSAsoma, PHQ, cond)) %>% 
  mutate(MWQbelief = MWQbelief / 7) %>% # I created a sum score here just like the other questionnaires but for this particular one a mean is more easily interpretable
  # all questionnaire scores are sum scores of the questionnaires coded from 0 to (K-1) instead of 1 to K 
  # I will fix this for STICSA in the interest of comparability between studies
  mutate(STICSAcog = STICSAcog +10,
         STICSAsoma = STICSAsoma +11) %>% 
  pivot_longer(cols = -c(ID, cond), names_to = "questionnaire", values_to = "score") %>% 
  mutate(questionnaire = recode(questionnaire,
                                "STICSAcog" = "STICSAcognitive",
                                "STICSAsoma" = "STICSAsomatic"))

load("replication_study/data/qs_item_level_clean_full.Rda")

questionnaires_3 <- questionnaires %>% 
  subset(is.element(ID, incl)) %>% 
  mutate(Measure = ifelse(Measure == "STICSA_T", paste0(Measure, subscale), Measure),
         value = as.numeric(value)) %>% 
  group_by(ID, Measure) %>% 
  summarise(score = sum(value+1)) %>% # +1 bc questionnaires should be coded 1 to K not 0 to K-1
  mutate(questionnaire = recode(Measure,
                                "STICSA_Tc" = "STICSAcognitive",
                                "STICSA_Ts" = "STICSAsomatic",
                                "CAPE" = "CAPEdepressivity",
                                "PID" = "PIDnegative affect"))

comb <- questionnaires_1 %>% 
  rbind(questionnaires_2 %>% select(ID, questionnaire, score), 
        questionnaires_3 %>% select(ID, questionnaire, score), .id = "Study") %>% 
  mutate(score = as.numeric(score))

ggplot(comb, aes(score, fill = Study)) + geom_histogram(alpha = 0.3, position = "identity")+
  facet_wrap(vars(questionnaire), scales = "free")

common_qs <- unique(comb$questionnaire[comb$Study == 3])
df <- comb %>% 
  subset(questionnaire %in% common_qs)

ggplot(df, aes(score, fill = Study)) + geom_histogram(alpha = 0.3, position = "identity")+
  geom_boxplot(aes(y = 0),width = 10, position = "dodge")+
  facet_wrap(vars(questionnaire), scales = "free")

############## Factor loadings ##########

############ Study 1
load("Study1/data/questionnaires_processed.Rda")

df <- read.csv("Study1/factor_loadings__4.csv") %>%
  pivot_longer(cols = 1:4, names_to = "factor", values_to = "loading") %>%
  mutate(item = sapply(item, function(x) {
     if (grepl("STICSA", x)){
      sub("^STICSA(T_\\d+)", "STICSA_\\1", paste0(strsplit(x,"_")[[1]][1], "_", ifelse(as.numeric(strsplit(x, "_")[[1]][2]) < 10, 
                                                                                       as.numeric(strsplit(x, "_")[[1]][2])-1, 
                                                                                       strsplit(x, "_")[[1]][2])))
     } else if (grepl("PID", x)){
       sub("^PID(5_\\d+)", "PID_\\1", paste0(strsplit(x, "_")[[1]][1], "_", as.numeric(strsplit(x, "_")[[1]][2]) -1))
     } else if(grepl("CAPE", x)){
      paste0(strsplit(x,"_")[[1]][1], "_", ifelse(as.numeric(strsplit(x, "_")[[1]][2]) < 16, 
                                                  as.numeric(strsplit(x, "_")[[1]][2])-1, 
                                                  strsplit(x, "_")[[1]][2]))
    } else {x} })) %>% 
  left_join(questionnaires %>% subset(ID == 1 , c(item, questionnaire)), by = "item") %>% 
  mutate(measure = ifelse(is.na(questionnaire), measure, questionnaire))



p1 <- ggplot(df, aes(itemNumber, loading, color = measure, fill = measure)) +
  geom_col()+
  facet_grid(rows = vars(factor)) +
  labs(title = "Factor loadings in Study 1",
       x = "item")
p1
ggsave(p1, file = "plots/factor_loadings_study1.png", width = 10, height = 5)
######## Study 3

load("replication_study/data/loadings4.Rda")

df <- loadings4 %>% 
  mutate(Measure = recode(measure, "CAPE" = "CAPE depressivity",
                          "IUS" = "Intolerance to uncertainty",
                          "PID" = "PID5 negative affect",
                          "PSWQ" = "Penn State Worry Questionnaire",
                          "RRQ" = "Reflection Rumination Questionnaire",
                          "STICSA_Tc" = "STICSA cognitive anxiety",
                          "STICSA_Ts" = "STICSA somatic anxiety"))

p1 <-  ggplot(df, aes(qnum, loading, color = Measure, fill = Measure)) +
  geom_col()+
  facet_grid(rows = vars(factor)) +
  labs(title = "Factor loadings in the replication study",
       x = "item")

p1

ggsave(p1, file = "plots/factor_loadings_study3.png", width = 10, height = 5)

########### correlation matrix of factor scores ##########

heatmap <- function(df, x = x, y = y, limits = c(-1,1)){
  
  ggplot(df, aes(x = x, y = y, fill = cor)) + geom_raster() + 
    scale_fill_gradient2(high = darkBlue, low = red, mid = "white", limits = limits)+
    geom_label(aes(label = round(cor, digits = 2)), fill = "white") +
    scale_x_discrete(expand = c(0.01, 0)) +
    scale_y_discrete(expand = c(0.01, 0))
  
  
  
}

self_cor <- function(df){
  cors <- df %>% 
    select(-ID) %>% 
    cor(use = "pairwise.complete.obs") %>% 
    as.data.frame() %>% 
    mutate(x = rownames(.)) %>% 
    pivot_longer(cols = -x, names_to = "y", values_to = "cor")
  
  return(cors)
  
}

### Study 1

p <- read.csv("Study1/data/factor_scores__4.csv") %>% 
  rename("ID" = subjectID) %>% 
  subset(!grepl("subject", ID)) %>% # remove participants that we pooled with that are not from this study
  self_cor() %>% 
  heatmap() + labs(x = element_blank(),
                   y = element_blank(),
                   title = "Correlations between factor scores in Study 1")

p

ggsave(p, filename = "plots/SIFactorCorsSt1.png")


######### Study 3

p <- read.csv("replication_study/data/fa_scores4_strict.csv") %>% 
  select(-X) %>% 
  self_cor() %>% 
  heatmap() + labs(x = element_blank(),
                   y = element_blank(),
                   title = "Correlations between factor scores in Study 3")

p

ggsave(p, filename = "plots/SIFactorCorsSt3.png")

############ tables for model recovery ############

study1 <- read.csv("Study1/analyses/waics.csv") %>% 
  mutate(Model = factor(fit_model, levels = c("ucb_lcb", "ucb_b0", "ucb_lcb_n"),
                            labels = c("full CB", "CB $\bbeta=0$", "novelty bonus" )),
         Data = factor(data_model, levels = c("CB", "_b0", "_cb_n"),
                             labels = c("full CB", "CB $\bbeta=0$", "novelty bonus")),
         Condition = factor(kraken, levels = c(0,1),
                            labels = c("safe", "risky"))) %>% 
  rename(WAIC = waic,
         SE = se) %>% 
  arrange(Data, Condition) %>% 
  select(Data, Model, Condition, WAIC, SE, best)

library(stargazer)
stargazer(study1, summary = F, rownames = F)

study2 <- read.csv("Study2/analyses/waics.csv") %>% 
  mutate(Model = factor(fit_model, levels = c("ucb_lcb", "ucb_b0", "ucb_lcb_n"),
                        labels = c("full CB", "CB $\bbeta=0$", "novelty bonus" )),
         Data = factor(data_model, levels = c("cb", "cb_b0", "cb_n"),
                       labels = c("full CB", "CB $\bbeta=0$", "novelty bonus")),
         Condition = factor(cond, levels = c(0,1),
                            labels = c("control", "intervention")),
         Timepoint = factor(tp, levels = c(0,1),
                           labels = c("Pre", "Post"))) %>% 
  rename(WAIC = waic,
         SE = se) %>% 
  arrange(Data, Condition, Timepoint) %>% 
  select(Data, Model, Condition, Timepoint, WAIC, SE, best)


stargazer(study2, summary = F, rownames = F)


## replication study

study1 <- read.csv("replication_study/analyses/model_fitting/results/identifiability_waics.csv") %>% 
  mutate(Model = factor(fit_model, levels = c("ucb_lcb", "ucb_b0", "ucb_lcb_n"),
                        labels = c("full CB", "CB $\bbeta=0$", "novelty bonus" )),
         Data = factor(data_model, levels = c("CB", "CB_b0", "CB_n"),
                       labels = c("full CB", "CB $\bbeta=0$", "novelty bonus")),
         Condition = factor(kraken, levels = c(0,1),
                            labels = c("safe", "risky"))) %>% 
  rename(WAIC = waic,
         SE = se) %>% 
  arrange(Data, Condition) %>% 
  select(Data, Model, Condition, WAIC, SE, best)

stargazer(study1, summary = F, rownames = F)
