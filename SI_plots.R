################ SI plots and tables ###################


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


############## get data ##################
load("Study1/master.Rda")
Master1 <- Master
Master1$krakenPres<- factor(Master1$krakenPres, levels = c(0,1), labels = c("safe", "risky"))

load("Study2/Master.Rda")
load("Study2/nervous.Rda")

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







#################### Tables for SI ################

#### this code only provides a general formatting and the data for the latex tables, 
# but we still had to do some manual prettifying


######### St1 NUO ~Q
library(stargazer)
load("Study1/NUOQs.Rda")

main <- rbind(Sc$fixed[c(3), ],Ss$fixed[c(3), ], C$fixed[c(3), ], I$fixed[c(3), ], R$fixed[c(3), ], P$fixed[c(3), ])
df <- data.frame(var = c("cognitive anxiety", "somatic anxiety", "depressivity", "intolerance to uncertainty", "rumination", "negative affect"), Estimate = main[ ,1], lower = main[ ,3], upper = main[ ,4])
df$var = factor(df$var, levels = df$var, labels = df$var)

stargazer(df, type = "latex", summary = F, rownames = F, column.labels  = c("Predictor","beta", 
                                                                            "95HDI lower bound", "95HDI upper bound"))

interact <- rbind(Sc$fixed[7, ],Ss$fixed[7, ], C$fixed[c(7), ], I$fixed[c(7), ], R$fixed[c(7), ], P$fixed[c(7), ])
df <- data.frame(var = c("cognitive anxiety", "somatic anxiety", "depressivity", "intolerance to uncertainty", "rumination", "negative affect"), 
                 Estimate = interact[ ,1], lower = interact[ ,3], upper = interact[ ,4])
df$var = factor(df$var, levels = df$var, labels = df$var)

stargazer(df, type = "latex", summary = F, rownames = F, column.labels  = c("Predictor","beta", 
                                                                            "95HDI lower bound", "95HDI upper bound"))


############## Study 1 eta ~ Qs 

load("Study1/etaQs.Rda")

View(beta_PID5$fixed)

main <- rbind(beta_STICSAcog$fixed[c(2), ],beta_STICSAsoma$fixed[c(2), ], beta_CAPE$fixed[c(2), ], beta_IUS$fixed[c(2), ], beta_RRQ$fixed[c(2), ], beta_PID5$fixed[c(2), ])
df <- data.frame(var = c("cognitive anxiety", "somatic anxiety", "depressivity", "intolerance to uncertainty", "rumination", "negative affect"), Estimate = main[ ,1], lower = main[ ,3], upper = main[ ,4])
df$var = factor(df$var, levels = df$var, labels = df$var)

stargazer(df, type = "latex", summary = F, rownames = F, column.labels  = c("Predictor","beta", 
                                                                            "95HDI lower bound", "95HDI upper bound"))


interact <- rbind(beta_STICSAcog$fixed[c(4), ],beta_STICSAsoma$fixed[c(4), ], beta_CAPE$fixed[c(4), ], beta_IUS$fixed[c(4), ], beta_RRQ$fixed[c(4), ], beta_PID5$fixed[c(4), ])
df <- data.frame(var = c("cognitive anxiety", "somatic anxiety", "depressivity", "intolerance to uncertainty", "rumination", "negative affect"), Estimate = interact[ ,1], lower = interact[ ,3], upper = interact[ ,4])
df$var = factor(df$var, levels = df$var, labels = df$var)

stargazer(df, type = "latex", summary = F, rownames = F, column.labels  = c("Predictor","beta", 
                                                                            "95HDI lower bound", "95HDI upper bound"))


############# St1 tau ~Qs

load("Study1/tauQs.Rda")
View(tau_PID5$fixed)

main <- rbind(tau_STICSAcog$fixed[c(2), ],tau_STICSAsoma$fixed[c(2), ], tau_CAPE$fixed[c(2), ], 
              tau_IUS$fixed[c(2), ], tau_RRQ$fixed[c(2), ], tau_PID5$fixed[c(2), ])
df <- data.frame(var = c("cognitive anxiety", "somatic anxiety", "depressivity", "intolerance to uncertainty", "rumination", "negative affect"), Estimate = main[ ,1], lower = main[ ,3], upper = main[ ,4])
df$var = factor(df$var, levels = df$var, labels = df$var)

stargazer(df, type = "latex", summary = F, rownames = F, column.labels  = c("Predictor","beta", 
                                                                            "95HDI lower bound", "95HDI upper bound"))


# extract interaction with condition

interact <- rbind(tau_STICSAcog$fixed[c(4), ],tau_STICSAsoma$fixed[c(4), ], tau_CAPE$fixed[c(4), ], tau_IUS$fixed[c(4), ], tau_RRQ$fixed[c(4), ], tau_PID5$fixed[c(4), ])
df <- data.frame(var = c("cognitive anxiety", "somatic anxiety", "depressivity", "intolerance to uncertainty", "rumination", "negative affect"), Estimate = interact[ ,1], lower = interact[ ,3], upper = interact[ ,4])
df$var = factor(df$var, levels = df$var, labels = df$var)

stargazer(df, type = "latex", summary = F, rownames = F, column.labels  = c("Predictor","beta", 
                                                                            "95HDI lower bound", "95HDI upper bound"))

########## St1 ls ~Qs

load("Study1/lsQs.Rda")

main <- rbind(ls_STICSAcog$fixed[c(2), ],ls_STICSAsoma$fixed[c(2), ], ls_CAPE$fixed[c(2), ], ls_IUS$fixed[c(2), ], ls_RRQ$fixed[c(2), ], ls_PID5$fixed[c(2), ])
df <- data.frame(var = c("cognitive anxiety", "somatic anxiety", "depressivity", "intolerance to uncertainty", "rumination", "negative affect"), 
                 Estimate = main[ ,1], lower = main[ ,3], upper = main[ ,4])
df$var = factor(df$var, levels = df$var, labels = df$var)

stargazer(df, type = "latex", summary = F, rownames = F, column.labels  = c("Predictor","beta", 
                                                                            "95HDI lower bound", "95HDI upper bound"))


# extract interaction with condition

interact <- rbind(ls_STICSAcog$fixed[c(4), ],ls_STICSAsoma$fixed[c(4), ], ls_CAPE$fixed[c(4), ], ls_IUS$fixed[c(4), ], ls_RRQ$fixed[c(4), ], ls_PID5$fixed[c(4), ])
df <- data.frame(var = c("cognitive anxiety", "somatic anxiety", "depressivity", "intolerance to uncertainty", "rumination", "negative affect"), 
                 Estimate = interact[ ,1], lower = interact[ ,3], upper = interact[ ,4])
df$var = factor(df$var, levels = df$var, labels = df$var)

stargazer(df, type = "latex", summary = F, rownames = F, column.labels  = c("Predictor","beta", 
                                                                            "95HDI lower bound", "95HDI upper bound"))

############## replication of findings using distances instead of P(novel) #############

load("Study1/distanceQs.Rda")

main <- rbind(STICSAcog$fixed[rownames(STICSAcog$fixed) == "STICSAcog", ],
              STICSAsoma$fixed[rownames(STICSAsoma$fixed) == "STICSAsoma", ], 
              CAPE_depressed$fixed[rownames(CAPE_depressed$fixed) == "CAPE_depressed", ], 
              IUS$fixed[rownames(IUS$fixed) == "IUS", ], 
              RRQ$fixed[rownames(RRQ$fixed) == "RRQ", ], 
              PID5_negativeAffect$fixed[rownames(PID5_negativeAffect$fixed) == "PID5_negativeAffect", ])

df <- data.frame(var = c("cognitive anxiety", "somatic anxiety", "depressivity", "intolerance to uncertainty", "rumination", "negative affect"), 
                 Estimate = main[ ,1], lower = main[ ,3], upper = main[ ,4])

p2 <- errorBarPlot(df, title = "Main effects of questionnaires on distance between clicks")
p2

# extract interaction with condition

interact <- rbind(STICSAcog$fixed[rownames(STICSAcog$fixed) == "STICSAcog:krakenPres", ],
                  STICSAsoma$fixed[rownames(STICSAsoma$fixed) == "STICSAsoma:krakenPres", ], 
                  CAPE_depressed$fixed[rownames(CAPE_depressed$fixed) == "CAPE_depressed:krakenPres", ], 
                  IUS$fixed[rownames(IUS$fixed) == "IUS:krakenPres", ], 
                  RRQ$fixed[rownames(RRQ$fixed) == "RRQ:krakenPres", ], 
                  PID5_negativeAffect$fixed[rownames(PID5_negativeAffect$fixed) == "PID5_negativeAffect:krakenPres", ])
df <- data.frame(var = c("cognitive anxiety", "somatic anxiety", "depressivity", "intolerance to uncertainty", "rumination", "negative affect"), 
                 Estimate = interact[ ,1], lower = interact[ ,3], upper = interact[ ,4])

p3 <- errorBarPlot(df, title = "Interaction effects with condition")

p3

dist <- ggarrange(p2,p3, ncol = 2, nrow = 1)
dist

ggsave("plots/SIdistanceQs.png", dist, width = 17.5, height = 3)

############### table of these stats 

df <- data.frame(var = c("cognitive anxiety", "somatic anxiety", "depressivity", "intolerance to uncertainty", "rumination", "negative affect"), 
                 Estimate = main[ ,1], lower = main[ ,3], upper = main[ ,4])
df$var = factor(df$var, levels = df$var, labels = df$var)

stargazer(df, type = "latex", summary = F, rownames = F, column.labels  = c("Predictor","beta", 
                                                                            "95HDI lower bound", "95HDI upper bound"))


df <- data.frame(var = c("cognitive anxiety", "somatic anxiety", "depressivity", "intolerance to uncertainty", "rumination", "negative affect"), 
                 Estimate = interact[ ,1], lower = interact[ ,3], upper = interact[ ,4])
df$var = factor(df$var, levels = df$var, labels = df$var)

stargazer(df, type = "latex", summary = F, rownames = F, column.labels  = c("Predictor","beta", 
                                                                            "95HDI lower bound", "95HDI upper bound"))


################### model comparison plots ##############

############ model comparison St 2

df <- read.csv("Study2/groupBMC_results_incl_cb_n.csv")

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

exc <- read.csv("Study1/groupBMC_results_incl_cb_n.csv")
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

############# parameter recovery plots ##########

### scatter plot

trueParameters <- read.csv("Study1/estimatesCB_n.csv")%>% 
  pivot_longer(cols = 3:5, names_to = "Parameter", values_to = "generating")
recoveredParameters <- read.csv("Study1/recoveredEstimates.csv")%>% 
  pivot_longer(cols = 3:5, names_to = "Parameter", values_to = "recovered")

parameters <- trueParameters %>% 
  left_join(recoveredParameters, by = c("ID", "kraken_present", "Parameter")) %>% 
  mutate(Condition = factor(kraken_present, levels = c(0,1), labels = c("safe", "risky")),
         Parameter = recode(Parameter, "beta" = "eta", "ls" = "lambda"))

p1 <- ggplot(parameters, aes(generating, recovered, color = Condition)) + 
  geom_jitter(alpha = 0.3) +
  facet_wrap(vars(Parameter), scales = "free") +
  geom_abline(aes(intercept = 0, slope = 1))+
  scale_color_manual(values = c(darkBlue, red))+
  ggtitle("Parameter recovery in Study 1")

p1

ggsave("plots/SIParameterRecoveryScatterStudy1.png", plot = p1, width = 9, height = 3)


####### confusion matrix plot

parameters <- read.csv("Study1/estimatesCB_n.csv") %>% 
  left_join(read.csv("Study1/recoveredEstimates.csv"), by = c("ID", "kraken_present"))

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

ggsave("plots/SIParameterRecoveryGridStudy1.png", plot = p2, width = 9, height = 3)



##### same for study 2


### scatter plot

trueParameters <- read.csv("Study2/estimatesCB_n.csv")%>% 
  pivot_longer(cols = 4:6, names_to = "Parameter", values_to = "generating")
recoveredParameters <- read.csv("Study2/recoveredEstimates.csv")%>% 
  subset(select = -X) %>% 
  pivot_longer(cols = 4:6, names_to = "Parameter", values_to = "recovered")

parameters <- trueParameters %>% 
  left_join(recoveredParameters, by = c("ID", "cond","tp", "Parameter")) %>% 
  mutate(Condition = factor(cond, levels = c(0,1), labels = c("control", "intervention")),
         Timepoint = factor(tp, levels = c(0,1), labels = c("Pre", "Post")),
         Parameter = recode(Parameter, "beta" = "eta", "ls" = "lambda"))


pa <- ggplot(parameters[parameters$Timepoint == "Pre", ], aes(generating, recovered, color = Condition)) + 
  geom_jitter() +
  facet_wrap(vars(Parameter), scales = "free") +
  geom_abline(aes(intercept = 0, slope = 1))+
  scale_color_manual(values = c(control, red)) +
  ggtitle("Baseline")

pa

pb <- ggplot(parameters[parameters$Timepoint == "Post", ], aes(generating, recovered, color = Condition)) + 
  geom_jitter() +
  facet_wrap(vars(Parameter), scales = "free") +
  geom_abline(aes(intercept = 0, slope = 1))+
  scale_color_manual(values = c(control, red))+
  ggtitle("After intervention")

pb

p1 <- ggarrange(pa, pb, ncol = 1, nrow = 2,common.legend = T, legend = "bottom") + ggtitle("Parameter recovery in Study 2")

p1

ggsave("plots/SIParameterRecoveryScatterStudy2.png", plot = p1, width = 9, height = 6)


####### confusion matrix plot

parameters <- read.csv("Study2/estimatesCB_n.csv") %>% 
  left_join(read.csv("Study2/recoveredEstimates.csv"), by = c("ID", "cond", "tp"))


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

ggsave("plots/SIParameterRecoveryGridStudy2.png", plot = p2, width = 9, height = 6)




