######################### final plots for the combined project (main text) ######################
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


############## get data ##################
load("Study1/data/master.Rda")
Master1 <- Master
Master1$krakenPres<- factor(Master1$krakenPres, levels = c(0,1), labels = c("safe", "risky"))

load("replication_study/data/Master_strict.Rda")
Master3 <- Master
Master3$krakenPresent<- factor(Master3$krakenPresent, levels = c(0,1), labels = c("safe", "risky"))

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

Master3$unique<-ave(paste(Master3$x, Master3$y), paste(Master3$ID, 'x', Master3$block), FUN=duplicated)
Master3$unique<-ifelse(Master3$unique==TRUE, 0, 1)
Master3$unique[is.na(Master3$z)] <- NA

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




############# Figure 1: learning curves ##############

# first two subplots are screenshots of the task and added in powerpoint

## study 1 rewards over clicks


meanmean <- ddply(Master1[Master1$blocknr != 6, ], .(click, krakenPres), summarize, se = se(na.omit(z)), z = mean(z, na.rm = TRUE))
meanmean$click <- meanmean$click -1

p1 <- ggplot(data = meanmean, aes(x = click, y = z, color = krakenPres)) +
  geom_point()+
  geom_line(size = 1) +
  geom_linerange(aes(ymin = z - se, ymax = z+se)) +
  scale_x_continuous(breaks = round(seq(0,10, by = 1),1)) +
  labs(title = "Mean rewards over clicks study 1",
       y = "Rewards ± SE",
       x = "Click") +
  scale_color_manual(values = c(darkBlue, red), name = "Condition") +
  theme(legend.position = c(0.9,0.2))
p1

meanmean <- ddply(Master, .(trial, cond), summarize, se = se(na.omit(z)), z = mean(z, na.rm = TRUE))
meanmean$trial <- meanmean$trial -1

p2 <- ggplot(data = meanmean, aes(x = trial, y = z, color = cond)) +
  geom_point()+
  geom_line(size = 1) +
  geom_linerange(aes(ymin = z - se, ymax = z+se)) +
  scale_x_continuous(breaks = round(seq(0,25, by = 1),1)) +
  labs(title = "Mean rewards over clicks study 2",
       y = "Rewards ± SE",
       x = "Click") +
  scale_color_manual(values = c(control, red), name = "Condition") +
  theme(legend.position = c(0.9,0.2))
p2


meanmean <- ddply(Master3, .(trial, krakenPresent), summarize, se = se(na.omit(z)), z = mean(z, na.rm = TRUE))


p3 <- ggplot(data = meanmean, aes(x = trial, y = z, color = krakenPresent)) +
  geom_point()+
  geom_line(size = 1) +
  geom_linerange(aes(ymin = z - se, ymax = z+se)) +
  scale_x_continuous(breaks = round(seq(0,10, by = 1),1)) +
  labs(title = "Mean rewards over clicks replication",
       y = "Rewards ± SE",
       x = "Click") +
  scale_color_manual(values = c(darkBlue, red), name = "Condition") +
  theme(legend.position = c(0.9,0.2))
p3


fig1 <- ggarrange(p1, p2, p3, ncol = 3, nrow = 1, labels = c("C", "D", "E"), widths = c(0.5, 1, 0.5))
fig1

ggsave(plot = fig1, filename = "plots/Fig1CDE.png", width = 19, height = 4)

############# Figure 2: Model agnostic results ##############

################ A: Study 1 P(novel)

d2<-ddply(Master1[Master1$blocknr != 6, ], ~krakenPres+ID, summarize, mu=mean(unique, na.rm=TRUE), se=se(na.omit(unique)))
n_lines <- nrow(d2)/2
p1 <- ggplot(d2, aes(y=mu, x=krakenPres)) +
  geom_half_violin(side = c("l", "r"), aes(fill = krakenPres))+
  geom_boxplot(width = 0.05) +
  geom_line(aes(x = rep(c(1.2, 1.8), each = n_lines), group = ID), alpha = 0.2) + 
  geom_jitter(aes(x = rep(c(1.1, 1.9), each = n_lines), color = krakenPres), alpha = 0.2, width = 0.05)+
  scale_fill_manual(name = "Condition", values = c(darkBlue, red))+
  scale_color_manual(name = "Condition", values = c(darkBlue, red))+
  #title
  labs(title = "Proportion of novel options selected", 
       x = "Condition", 
       y = "P(novel)")+
  theme(legend.position = "none")+
  #scale_x_discrete(labels = c("safe", "risky"))+
  #adjust text size
  scale_y_continuous(expand = c(0, 0)) 

p1

################# B: Study 3: nervousness in conditions

d2<-ddply(Master3, ~krakenPresent+ID, summarize, mu=mean(as.numeric(nervous), na.rm=TRUE), se=se(na.omit(nervous)))
p2 <- ggplot(d2, aes(y=mu, x=krakenPresent)) +
  geom_half_violin(side = c("l", "r"), aes(fill = krakenPresent))+
  geom_boxplot(width = 0.05) +
  geom_line(aes(x = c(rep(c(1.2, 1.8), each = (nrow(d2)/2))), group = ID), alpha = 0.2) + 
  geom_jitter(aes(x = c(rep(c(1.1, 1.9), each = (nrow(d2)/2))), color = krakenPresent), alpha = 0.2, width = 0.05)+
  scale_fill_manual(name = "Condition", values = c(darkBlue, red))+
  scale_color_manual(name = "Condition", values = c(darkBlue, red))+
  #title
  labs(title = "Nervousness", 
       x = "Condition", 
       y = "Nervousness")+
  theme(legend.position = "none")+
  #scale_x_discrete(labels = c("safe", "risky"))+
  #adjust text size
  scale_y_continuous(expand = c(0, 0)) 

p2

########### C: Study 3:  P(novel) by nervousness

load("replication_study/analysis/nerv_pnovel_strict.Rda")
view(nerv_pnovel$fixed)
selected_rows <- c("nervous:prev_z", "krakenPresent:nervous", "prev_z", "krakenPresent", "nervous")
main <- nerv_pnovel$fixed[selected_rows, ]
df <- data.frame(var = c("nervousness* prev. reward","nervousness * condition","prev. reward", "condition", "nervousness"), 
                 Estimate = main[ ,1], lower = main[ ,3], upper = main[ ,4])

p3 <- errorBarPlot(df, title = "Predicting P(novel) in replication")

p3


########## D: Study 2 nervousness over rounds of the task
nervous$round <- rep(c(2, 4, 6, 7, 9, 11), nrow(nervous)/6)
df <- ddply(nervous, ~cond+round, summarise, se = se(nervous), nervousness = meann(nervous))
df$block <- rep(c(-5, -3, -1, 1, 3, 5),2)

Nerv <- ggplot(df, aes(block, nervousness, color = cond)) + geom_line(size = 1.5) +
  geom_linerange(aes(ymin = nervousness -se, ymax = nervousness+se), size = 1.5) +
  geom_point(size = 1.5) +
  geom_vline(xintercept = 0)+
  labs(title = "Nervousness",
       x = "Block since intervention",
       y = "Nervousness ± SE")+
  scale_x_continuous(breaks = c(seq(-5,-1), seq(1,5)))+
  scale_color_manual(name = "Condition", values = c(control, red))+
  theme(legend.position = c(0.2, 0.2),
        axis.title.y = element_text(margin = margin("l" = 7, "r"= 5)),
        legend.background = element_rect(fill = "transparent"))

Nerv
############## E: St2 P(novel) over rounds of the task

df <- ddply(Master, ~cond+block, summarise, se = se(unique), Punique = mean(na.omit(unique)))
df$block <- rep(c(seq(-5,-1), seq(1,5)),2)

NUO <- ggplot(df, aes(block, Punique, color = cond)) + geom_line(size = 1.5) +
  geom_linerange(aes(ymin = Punique -se, ymax = Punique+se), size = 1.5) +
  geom_point(size = 1.5) +
  geom_vline(xintercept = 0)+
  labs(title = "Proportion of novel options selected",
       x = "Block since intervention",
       y = "P(novel) ± SE")+
  scale_x_continuous(breaks = c(seq(-5,-1), seq(1,5)))+
  scale_color_manual(name = "Condition", values = c(control, red))+
  theme(legend.position = c(0.2, 0.2),
        axis.title.y = element_text(margin = margin("l" = 22, "r" = 5)),
        legend.background = element_rect(fill = "transparent"))

NUO

############### F: Study 2 exploration ~ nervousness
load("Study2/analysis/nervousByInterv.Rda")

nerv <- summary(model)
nerv$fixed
selected_rows <- c("nervous:cond:tp", "cond:tp", "nervous:tp", "nervous:prev_z",
                   "nervous:cond", "prev_z", "cond", "nervous")
main <- nerv$fixed[selected_rows, ]
df <- data.frame(var = c("nervousness*intervention","intervention", "nervousness*time point",
                         "nervousness * prev.reward", "nervousness*condition","prev. reward","condition","nervousness"), 
                 Estimate = main[ ,1], lower = main[ ,3], upper = main[ ,4])

p4 <- errorBarPlot(df, title = "Effects of nervousness on P(novel)")

p4


# first put together plots that will be underneath each other to align them bc otherwise the alignment is super difficult
left <- ggarrange(p1, Nerv, nrow = 2, heights = c(0.8, 1), labels = c("A", "D"), align = "hv")
left

middle <- ggarrange(p2, NUO, nrow = 2, heights = c(0.8, 1), labels = c("B", "E"),align = "hv")
middle

right <- ggarrange(p3, p4, nrow = 2, heights = c(0.8,1),labels = c("C", "F"), align = "hv")
right

fig2 <- ggarrange(left, middle, right, ncol = 3,widths = c(0.5, 0.5, 0.8), align = "hv")
fig2

ggsave("plots/Fig2.png", plot = fig2, width = 19, height = 6)


##################### Figure 3: Modelling results ################# 

################ A Study 3: eta ~ nervous

load("replication_study/analysis/estims_nervous.Rda")

eta$fixed
selected_rows <- c("krakenPresent:nervous", "krakenPresent", "nervous")
main <- eta$fixed[selected_rows, ]
main
df <- data.frame(var = c("nervousness*condition","condition","nervousness"), 
                 Estimate = main[ ,1], lower = main[ ,3], upper = main[ ,4])

p1 <- errorBarPlot(df, title = expression("Effects of nervousness on"~eta))

p1

############# B Study 3: ls ~  nervous
ls$fixed
main <- ls$fixed[selected_rows, ]
main
df <- data.frame(var = c("nervousness*condition","condition","nervousness"), 
                 Estimate = main[ ,1], lower = main[ ,3], upper = main[ ,4])

p2 <- errorBarPlot(df, title = expression("Effects of nervousness on"~lambda))

p2



############## C: Study 2:  eta by intervention

df1 <- read.csv("Study2/data/estimatesCB_n.csv")

df1$tp <- ifelse(df1$tp == 0, "Pre", "Post")
df1$cond <- ifelse(df1$cond == 0, "Control", "Intervention")

df1$tp <- factor(df1$tp, levels = df1$tp, labels = df1$tp)
df1$cond <- factor(df1$cond, levels = df1$cond, labels = df1$cond)

dd <- ddply(df1, ~tp+cond,summarise, eta = meann(beta), se = se(beta)) 

p3 <-  ggplot(dd, aes(tp, eta, color = cond, group = cond)) + geom_line(size = 1.5) +
  geom_linerange(aes(ymin = eta -se, ymax = eta+se), size = 1.5) +
  labs(title = expression("Intervention effect on"~eta),
       x = "Timepoint",
       y = expression(eta~"parameter"))+
  scale_color_manual(name = "Condition", values = c(control, red))+
  theme(legend.position = c(0.17, 0.17),
        axis.title.y = element_text(margin = margin("l" = 22, "r" = 5)),
        legend.background = element_rect(fill = "transparent"))
p3

fig3 <-  ggarrange(p3,p1,p2, ncol = 3, nrow = 1, labels = "AUTO", widths = c(0.8,1,1), align = "h")
fig3

ggsave("plots/Fig3Novelty.png", plot = fig3, width = 19, height = 3.5)

