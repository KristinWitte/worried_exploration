######################### final plots for the combined project (main text) ######################

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






############# Figure 1: learning curves ##############

# first two subplots are screenshots of the task and added in powerpoint

## study 1 rewards over clicks


meanmean <- ddply(Master1[Master1$blocknr != 6, ], .(click, krakenPres), summarize, se = se(na.omit(z)), z = mean(z, na.rm = TRUE))


p1 <- ggplot(data = meanmean, aes(x = click, y = z, color = krakenPres)) +
  geom_point()+
  geom_line(size = 1) +
  geom_linerange(aes(ymin = z - se, ymax = z+se)) +
  scale_x_continuous(breaks = round(seq(1,11, by = 1),1)) +
  labs(title = "Mean rewards over clicks study 1",
       y = "Rewards ± SE",
       x = "Click") +
  scale_color_manual(values = c(darkBlue, red), name = "Condition") +
  theme(legend.position = c(0.9,0.2))
p1

meanmean <- ddply(Master[Master$block != 6, ], .(trial, cond), summarize, se = se(na.omit(z)), z = mean(z, na.rm = TRUE))


p2 <- ggplot(data = meanmean, aes(x = trial, y = z, color = cond)) +
  geom_point()+
  geom_line(size = 1) +
  geom_linerange(aes(ymin = z - se, ymax = z+se)) +
  scale_x_continuous(breaks = round(seq(1,26, by = 1),1)) +
  labs(title = "Mean rewards over clicks study 2",
       y = "Rewards ± SE",
       x = "Click") +
  scale_color_manual(values = c(control, red), name = "Condition") +
  theme(legend.position = c(0.9,0.2))
p2


ggarrange(p1, p2, ncol = 2, nrow = 1, labels = c("C", "D"), widths = c(0.5, 1))

############# Figure 2: Study 1 NUO, NUO ~Q , Study 2 Nerv, NUO over blocks, NUO ~nerv ##############

################ A: St1 NUO

d2<-ddply(Master1[Master1$blocknr != 6, ], ~krakenPres+ID, summarize, mu=mean(unique, na.rm=TRUE), se=se(na.omit(unique)))
p1 <- ggplot(d2, aes(y=mu, x=krakenPres)) +
  geom_half_violin(side = c("l", "r"), aes(fill = krakenPres))+
  geom_boxplot(width = 0.05) +
  geom_line(aes(x = c(rep(c(1.2, 1.8), each = (nrow(d2)/2))), group = ID), alpha = 0.2) + 
  geom_jitter(aes(x = c(rep(c(1.1, 1.9), each = (nrow(d2)/2))), color = krakenPres), alpha = 0.2, width = 0.05)+
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


######### B+C: St1 NUO ~Q

load("Study1/NUOQs.Rda")

main <- rbind(Sc$fixed[c(3), ],Ss$fixed[c(3), ], C$fixed[c(3), ], I$fixed[c(3), ], R$fixed[c(3), ], P$fixed[c(3), ])
df <- data.frame(var = c("cognitive anxiety", "somatic anxiety", "depressivity", "intolerance to uncertainty", "rumination", "negative affect"), Estimate = main[ ,1], lower = main[ ,3], upper = main[ ,4])

p2 <- errorBarPlot(df, title = "Main effects of questionnaires on P(novel)")
p2

# extract interaction with condition

interact <- rbind(Sc$fixed[7, ],Ss$fixed[7, ], C$fixed[c(7), ], I$fixed[c(7), ], R$fixed[c(7), ], P$fixed[c(7), ])
df <- data.frame(var = c("cognitive anxiety", "somatic anxiety", "depressivity", "intolerance to uncertainty", "rumination", "negative affect"), 
                 Estimate = interact[ ,1], lower = interact[ ,3], upper = interact[ ,4])

p3 <- errorBarPlot(df, title = "Interaction effects with condition on P(novel)")

p3

######## D: St2 nervous
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

## E: exploration ~ nervous
load("Study2/nervousByInterv.Rda")

nerv <- summary(model)

main <- nerv$fixed[c(2,7:10), ]
df <- data.frame(var = c("nervousness", "nervousness*condition", "nervousness*time point", "intervention", "nervousness*intervention"), 
                 Estimate = main[ ,1], lower = main[ ,3], upper = main[ ,4])

p4 <- errorBarPlot(df, title = "Effects of nervousness on P(novel)")

p4

### F: St2 NUO
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


ggarrange(p1,p2,p3,Nerv,p4,NUO, nrow = 2, ncol = 3, widths = c(0.6, 0.9, 0.8, 0.6, 0.9, 0.8), labels = "AUTO")


##################### Figure 3: St1 eta ~Q, St2 eta ################# 


############### A+B: St1 eta ~ questionnaires

load("Study1/etaQs.Rda")

# eta is called beta throughout these scripts for convenience of reusing code
View(beta_PID5$fixed)

main <- rbind(beta_STICSAcog$fixed[c(2), ],beta_STICSAsoma$fixed[c(2), ], beta_CAPE$fixed[c(2), ], 
              beta_IUS$fixed[c(2), ], beta_RRQ$fixed[c(2), ], beta_PID5$fixed[c(2), ])
df <- data.frame(var = c("cognitive anxiety", "somatic anxiety", "depressivity", "intolerance to uncertainty", "rumination", "negative affect"), Estimate = main[ ,1], lower = main[ ,3], upper = main[ ,4])

p5 <- errorBarPlot(df, title = expression("Main effects of questionnaires on" ~eta))

p5

# extract interaction with condition

interact <- rbind(beta_STICSAcog$fixed[c(4), ],beta_STICSAsoma$fixed[c(4), ], beta_CAPE$fixed[c(4), ], beta_IUS$fixed[c(4), ], beta_RRQ$fixed[c(4), ], beta_PID5$fixed[c(4), ])
df <- data.frame(var = c("cognitive anxiety", "somatic anxiety", "depressivity", "intolerance to uncertainty", "rumination", "negative affect"), Estimate = interact[ ,1], lower = interact[ ,3], upper = interact[ ,4])
df$var = factor(df$var, levels = df$var, labels = df$var)

p6 <- errorBarPlot(df, title  = expression("Interaction effects with condition on"~eta) )


ggarrange(p5, p6, ncol = 2, widths = c(1, 0.75))


########### C: st2 eta by intervention

df1 <- read.csv("Study2/estimatesCB_n.csv")

df1$tp <- ifelse(df1$tp == 0, "Pre", "Post")
df1$cond <- ifelse(df1$cond == 0, "Control", "Intervention")

df1$tp <- factor(df1$tp, levels = df1$tp, labels = df1$tp)
df1$cond <- factor(df1$cond, levels = df1$cond, labels = df1$cond)

dd <- ddply(df1, ~tp+cond,summarise, eta = meann(beta), se = se(beta)) 

p3 <-  ggplot(dd, aes(tp, eta, color = cond, group = cond)) + geom_line(size = 1.5) +
  geom_linerange(aes(ymin = eta -se, ymax = eta+se), size = 1.5) +
  labs(title = "Intervention effect on the novelty bonus",
       x = "Timepoint",
       y = expression(eta~"parameter"))+
  scale_color_manual(name = "Condition", values = c(control, red))+
  theme(legend.position = c(0.17, 0.17),
        axis.title.y = element_text(margin = margin("l" = 22, "r" = 5)),
        legend.background = element_rect(fill = "transparent"))
p3

ggarrange(p5,p6, p3, ncol = 3, nrow = 1, labels = "AUTO", widths = c(1,0.9, 0.9), align = "h")


