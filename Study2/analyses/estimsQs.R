####### parameter estimates as a function of questionnaire scores #########

library(brms)

###### importing and formatting data #########
load("Study2/data/Master.Rda")
load("Study2/data/questionnairesPre.Rda")


estims <- read.csv("Study2/data/estimatesCB_n.csv")
estims$tp <- estims$tp -0.5
estims$cond <- estims$cond - 0.5

# the parameter eta is called beta throughout this script for convenience of recycling code

estims$ls <- scale(estims$ls, center = T, scale = F)
estims$tau <- scale(estims$tau, center = T, scale = F)
estims$beta <- scale(estims$beta, center = T, scale = F)

# get age and sex
estims$age <- scale(questionnairesPre$age[match(estims$ID, questionnairesPre$ID)])
estims$gender <- questionnairesPre$Sex_0[match(estims$ID, questionnairesPre$ID)]



# get the questionnaire of interest

questionnaires <- c("STICSAcog", "STICSAsoma", "MCQ", "MCQpos", "MCQneg", "MCQconf",
                    "MCQcontrol", "MCQselfcons", "MWQbelief", "MWQbelief", "CASpre", "CASpost", "CASchange", "PHQ")

parameters <- c("ls", "beta", "tau")

combs <- data.frame(q = rep(questionnaires, length(parameters)),
                    p = rep(parameters, each = length(questionnaires)))

## get SLURM array ID
# this script is optimised to work on a high performance cluster such that 
# all regressions run in parallel

task_id <- as.numeric(commandArgs(TRUE)[1])

q <- combs$q[task_id]
p <- combs$p[task_id]
print(q)
print(p)

estims$Q <- scale(Master[match(estims$ID, Master$ID),colnames(Master) == q])

############# actual analyses #####################


# create a directory to save the results if it doesn't exist yet

if (!file.exists("Study2/analyses/CB_nQ")){
  
  dir.create(file.path("Study2/analyses/CB_nQ"))}


equation <- as.formula(paste(p, "~Q* cond * tp + age + gender + (1|ID)"))


# fit the model we want at this array job
model <- brm(equation, 
             data = estims,
             iter = 4000,
             cores = 4,
             chains = 4, 
             control = list(adapt_delta = 0.99, max_treedepth = 15))

# give the output the name of the questionnaire we are using here
assign(paste(p,q,sep = "_"), summary(model))

# save the whole thing so we can make the visualisations locally
save.image(paste("Study2/analyses/CB_nQ/",p,q,".Rdata", sep = ""))
