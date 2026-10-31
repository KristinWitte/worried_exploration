####### distributions of questionnaire scores in all three studies #########
# Draws the SI histograms and Q-Q plots of all questionnaire scores, prints the SI table
# of descriptives and normality tests, and prints the correlations reported in the main text.
# Run from the repo root. Study 3 uses the strict inclusion criteria.
# Scores are total scores on the scale used in the SI tables of questionnaire scores.

library(plyr)
library(tidyverse)
library(patchwork)
library(stargazer)

one_per_id <- function(df) df[!duplicated(df$ID), ]

###### Study 1: questionnaires analysed in the main text #########
load("Study1/data/master.Rda")
m1 <- one_per_id(Master)
s1 <- data.frame(
  ID = m1$ID,
  `STICSA cognitive` = m1$STICSAcog + 10,              # stored as 0-based sum of 10 items
  `STICSA somatic` = m1$STICSAsoma + 11,               # stored as 0-based sum of 11 items
  IUS = m1$IUS + 12,                                   # stored as 0-based sum of 12 items
  RRQ = (m1$RRQ - 0.25) * 12,                          # stored as 0-based item mean + 1.25
  `CAPE depressivity` = m1$CAPE_depressed * 8,         # stored as 1-based item mean of 8 items
  `PID-5 negative affect` = m1$PID5_negativeAffect + 5, # stored as 0-based sum of 5 items
  check.names = FALSE)

###### Study 1: other questionnaires (SI only), summed from item level #########
load("Study1/data/questionnaires_processed.Rda")
s1_other <- questionnaires %>%
  filter(questionnaire %in% c("AQ", "ASRS", "CAPEpositive", "CAPEnegative", "ESI",
                              "PIDantagonism", "PIDdetachment", "PIDdisinhibition", "PIDpsychoticism"),
         ID %in% s1$ID) %>%
  group_by(ID, questionnaire) %>%
  summarise(score = sum(score) + n(), .groups = "drop") %>% # items coded from 0, SI tables count from 1
  mutate(questionnaire = dplyr::recode(questionnaire,
                                       AQ = "AQ-10", CAPEpositive = "CAPE positive", CAPEnegative = "CAPE negative",
                                       ESI = "ESI-bf (selected subscales)", PIDantagonism = "PID-5 antagonism",
                                       PIDdetachment = "PID-5 detachment", PIDdisinhibition = "PID-5 disinhibition",
                                       PIDpsychoticism = "PID-5 psychoticism"))

###### Study 2 #########
load("Study2/data/Master.Rda")
m2 <- one_per_id(Master)
s2 <- data.frame(
  ID = m2$ID,
  `STICSA cognitive` = m2$STICSAcog + 10,
  `STICSA somatic` = m2$STICSAsoma + 11,
  `PHQ-9` = m2$PHQ,
  `MCQ-30 total` = m2$MCQ,
  `MCQ positive beliefs` = m2$MCQpos,
  `MCQ negative beliefs` = m2$MCQneg,
  `MCQ cognitive confidence` = m2$MCQconf,
  `MCQ need for control` = m2$MCQcontrol,
  `MCQ cognitive self-consciousness` = m2$MCQselfcons,
  `MWQ frequency` = m2$MWQfreq,
  `MWQ belief` = m2$MWQbelief / 7,                     # stored as sum of 7 ratings (0-100)
  `CAS-1 (start)` = m2$CASpre,
  `CAS-1 (end)` = m2$CASpost,
  check.names = FALSE)

###### Study 3 #########
load("replication_study/data/Master_strict.Rda")
m3 <- one_per_id(Master)
# stored as 0-based item means
s3 <- data.frame(
  ID = m3$ID,
  `STICSA cognitive` = (m3$STICA_T_c + 1) * 10,
  `STICSA somatic` = (m3$STICA_T_s + 1) * 11,
  IUS = (m3$IUS + 1) * 12,
  RRQ = (m3$RRQ + 1) * 12,
  `CAPE depressivity` = (m3$CAPE + 1) * 7,
  `PID-5 negative affect` = (m3$PID + 1) * 5,
  PSWQ = (m3$PSWQ + 1) * 16,
  check.names = FALSE)

studies <- c("Study 1", "Study 1 (other questionnaires)", "Study 2", "Study 3")

long <- bind_rows(
  pivot_longer(s1, -ID, names_to = "questionnaire", values_to = "score") %>% mutate(study = studies[1]),
  s1_other %>% mutate(study = studies[2]),
  pivot_longer(s2, -ID, names_to = "questionnaire", values_to = "score") %>% mutate(study = studies[3]),
  pivot_longer(s3, -ID, names_to = "questionnaire", values_to = "score") %>% mutate(study = studies[4]))

# possible score ranges on the scale used here
possible <- tribble(
  ~questionnaire, ~pos_min, ~pos_max,
  "STICSA cognitive", 10, 40, "STICSA somatic", 11, 44, "IUS", 12, 60, "RRQ", 12, 60,
  "PID-5 negative affect", 5, 20, "PSWQ", 16, 80, "PHQ-9", 0, 27, "MCQ-30 total", 0, 90,
  "MCQ positive beliefs", 0, 18, "MCQ negative beliefs", 0, 18, "MCQ cognitive confidence", 0, 18,
  "MCQ need for control", 0, 18, "MCQ cognitive self-consciousness", 0, 18, "MWQ frequency", 0, 21,
  "MWQ belief", 0, 100, "CAS-1 (start)", 0, 100, "CAS-1 (end)", 0, 100, "AQ-10", 10, 40,
  "ASRS", 18, 90, "CAPE positive", 20, 80, "CAPE negative", 14, 56, "ESI-bf (selected subscales)", 39, 156,
  "PID-5 antagonism", 5, 20, "PID-5 detachment", 5, 20, "PID-5 disinhibition", 5, 20, "PID-5 psychoticism", 5, 20) %>%
  mutate(study = NA_character_) %>%
  bind_rows(tibble(questionnaire = "CAPE depressivity", pos_min = c(8, 7), pos_max = c(32, 28),
                   study = c(studies[1], studies[4])))

long <- bind_rows(
  long %>% filter(questionnaire != "CAPE depressivity") %>% left_join(select(possible, -study), by = "questionnaire"),
  long %>% filter(questionnaire == "CAPE depressivity") %>% left_join(possible, by = c("questionnaire", "study")))
stopifnot(!anyNA(long$pos_min), all(long$score >= long$pos_min), all(long$score <= long$pos_max))

###### descriptives and normality tests #########
skew <- function(x) { d <- x - mean(x); mean(d^3) / mean(d^2)^1.5 }
exkurt <- function(x) { d <- x - mean(x); mean(d^4) / mean(d^2)^2 - 3 }

desc <- long %>%
  group_by(study, questionnaire) %>%
  summarise(N = n(), pos_min = first(pos_min), pos_max = first(pos_max),
            mean = mean(score), SD = sd(score),
            pct_at_min = 100 * mean(score == pos_min),
            skewness = skew(score), excess_kurtosis = exkurt(score),
            W = shapiro.test(score)$statistic, p = shapiro.test(score)$p.value,
            .groups = "drop") %>%
  mutate(study = factor(study, levels = studies)) %>%
  arrange(study, questionnaire)

cat("Scores:", nrow(desc), "| Shapiro-Wilk p < .05:", sum(desc$p < .05),
    "| max |skewness|:", round(max(abs(desc$skewness)), 2),
    "| max |excess kurtosis|:", round(max(abs(desc$excess_kurtosis)), 2), "\n")
print(as.data.frame(desc %>% filter(questionnaire == "PHQ-9") %>% select(study, pct_at_min)))

table_out <- desc %>%
  transmute(Study = as.character(study), Questionnaire = questionnaire,
            Range = paste0(pos_min, "--", pos_max),
            `Mean (SD)` = sprintf("%.1f (%.1f)", mean, SD),
            Skewness = sprintf("%.2f", skewness),
            `Excess kurtosis` = sprintf("%.2f", excess_kurtosis),
            W = sprintf("%.2f", W),
            p = ifelse(p < .001, "< .001", sprintf("%.3f", p)))
table_out$Study[duplicated(table_out$Study)] <- ""

table_tex <- capture.output(
  stargazer(table_out, summary = FALSE, rownames = FALSE,
            title = "Distributions of questionnaire scores. Range: possible range of the total score. W and p: Shapiro-Wilk test. Study 3: strict inclusion criteria.",
            label = "tab:QsDistributions"))
# typeset negative numbers as in the other SI tables
cat(gsub("(^| )-([0-9])", "\\1$-$\\2", table_tex), sep = "\n")

###### correlations reported in the main text #########
report_cor <- function(x, y, label) {
  ct <- cor.test(x, y)
  cat(sprintf("%s: r = %.2f, 95%%CI: %.2f, %.2f, p = %.2g\n", label, ct$estimate, ct$conf.int[1], ct$conf.int[2], ct$p.value))
}
report_cor(s1$`CAPE depressivity`, s1$`STICSA cognitive`, "Study 1 CAPE depressivity ~ STICSA cognitive")
report_cor(s3$`CAPE depressivity`, s3$`STICSA cognitive`, "Study 3 CAPE depressivity ~ STICSA cognitive")
report_cor(s2$`PHQ-9`, s2$`STICSA cognitive`, "Study 2 PHQ-9 ~ STICSA cognitive")

###### plots #########
study_block <- function(s, type) {
  d <- long %>% filter(study == s)
  if (type == "hist") {
    normal <- d %>% group_by(questionnaire) %>%
      reframe(x = seq(min(score), max(score), length.out = 200), y = dnorm(x, mean(score), sd(score)))
    p <- ggplot(d, aes(score)) +
      geom_histogram(aes(y = after_stat(density)), fill = "grey70", colour = "white",
                     binwidth = function(x) if (all(x == round(x))) max(1, round(diff(range(x)) / 25)) else diff(range(x)) / 25) +
      geom_line(data = normal, aes(x, y), colour = "#b2182b", linewidth = 0.6) +
      labs(x = "Total score", y = "Density")
  } else {
    p <- ggplot(d, aes(sample = score)) +
      stat_qq(size = 0.5, alpha = 0.6) +
      stat_qq_line(colour = "#b2182b") +
      labs(x = "Theoretical quantile", y = "Total score")
  }
  p + facet_wrap(~questionnaire, scales = "free", ncol = 5, labeller = label_wrap_gen(22)) +
    ggtitle(s) +
    theme_classic(base_size = 9) +
    theme(strip.background = element_blank(), strip.text = element_text(size = 7.5),
          plot.title = element_text(face = "bold"))
}

rows <- sapply(studies, function(s) ceiling(n_distinct(long$questionnaire[long$study == s]) / 5))

for (type in c("hist", "qq")) {
  fig <- wrap_plots(lapply(studies, study_block, type = type), ncol = 1, heights = rows)
  name <- ifelse(type == "hist", "SIQuestionnaireDistributions", "SIQuestionnaireQQ")
  for (ext in c("png", "svg")) {
    ggsave(file.path("plots/SI", paste0(name, ".", ext)), fig, width = 9, height = 1.75 * sum(rows) + 1, dpi = 300)
  }
}
