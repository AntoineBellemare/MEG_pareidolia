library(dplyr)
library(lme4)

filename <- paste("C:/Users/Antoine/github/MEG_pareidolia/Merged_dataframes/df_ALL_metadata_MEG_sub00to11_epo_long_ALL_sham_behav.csv")
MyData_all <- read.csv(file=filename, header=TRUE, sep=",")

install.packages("lme4")
library(lme4)

# Mixed model
model <- lmer(pos_n_obj ~ sham_cond_num * DAT + (1 + FD | participant), data = MyData_all)

summary(model)


binomial_model <- glmer(parei ~ sham_cond_num * DAT + (1 + FD | participant), 
                        data = MyData_all, 
                        family = binomial)

summary(binomial_model)
