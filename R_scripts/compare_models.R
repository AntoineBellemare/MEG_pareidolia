library(lme4)
library(lmerTest)
remove.packages("rlang")
install.packages("rlang")
install.packages("vctrs")

install.packages("lmerTest", dependencies = TRUE)  # Install 'lmerTest' package
library(lmerTest)
# Define the directories for the two sets of models
folder1 <- "C:/Users/Antoine/github/MEG_pareidolia/R_data/RT_fooofed_single_trial_simple/"
folder2 <- "C:/Users/Antoine/github/MEG_pareidolia/R_data/results_20240106-000102/" 

# Create a data frame to store the results
results <- data.frame(
  Electrode = integer(270),
  LRT_Statistic = numeric(270),
  P_Value = numeric(270),
  AIC_Model1 = numeric(270),
  AIC_Model2 = numeric(270),
  AIC_Difference = numeric(270),
  BIC_Model1 = numeric(270),
  BIC_Model2 = numeric(270),
  BIC_Difference = numeric(270),
  stringsAsFactors = FALSE
)
model1
# Loop over each electrode
for (i in 0:269) {
  # Construct the filenames for the .rds files
  model1_file <- paste0(folder1, "model_electrode_", i, ".rds")
  model2_file <- paste0(folder2, "model_electrode_", i, ".rds")
  
  # Load the models
  model1 <- readRDS(model1_file)
  model2 <- readRDS(model2_file)
  
  # Perform LRT
  lrt <- anova(model1, model2)
  
  # Extract LRT statistic and p-value
  lrt_stat <- lrt$Chisq[2] # Assuming the second model is the one being tested
  p_val <- lrt$`Pr(>Chisq)`[2]
  
  # Calculate AIC and BIC for both models
  aic_model1 <- AIC(model1)
  aic_model2 <- AIC(model2)
  bic_model1 <- BIC(model1)
  bic_model2 <- BIC(model2)
  
  # Calculate differences
  aic_diff <- aic_model2 - aic_model1
  bic_diff <- bic_model2 - bic_model1
  
  # Add the results to the data frame
  results[i,] <- c(i, lrt_stat, p_val, aic_model1, aic_model2, aic_diff, bic_model1, bic_model2, bic_diff)
}

# Save the results to a CSV file
write.csv(results, "Model_comparison_results.csv", row.names = FALSE)

# Print the results
print(results)

# Filter the results for electrodes 120 to 269
filtered_results <- results[results$Electrode >= 120 & results$Electrode <= 269, ]

# Print the filtered results
print(filtered_results)

