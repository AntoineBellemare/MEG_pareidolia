# Load required libraries
library(lme4)
library(lmerTest)
library(ggplot2)
library(ggeffects)
library(sjPlot)
library(ggpubr)

MEG_atlas <- list(
  CL = 0:23,
  FL = 24:56,
  OL = 57:75,
  PL = 76:96,
  TL = 97:130,
  CR = 131:152,
  FR = 153:185,
  OR = 186:203,
  PR = 204:225,
  TR = 226:258,
  CZ = 259:262,
  Fz = 263:265,
  OZ = 266:268,
  PZ = 269
)


# Specify the fixed effect and the desired effect direction
fixed_effect_name <- "gamma1.z"  # Replace with your desired fixed effect name
effect_direction <- "positive"  # Can be either "positive" or "negative"

# Define the directory for the models
folder <- "C:/Users/Antoine/github/MEG_pareidolia/R_data/results_20231227-224237/" 

# Specify the region(s) of interest
regions_of_interest <- c("CL", "CR", "CZ", "PZ","PL", "PR")  # Example region

# Convert the regions of interest into a list of electrode indices
electrodes_of_interest <- unlist(MEG_atlas[regions_of_interest])

# Initialize variables to store ranges
effect_size_range <- numeric(0)
p_value_range <- numeric(0)
degrees_of_freedom_range <- numeric(0)

# Loop through each electrode of interest
for (elec in electrodes_of_interest) {
  file_name <- sprintf("model_electrode_%03d.rds", elec)
  full_path <- file.path(folder, file_name)
  
  if (file.exists(full_path)) {
    model <- readRDS(full_path)
    print(summary(model))
    if (fixed_effect_name %in% names(fixef(model))) {
      effect_value <- fixef(model)[fixed_effect_name]
      summary_model <- summary(model)
      print(summary_model)
      p_value <- summary_model$coefficients[fixed_effect_name, "Pr(>|z|)"]
      df <- summary_model$coefficients[fixed_effect_name, "df"]
      
      if ((effect_direction == "positive" && effect_value > 0 && p_value < 0.01) ||
          (effect_direction == "negative" && effect_value < 0 && p_value < 0.01)) {
        
        # Store ranges
        effect_size_range <- c(effect_size_range, effect_value)
        p_value_range <- c(p_value_range, p_value)
        degrees_of_freedom_range <- c(degrees_of_freedom_range, df)
      }
    }
  }
}
summary(model)
# Print ranges
print(paste("Effect Size Range: ", min(effect_size_range), "to", max(effect_size_range)))
print(paste("P-Value Range: ", min(p_value_range), "to", max(p_value_range)))
print(paste("Degrees of Freedom Range: ", min(degrees_of_freedom_range), "to", max(degrees_of_freedom_range)))
