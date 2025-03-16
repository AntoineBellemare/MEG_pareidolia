# Load required libraries
library(lme4)
library(lmerTest)
library(ggplot2)
library(ggeffects)
library(sjPlot)
library(ggpubr)

# Define the atlas in R
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
fixed_effect_name <- "parei:DAT.z:bloc"  # Replace with your desired fixed effect name
effect_direction <- "positive"  # Can be either "positive" or "negative"

# Define the directory for the models
folder <- "C:/Users/Antoine/github/MEG_pareidolia/R_data/RS_LZ/" 

# Specify the region(s) of interest
regions_of_interest <- c("OL", "OR", "CL", "CR", "TL", "TR", "PL", "PR", "FL", "FR")  # Example region

# Convert the regions of interest into a list of electrode indices
electrodes_of_interest <- unlist(MEG_atlas[regions_of_interest])

# Initialize a list to store the predictions for the significant models
all_preds <- list()

# Loop through each electrode of interest
for (elec in electrodes_of_interest) {
  # Construct the file name based on the electrode index
  file_name <- sprintf("model_electrode_%03d.rds", elec)
  full_path <- file.path(folder, file_name)
  
  # Check if the file exists
  if (file.exists(full_path)) {
    model <- readRDS(full_path)
    
    # Check if the specified fixed effect exists in the model
    if (fixed_effect_name %in% names(fixef(model))) {
      effect_value <- fixef(model)[fixed_effect_name]
      
      # Check if the effect is significant in the specified direction
      summary_model <- summary(model)
      p_value <- summary_model$coefficients[fixed_effect_name, "Pr(>|t|)"]
      
      if ((effect_direction == "positive" && effect_value > 0 && p_value < 0.001) ||
          (effect_direction == "negative" && effect_value < 0 && p_value < 0.001)) {
        
        # Get predictions for the triple interaction
        preds <- ggpredict(model, terms = c("bloc[all]", "parei[all]", "DAT.z[all]"), ci.lvl=0.01)
        
        # Add predictions to the list if not null
        if (!is.null(preds)) {
          all_preds[[length(all_preds) + 1]] <- preds
        }
      }
    }
  }
}

# Combine predictions from all models/electrodes
combined_preds <- do.call(rbind, all_preds)

# Convert to factors if necessary
combined_preds$parei <- factor(combined_preds$group)
combined_preds$DAT.z <- factor(combined_preds$facet)
print(names(combined_preds))
# Calculate the average prediction for each combination of bloc (x), parei, and DAT.z
averaged_preds <- aggregate(predicted ~ x + parei + DAT.z, data = combined_preds, FUN = mean)

# Calculate the average confidence intervals for each combination
averaged_preds$conf.low <- aggregate(conf.low ~ x + parei + DAT.z, data = combined_preds, FUN = mean)$conf.low
averaged_preds$conf.high <- aggregate(conf.high ~ x + parei + DAT.z, data = combined_preds, FUN = mean)$conf.high

# Now you can plot these averages using ggplot2
plot <- ggplot(averaged_preds, aes(x = x, y = predicted, color = parei)) +
  geom_line(size = 1.5) +
  geom_ribbon(aes(ymin = conf.low, ymax = conf.high), alpha = 0.1) +
  facet_wrap(~DAT.z, scales = 'free') + # Here we facet by DAT.z
  labs(title = "Interaction of Bloc, Pareidolia, and Creativity\nIn Predicting Lempel-Ziv Complexity",
       x = "Bloc", y = "Lempel-Ziv Complexity",
       color = "Averaged\nPareidolia") +
  theme_minimal() +
  theme(text = element_text(size = 30))

# Print the plot
print(plot)
