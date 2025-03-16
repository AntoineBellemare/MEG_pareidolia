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
fixed_effect_name <- "gamma3.z:DAT.z"  # Replace with your desired fixed effect name
effect_direction <- "positive"  # Can be either "positive" or "negative"

# Define the directory for the models
folder <- "C:/Users/Antoine/github/MEG_pareidolia/R_data/RS_predict_bloc_fooofed" 

# Specify the region(s) of interest
regions_of_interest <- c("OL", 'OZ', 'OR')  # Example region

# Convert the regions of interest into a list of electrode indices
electrodes_of_interest <- unlist(MEG_atlas[regions_of_interest])

# Initialize a list to store the predictions for the significant models
all_preds <- list()

# Loop through each electrode of interest
for (elec in electrodes_of_interest) {
  # Construct the file name based on the electrode index
  file_name <- sprintf("model_electrode_%03d.rds", elec)
  full_path <- file.path(folder, file_name)
  print(full_path)
  # Check if the file exists
  if (file.exists(full_path)) {
    model <- readRDS(full_path)
    # Check if the specified fixed effect exists in the model
    if (fixed_effect_name %in% names(fixef(model))) {
      effect_value <- fixef(model)[fixed_effect_name]
      
      # Check if the effect is significant in the specified direction
      summary_model <- summary(model)
      p_value <- summary_model$coefficients[fixed_effect_name, "Pr(>|z|)"]
      
      if ((effect_direction == "positive" && effect_value > 0 && p_value < 0.01) ||
          (effect_direction == "negative" && effect_value < 0 && p_value < 0.01)) {
        
        # Get predictions for the model
        preds <- ggpredict(model, terms = c("theta.z[all]", 'DAT.z'), ci.lvl=0.01)
        
        # Add predictions to the list if not null
        if (!is.null(preds)) {
          all_preds[[length(all_preds) + 1]] <- preds
        }
      }
    }
  }
}

# Check if all_preds is populated and combine predictions
summary(model)
# Assuming all_preds is a list of data frames with columns: x, predicted, conf.low, conf.high, group
# Combine predictions from all models/electrodes
combined_preds <- do.call(rbind, all_preds)
print(names(combined_preds))
combined_preds
# Ensure 'group' column is treated as a factor for grouping
combined_preds$group <- as.factor(combined_preds$group)
combined_preds
# Calculate the average prediction for each combination of FD.z (x) and pos_nobj (group)
averaged_preds <- aggregate(predicted ~ x + group, data = combined_preds, FUN = mean)

# Calculate the average confidence intervals for each combination
averaged_preds$conf.low <- aggregate(conf.low ~ x + group, data = combined_preds, FUN = mean)$conf.low
averaged_preds$conf.high <- aggregate(conf.high ~ x + group, data = combined_preds, FUN = mean)$conf.high

# ... [The rest of your code remains unchanged] ...

# ... [The rest of your code remains unchanged] ...

# Plotting the averaged predictions with smoothed curves
plot <- ggplot(averaged_preds, aes(x = x, y = predicted, color = group)) +
  geom_smooth(method = "loess", span = 0.5, se = FALSE, size = 2.5, aes(color = group)) +  # Apply smoothing with the same color aesthetic
  #geom_ribbon(aes(ymin = conf.low, ymax = conf.high, fill = group), alpha = 0.1) +  # Confidence interval bands
  scale_fill_manual(values = c("red", "green", "blue")) +  # Manual colors for fill, if needed
  labs(title = "Interaction of gamma and creativity\nin central regions",
       x = "theta", y = "Pre / Post",
       color = "Pareidolia") +
  theme_minimal() + 
  theme(text = element_text(size = 40)) +
  guides(color = guide_legend(), fill = guide_legend())  # Merge the legends if necessary



# Print the plot with smoothed curves
print(plot)
