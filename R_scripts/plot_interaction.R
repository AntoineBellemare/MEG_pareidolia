library(lme4)
library(lmerTest)
library(ggplot2)
library(ggeffects)

# Specify the fixed effect and whether you want the maximum positive or negative value
fixed_effect_name <- "your_fixed_effect_name"  # Replace with your desired fixed effect name
effect_direction <- "positive"  # Can be either "positive" or "negative"

# Define the directory for the models
folder <- "path_to_your_folder_with_models/" 

# Initialize variables to store the targeted effect
targeted_effect_value <- ifelse(effect_direction == "positive", -Inf, Inf)
targeted_effect_model <- NULL

# Load models and find the targeted effect
for (file_name in list.files(folder, full.names = TRUE)) {
  model <- readRDS(file_name)
  
  # Check if the specified fixed effect exists in the model
  if (fixed_effect_name %in% names(fixef(model))) {
    # Extract the value of the specified fixed effect
    effect_value <- fixef(model)[fixed_effect_name]
    
    # Check for the targeted effect based on the specified direction
    if ((effect_direction == "positive" && effect_value > targeted_effect_value) ||
        (effect_direction == "negative" && effect_value < targeted_effect_value)) {
      targeted_effect_value <- effect_value
      targeted_effect_model <- model
    }
  }
}

# Check if a model with the targeted effect was found
if (is.null(targeted_effect_model)) {
  stop("No targeted effects found in the models.")
}

# Plot the interaction effect of the identified fixed effect
# Replace "your_second_variable_here" with the name of the second variable for interaction
plot <- plot_model(targeted_effect_model, type = "pred", terms = c(fixed_effect_name, "your_second_variable_here"), ci.lvl = 0.95)
plot <- ggpar(plot, 
              title = "Interaction Effect Plot",
              xlab = fixed_effect_name, 
              ylab = "Response Variable",
              legend.title = "Second Variable")

# Customize the plot appearance as needed
plot <- plot + theme_minimal() + theme(text = element_text(size = 12))
print(plot)
