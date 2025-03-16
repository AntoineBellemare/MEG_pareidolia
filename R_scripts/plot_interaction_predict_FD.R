library(lme4)
library(lmerTest)
library(ggplot2)
library(ggeffects)
library(sjPlot)
library(ggpubr)
install.packages("xfun")

# Specify the fixed effect and whether you want the maximum positive or negative value
fixed_effect_name <- "pos_nobj:FD.z"  # Replace with your desired fixed effect name
effect_direction <- "negative"  # Can be either "positive" or "negative"

# Define the directory for the models
folder <- "C:/Users/Antoine/github/MEG_pareidolia/R_data/results_20231127-222616/" 


########################################################################################################################

# Initialize variables to store the targeted effect
targeted_effect_value <- ifelse(effect_direction == "positive", -Inf, Inf)
targeted_effect_model <- NULL

# Initialize variables to store the targeted effect
if (effect_direction == "positive") {
  targeted_effect_value <- -Inf
} else if (effect_direction == "negative") {
  targeted_effect_value <- Inf
} else {
  stop("Invalid effect direction. Use 'positive' or 'negative'.")
}

targeted_effect_model <- NULL
targeted_effect_file <- ""

# List only .rds files matching the pattern "model_electrode_*.rds"
pattern <- "^model_electrode_\\d+\\.rds$"
model_files <- list.files(folder, pattern = pattern, full.names = TRUE)

# Load models and find the targeted effect
for (file_name in model_files) {
  model <- readRDS(file_name)
  
  # Check if the specified fixed effect exists in the model
  if (fixed_effect_name %in% names(fixef(model))) {
    # Extract the value of the specified fixed effect
    effect_value <- fixef(model)[fixed_effect_name]
    
    # Check for the targeted effect based on the specified direction
    is_targeted_effect <- (effect_direction == "positive" && effect_value > targeted_effect_value) ||
      (effect_direction == "negative" && effect_value < targeted_effect_value)
    
    if (is_targeted_effect) {
      targeted_effect_value <- effect_value
      targeted_effect_model <- model
      targeted_effect_file <- file_name
    }
  }
}
summary(targeted_effect_model)
summary(model)

########################################################################################################################

# Specify the electrode of interest
electrode_of_interest <- "63"  # Replace with your electrode identifier


# Initialize variable to store the model for the specified electrode
selected_model <- NULL
selected_model_file <- ""

# Pattern to list only .rds files
pattern <- "^model_electrode_\\d+\\.rds$"
model_files <- list.files(folder, pattern = pattern, full.names = TRUE)

# Load models and find the one corresponding to the specified electrode
for (file_name in model_files) {
  if (grepl(electrode_of_interest, file_name)) {
    targeted_effect_model <- readRDS(file_name)
    selected_model_file <- file_name
    break
  }
}

summary(targeted_effect_model)
########################################################################################################################

# Plot the interaction effect of the identified fixed effect
# Replace "your_second_variable_here" with the name of the second variable for interaction
plot <- plot_model(targeted_effect_model, type = "pred", terms = c("bloc[all]", "parei", "DAT.z"), ci.lvl = 0.01)
plot
enhanced_plot <- plot +
  labs(title = "Predicting Brain Signal Complexity\nfrom Resting State, Pareidolia and Creativity",  # Add your custom title
       x = "Resting State",  # Custom X-axis label
       y = "Lempel-Ziv Complexity",  # Custom Y-axis label
       color = "Pareidolia") +  # Custom legend title
  theme_minimal(base_size = 30)   # Use theme_minimal with base font size
  
  #theme(
  # plot.title = element_text(size = 16, face = "bold"),  # Customize title
  #  axis.title.x = element_text(size = 14),  # Customize X-axis title
  #  axis.title.y = element_text(size = 14),  # Customize Y-axis title
  #  legend.title = element_text(size = 14),  # Customize legend title
  #  legend.text = element_text(size = 12)  # Customize legend text
  #)

enhanced_plot
#####################################################################################################################
# Generate predictions using ggpredict
preds <- ggpredict(targeted_effect_model, terms = c("FD.z[all]", "pos_nobj"), ci.lvl=0.01)

# Define the thresholds for 'low', 'mid', 'high' and categorize DAT.z
# Note: Adjust the thresholds as per your specific data
preds$group <- cut(as.numeric(as.character(preds$group)), 
                   breaks = c(-Inf, -0.5, 0.5, Inf), 
                   labels = c("low", "mid", "high"))


print(summary(as.numeric(as.character(preds$group))))

# Assuming these are the min and max values from the legend
min_value <- -0.915
max_value <- 2.43

# Calculate equal-sized intervals between min and max
interval <- (max_value - min_value) / 5

# Generate breaks
breaks <- seq(min_value, max_value, by = interval)

# Ensure that the sequence covers the entire range by including the max_value
breaks <- c(breaks, max_value)
labels <- c("1", "2", "3", "4", "5")

# Apply cut function to create a factor with levels
preds$group <- cut(as.numeric(as.character(preds$group)), breaks = breaks, labels = labels, include.lowest = TRUE)
print(breaks)
# Check the levels again
print(levels(preds$group))

# Now, create the plot
plot <- ggplot(preds, aes(x = x, y = predicted, color = group)) +
  geom_line(size = 1.5) +
  geom_ribbon(aes(ymin = conf.low, ymax = conf.high), alpha = 0.02) +
  labs(title = "Interaction of Stimulus FD and Pareidolia\nIn Predicting Detrended Fluctuation Analysis",
       x = "Stimulus FD", y = "DFA",
       color = "Number of Objects") +
  theme_minimal() + 
  theme(text = element_text(size = 30))

# Print the plot
print(plot)


