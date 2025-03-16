library(lme4)
library(lmerTest)
library(ggplot2)
library(ggeffects)
library(sjPlot)
library(ggpubr)
install.packages("xfun")

# Specify the fixed effect and whether you want the maximum positive or negative value
fixed_effect_name <- "gamma1.z:DAT.z"  # Replace with your desired fixed effect name
effect_direction <- "positive"  # Can be either "positive" or "negative"

# Define the directory for the models
folder <- "C:/Users/Antoine/github/MEG_pareidolia/R_data/RS/predict_bloc/" 

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

# Plot the interaction effect of the identified fixed effect
# Replace "your_second_variable_here" with the name of the second variable for interaction
plot <- plot_model(targeted_effect_model, type = "pred", terms = c("gamma1.z[all]", "DAT.z"), ci.lvl = 0.01)
plot
# Generate predictions using ggpredict
preds <- ggpredict(targeted_effect_model, terms = c( "gamma1.z[all]", "DAT.z"), ci.lvl=0.01)

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

preds <- data.frame(
  creativity = factor(rep(c("low", "mid", "high"), each = 6)),
  condition = rep(c("Pre", "Post"), each = 3),  # Assuming 'gamma1.z' corresponds to 'Pre' and 'Post'
  predicted = c(0.99, 0.96, 0.93, 0.88, 0.78, 0.01,
                0.85, 0.73, 0.68, 0.62, 0.55, 0.08,
                0.21, 0.25, 0.26, 0.28, 0.30, 0.47)
)

# Now, create the bar plot
plot <- ggplot(preds, aes(x = creativity, y = predicted, fill = condition)) +
  geom_bar(stat = "identity", position = position_dodge(width = 0.7)) +
  labs(title = "Predicted Gamma Power Pre vs Post Task by Creativity Level",
       x = "Creativity Level", y = "Predicted Gamma Power",
       fill = "Condition") +
  scale_fill_manual(values = c("Pre" = "blue", "Post" = "red")) +  # Customize colors if needed
  theme_minimal() +
  theme(text = element_text(size = 20))

# Print the plot
print(plot)

# Print the plot
print(plot)
print(preds)
