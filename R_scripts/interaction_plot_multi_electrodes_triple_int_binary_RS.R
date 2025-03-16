# Load required libraries
library(lme4)
library(lmerTest)
library(ggplot2)
library(ggeffects)
library(sjPlot)
library(ggpubr)
library(dplyr)
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
fixed_effect_name <- "gamma1.z:parei:DAT.z"  # Replace with your desired fixed effect name
effect_direction <- "positive"  # Can be either "positive" or "negative"

# Define the directory for the models
folder <- "C:/Users/Antoine/github/MEG_pareidolia/R_data/RS_predict_bloc_fooofed_triple/" 

# Specify the region(s) of interest
regions_of_interest <- c('OL', 'OR', 'OZ')  # Example region

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
      p_value <- summary_model$coefficients[fixed_effect_name, "Pr(>|z|)"]
      
      if ((effect_direction == "positive" && effect_value > 0 && p_value < 0.001) ||
          (effect_direction == "negative" && effect_value < 0 && p_value < 0.001)) {
        print('hello')
        # Get predictions for the triple interaction
        preds <- ggpredict(model, terms = c("DAT.z[all]", "gamma1.z[all]", "parei[all]"), ci.lvl=0.01)
        
        # Add predictions to the list if not null
        if (!is.null(preds)) {
          all_preds[[length(all_preds) + 1]] <- preds
        }
      }
    }
  }
}
# Assuming combined_preds is already defined and all_preds is a list of data frames
combined_preds <- do.call(rbind, all_preds)
combined_preds$LZ.z <- factor(combined_preds$group)
combined_preds$parei <- factor(combined_preds$facet)
combined_preds$DAT.z <- factor(combined_preds$x) # assuming x corresponds to LZ.z
combined_preds$LZ.z <- as.numeric(as.character(combined_preds$LZ.z))
combined_preds$parei <- as.numeric(as.character(combined_preds$parei))
combined_preds$DAT.z <- as.numeric(as.character(combined_preds$DAT.z))

# Calculate quartiles for parei, DAT.z, and LZ.z
parei_quartiles <- quantile(combined_preds$parei, probs = c(0.2, 0.5, 0.90), na.rm = TRUE)
datz_quartiles <- quantile(combined_preds$DAT.z, probs = c(0.1, 0.5, 0.90), na.rm = TRUE)
lz_quartiles <- quantile(combined_preds$LZ.z, probs = c(0.1, 0.5, 0.90), na.rm = TRUE)

# Manually set breaks if the quantiles are not unique
parei_breaks <- c(-Inf, parei_quartiles[1], median(combined_preds$parei), parei_quartiles[length(parei_quartiles)], Inf)
datz_breaks <- c(-Inf, datz_quartiles[1], median(combined_preds$DAT.z), datz_quartiles[length(datz_quartiles)], Inf)
lz_breaks <- c(-Inf, lz_quartiles[1], median(combined_preds$LZ.z), lz_quartiles[length(lz_quartiles)], Inf)

# Make sure breaks are unique after manual adjustment
parei_breaks <- unique(parei_breaks)
datz_breaks <- unique(datz_breaks)
lz_breaks <- unique(lz_breaks)

# Adjust labels if necessary after ensuring unique breaks
parei_labels <- if(length(parei_breaks) - 1 == 3) c("Low", "Medium", "High") else c("Low", "Medium", "Medium", "High")
datz_labels <- if(length(datz_breaks) - 1 == 3) c("Low", "Medium", "High") else c("Low", "Medium", "Medium", "High")
lz_labels <- if(length(lz_breaks) - 1 == 3) c("Low", "Medium", "High") else c("Low", "Medium", "Medium", "High")

# Use cut to create categorical variables
combined_preds$parei_cat <- cut(combined_preds$parei, breaks = parei_breaks, labels = parei_labels, include.lowest = TRUE)
combined_preds$DAT.z_cat <- cut(combined_preds$DAT.z, breaks = datz_breaks, labels = datz_labels, include.lowest = TRUE)
combined_preds$LZ.z_cat <- cut(combined_preds$LZ.z, breaks = lz_breaks, labels = lz_labels, include.lowest = TRUE)

# Identify predictions close to 1 (>= 0.9) and close to 0 (<= 0.1)
combined_preds$predicted_group <- ifelse(combined_preds$predicted >= 0.8, "Close to 1", 
                                         ifelse(combined_preds$predicted <= 0.8, "Close to 0", NA))

# Remove intermediate values
combined_preds <- combined_preds[!is.na(combined_preds$predicted_group), ]

# Calculate the average LZ.z for each category
average_lz <- combined_preds %>%
  group_by(parei_cat, DAT.z_cat, predicted_group) %>%
  summarise(average_LZ = mean(LZ.z, na.rm = TRUE)) %>%
  ungroup()

# Plot the interaction
# Plot the interaction with custom labels and larger fonts
# Define custom colors
custom_colors <- c("Low" = "turquoise", "Medium" = "orange", "High" = "purple")

# Plot the interaction with custom line thickness and colors
# Define custom colors
custom_colors <- c("Low" = "turquoise", "Medium" = "orange", "High" = "purple")

# Plot the interaction with custom line thickness and colors
ggplot(average_lz, aes(x = predicted_group, y = average_LZ, group = parei_cat, color = parei_cat)) +
  geom_line(size = 2) +  # Set line thickness to 2
  geom_point() +
  facet_wrap(~DAT.z_cat) +
  labs(title = "",
       x = "",
       y = "gamma 1",
       color = "Parei Category") +
  theme_minimal() +
  theme(legend.position = "right",  # Move legend to the right side
        text = element_text(size = 24),  # Set the font size to 24
        axis.title = element_text(size = 24),  # Set axis title font size
        axis.text = element_text(size = 22),  # Set axis text font size
        strip.text = element_text(size = 24),  # Set facet label font size
        axis.text.x = element_text(size = 22),  # Set x-axis text font size
        legend.title = element_text(size = 24),  # Set legend title font size
        legend.text = element_text(size = 22),
        legend.direction = "vertical",  # Display legend vertically
        legend.key.height = unit(3, "lines")) +  # Adjust legend key height
  scale_x_discrete(labels = c("Close to 0" = "Pre", "Close to 1" = "Post")) +  # Custom x-axis labels
  scale_color_manual(values = custom_colors) +  # Use custom colors
  labs(color = "Pareidolia")  # Rename the legend


#____________________________________________________________________

# Combine predictions from all models/electrodes
combined_preds <- do.call(rbind, all_preds)
combined_preds$LZ.z <- factor(combined_preds$group)
combined_preds$parei <- factor(combined_preds$facet)
combined_preds$DAT.z <- factor(combined_preds$x) # assuming x corresponds to LZ.z

# Convert factors to numeric if they are indeed numeric
combined_preds$LZ.z <- as.numeric(as.character(combined_preds$LZ.z))
combined_preds$parei <- as.numeric(as.character(combined_preds$parei))
combined_preds$DAT.z <- as.numeric(as.character(combined_preds$DAT.z))
#combined_preds$binary_outcome <- ifelse(combined_preds$predicted > 0.5, 1, 0)

# Calculate quartiles for parei, DAT.z, and LZ.z
parei_quartiles <- quantile(combined_preds$parei, probs = c(0.1, 0.5, 0.9), na.rm = TRUE)
datz_quartiles <- quantile(combined_preds$DAT.z, probs = c(0.1, 0.5, 0.9), na.rm = TRUE)
lz_quartiles <- quantile(combined_preds$LZ.z, probs = c(0.1, 0.5, 0.9), na.rm = TRUE)

# Manually set breaks if the quantiles are not unique
parei_breaks <- c(-Inf, parei_quartiles[1], median(combined_preds$parei), parei_quartiles[length(parei_quartiles)], Inf)
datz_breaks <- c(-Inf, datz_quartiles[1], median(combined_preds$DAT.z), datz_quartiles[length(datz_quartiles)], Inf)
lz_breaks <- c(-Inf, lz_quartiles[1], median(combined_preds$LZ.z), lz_quartiles[length(lz_quartiles)], Inf)

# Make sure breaks are unique after manual adjustment
parei_breaks <- unique(parei_breaks)
datz_breaks <- unique(datz_breaks)
lz_breaks <- unique(lz_breaks)

# Adjust labels if necessary after ensuring unique breaks
parei_labels <- if(length(parei_breaks) - 1 == 3) c("Low", "Medium", "High") else c("Low", "Medium", "Medium", "High")
datz_labels <- if(length(datz_breaks) - 1 == 3) c("Low", "Medium", "High") else c("Low", "Medium", "Medium", "High")
lz_labels <- if(length(lz_breaks) - 1 == 3) c("Low", "Medium", "High") else c("Low", "Medium", "Medium", "High")

# Use cut to create categorical variables
combined_preds$parei_cat <- cut(combined_preds$parei, breaks = parei_breaks, labels = parei_labels, include.lowest = TRUE)
combined_preds$DAT.z_cat <- cut(combined_preds$DAT.z, breaks = datz_breaks, labels = datz_labels, include.lowest = TRUE)
combined_preds$LZ.z_cat <- cut(combined_preds$LZ.z, breaks = lz_breaks, labels = lz_labels, include.lowest = TRUE)
                                

aggregated_data <- combined_preds %>%
  group_by(parei_cat, DAT.z_cat, LZ.z_cat) %>%
  summarise(proportion_of_1 = mean(binary_outcome)) %>%
  ungroup()  # Remove the grouping

# Plot the interaction
ggplot(aggregated_data, aes(x = proportion_of_1 , y = LZ.z_cat, group = parei_cat, color = parei_cat)) +
  geom_line(size = 1, position = position_dodge(width = 0.2)) +
  facet_wrap(~DAT.z_cat) +
  labs(title = "Triple Interaction Effect on Resting State Condition",
       x = "LZ.z Category", y = "Proportion of Binary Outcome = 1",
       color = "Parei Category") +
  theme_minimal() +
  theme(legend.position = "bottom")


print(plot)
#________________________________________________________________________________________________________________

# Assuming combined_preds is already defined and all_preds is a list of data frames
combined_preds <- do.call(rbind, all_preds)
combined_preds$LZ.z <- factor(combined_preds$group)
combined_preds$parei <- factor(combined_preds$facet)
combined_preds$DAT.z <- factor(combined_preds$x) # assuming x corresponds to LZ.z
combined_preds$LZ.z <- as.numeric(as.character(combined_preds$LZ.z))
combined_preds$parei <- as.numeric(as.character(combined_preds$parei))
combined_preds$DAT.z <- as.numeric(as.character(combined_preds$DAT.z))

# Identify predictions close to 1 (>= 0.9) and close to 0 (<= 0.1)
combined_preds$predicted_group <- cut(combined_preds$predicted,
                                      breaks = c(-Inf, 0.1, 0.9, Inf),
                                      labels = c("Close to 0", "Intermediate", "Close to 1"),
                                      include.lowest = TRUE)

# Filter out the intermediate group for a clearer plot
binary_groups <- combined_preds %>%
  filter(predicted_group %in% c("Close to 0", "Close to 1"))

ggplot(binary_groups, aes(x = predicted_group, y = LZ.z, color = as.factor(parei))) +
  geom_jitter(width = 0.2, height = 0, size = 2, alpha = 0.6) + # Use jitter to reduce overplotting
  facet_wrap(~as.factor(DAT.z)) + # Make sure to convert DAT.z to a factor if it's not already
  labs(
    title = "Triple Interaction Effect on Resting State Condition",
    x = "Predicted Outcome Category",
    y = "LZ.z Score",
    color = "Parei Score"
  ) +
  theme_minimal() +
  theme(legend.position = "bottom")

#___________________________________________________________________________________________________________________

# Convert the predicted values to binary outcomes
combined_preds$binary_outcome <- ifelse(combined_preds$predicted > 0.5, 1, 0)


aggregated_data <- combined_preds %>%
  group_by(LZ.z, parei, DAT.z) %>%
  summarise(proportion_of_1 = sum(binary_outcome) / n()) %>%
  ungroup()  # Remove the grouping

# Plot with lines
ggplot(aggregated_data, aes(x = proportion_of_1, y = LZ.z, color = as.factor(parei), group = interaction(parei, DAT.z))) +
  geom_point() + # Use points to represent the proportion of 1s at each LZ.z level
  geom_line() + # Optionally add lines to connect the points
  facet_wrap(~as.factor(DAT.z)) + # Facet by DAT.z
  labs(
    title = "Triple Interaction Effect on Resting State Condition",
    x = "Proportion of Binary Outcome = 1",
    y = "LZ.z Value",
    color = "Parei Level"
  ) +
  theme_minimal() +
  theme(
    axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1),
    strip.text.x = element_text(size = 8) # Ensure facet labels are small enough to fit
  )




# Convert factors to numeric after checking that the levels are numeric strings
combined_preds$LZ.z <- as.numeric(levels(combined_preds$LZ.z))[combined_preds$LZ.z]
combined_preds$parei <- as.numeric(levels(combined_preds$parei))[combined_preds$parei]
combined_preds$DAT.z <- as.numeric(levels(combined_preds$DAT.z))[combined_preds$DAT.z]

# Calculate quantiles instead of min, median, and max
lz_quantiles <- quantile(combined_preds$LZ.z, probs = c(0.1, 0.5, 0.9), na.rm = TRUE)
parei_quantiles <- quantile(combined_preds$parei, probs = c(0.1, 0.5, 0.9), na.rm = TRUE)
datz_quantiles <- quantile(combined_preds$DAT.z, probs = c(0.1, 0.5, 0.9), na.rm = TRUE)

some_tolerance=0.01
# Select data points that are close to the quantiles
combined_preds$close_to_lz_quantile <- with(combined_preds, 
                                            abs(LZ.z - lz_quantiles[1]) < some_tolerance | 
                                              abs(LZ.z - lz_quantiles[2]) < some_tolerance | 
                                              abs(LZ.z - lz_quantiles[3]) < some_tolerance
)
combined_preds$close_to_parei_quantile <- with(combined_preds, 
                                               abs(parei - parei_quantiles[1]) < some_tolerance | 
                                                 abs(parei - parei_quantiles[2]) < some_tolerance | 
                                                 abs(parei - parei_quantiles[3]) < some_tolerance
)
combined_preds$close_to_datz_quantile <- with(combined_preds, 
                                              abs(DAT.z - datz_quantiles[1]) < some_tolerance | 
                                                abs(DAT.z - datz_quantiles[2]) < some_tolerance | 
                                                abs(DAT.z - datz_quantiles[3]) < some_tolerance
)

# Subset the data to rows that are close to the quantiles
subset_data <- combined_preds[
  combined_preds$close_to_lz_quantile & 
    combined_preds$close_to_parei_quantile & 
    combined_preds$close_to_datz_quantile, 
]
# Threshold the predicted values at 0.5 to get binary outcomes
# Assuming that 'predicted' is a probability between 0 and 1


# Aggregate the data by LZ.z, parei, and DAT.z
# Since we're dealing with binary outcomes, we use sum instead of mean

       
aggregated_data <- combined_preds %>%
  group_by(LZ.z, parei, DAT.z) %>%
  summarise(mean_predicted = mean(predicted, na.rm = TRUE))

# Create a simplified plot
ggplot(aggregated_data, aes(x = LZ.z, y = mean_predicted, color = parei)) +
  geom_point() +
  facet_wrap(~DAT.z) +
  labs(title = "Simplified Triple Interaction Effect on Resting State Condition",
       x = "LZ.z Value", y = "Mean Predicted Outcome",
       color = "Parei Level") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1))
print(plot)
