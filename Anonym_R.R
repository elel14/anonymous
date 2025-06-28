# Thesis: Random Forest Forecasting Model for Austrian Housing Construction
# Author: [Anonym]
# Description: Yearly Random Forest model with evaluation, visualization, and diagnostics

# ========================
# 1. Libraries and Settings
# ========================
install.packages("caret")
install.packages("lattice")
install.packages("ranger")
install.packages("readr")
install.packages("dplyr")
install.packages("ggplot2")
install.packages("randomForest")
install.packages("tidyr")
install.packages("scales")
install.packages("magrittr")
install.packages("ggcorrplot")

library(readr)
library(dplyr)
library(ggplot2)
library(randomForest)
library(tidyr)
library(scales)
library(magrittr)
library(ggcorrplot)
library(caret)
library(ranger)
library(lattice)

# For reproducibility
set.seed(42)

# ========================
# 2. Load and Explore Data
# ========================


setwd("C:/Users/blend/Desktop/seminar new dataset")

data <- read.csv("Housing_Construction_RF_Iterative_Imputed.csv")

# View structure
str(data)
summary(data)





# ========================
# 3. Basic Diagnostics
# ========================
# Histogram of Building Completions

ggplot(data, aes(x = `Building.Completions`)) +
  geom_histogram(bins = 20, fill = "skyblue", color = "black") +
  theme_minimal(base_size = 14) +
  labs(x = "Completions", y = "Frequency") +
  theme(
    axis.text = element_text(size = 12),
    axis.title = element_text(size = 14)
  )

ggsave("Rplot.png", width = 7, height = 4.5, dpi = 300)




# Boxplot for detecting outliers


ggplot(data, aes(y = `Building.Completions`)) +
  geom_boxplot(fill = "tomato", alpha = 0.6) +
  theme_minimal(base_size = 14) +
  labs(x = NULL, y = "Building Completions") +
  theme(axis.text = element_text(size = 12))

ggsave("Rplot01.png", width = 7, height = 4.5, dpi = 300)





# remove the 'Year' column and compute correlation matrix
cor_data <- data %>%
  select(-Year) %>%
  cor(use = "complete.obs")

# Clean column/row names for readability
colnames(cor_data) <- gsub("\\.", " ", colnames(cor_data))
rownames(cor_data) <- colnames(cor_data)

# Generate the correlation heatmap
ggcorrplot(cor_data,
           method = "square",                    
           type = "lower",                         
           lab = FALSE,                            
           tl.cex = 9,                             
           tl.srt = 45,                             
           colors = c("blue", "white", "red"),      
           ggtheme = theme_minimal())


# ========================
# 4. Define Features and Target
# ========================
target <- "Building.Completions"
features <- setdiff(names(data), c("Year", target))
X <- data[, features]
y <- data[[target]]




# ========================
# 5. Time-Aware Train-Test Split
# ========================


split_idx <- floor( 0.8 * nrow(data))
X_train <- X[1:split_idx, ]
y_train <- y[1:split_idx]
X_test <- X[(split_idx + 1):nrow(data), ]
y_test <- y[(split_idx + 1):nrow(data)]
years_test <- data$Year[(split_idx + 1):nrow(data)]




str(X_train)

# ========================
# 6. Train Random Forest Model
# ========================
rf_model <- randomForest(x = X_train, y = y_train, ntree = 500)
rf_model



# ========================
# 7. Predictions & Evaluation
# ========================
y_pred <- predict(rf_model, X_test)
y_pred



# Evaluation metrics
evaluate_model <- function(actual, predicted) {
  mae <- mean(abs(actual - predicted))
  rmse <- sqrt(mean((actual - predicted)^2))
  r2 <- 1 - sum((actual - predicted)^2) / sum((actual - mean(actual))^2)
  return(list(MAE = mae, RMSE = rmse, R2 = r2))
}

metrics <- evaluate_model(y_test, y_pred)
print(metrics)



# ========================
# 8. Actual vs Predicted Plot
# ========================
pred_df <- data.frame(Year = years_test, Actual = y_test, Predicted = y_pred)
pred_df


library(scales)
library(ggplot2)

ggplot(pred_df, aes(x = Year)) +
  geom_line(aes(y = Actual,    color = "Actual"),    linewidth = 1.3) +
  geom_line(aes(y = Predicted, color = "Predicted"), linewidth = 1.3, linetype = "dashed") +
  scale_color_manual(
    values = c(Actual = "#1f77b4", Predicted = "#ff7f0e")
  ) +
  scale_y_continuous(labels = comma) +
  scale_x_continuous(breaks = pretty(pred_df$Year, n = 10)) +
  labs(
    subtitle = "Comparison of true and forecasted completions over time",
    x        = "Year",
    y        = "Number of Completions"
  ) +
  theme_minimal(base_size = 14) +
  theme(
    legend.title     = element_blank(),     
    legend.position  = "top",
    plot.subtitle    = element_text(
      size   = 13,
      margin = margin(t = 0, r = 0, b = 10, l = 0)
    ),
    axis.title       = element_text(face = "bold"),
    panel.grid.minor = element_blank()
  )




# ========================
# 9. Feature Importance
# ========================
library(randomForest)
library(ggplot2)
library(dplyr)
importance(rf_model)


importance_df <- data.frame(Feature = rownames(importance(rf_model)),
                            Importance = importance(rf_model)[, 1]) %>%
  arrange(desc(Importance))

importance_df


ggplot(importance_df, aes(x = reorder(Feature, Importance), y = Importance)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  coord_flip() +
  theme_minimal() +
  labs( x = "Feature", y = "Importance")



# ========================
# 10. Save Outputs
# ========================
ggsave("actual_vs_predicted.png", width = 8, height = 5)
ggsave("feature_importance.png", width = 8, height = 5)



# ─────────────────────────────────────────────────────────────
# Hyperparameter tuning: max depth & min samples per leaf
# ─────────────────────────────────────────────────────────────
library(caret)
library(ranger)
install.packages("lattice")
install.packages("ranger")

set.seed(42)

# 1) features + target into one data.frame
train_df <- data.frame(X_train, Building.Completions = y_train)

# 2) 5-fold CV control
ctrl <- trainControl(
  method        = "cv",
  number        = 5,
  verboseIter   = TRUE,
  allowParallel = TRUE
)

# 3) Grid
tune_grid <- expand.grid(
  mtry          = floor(sqrt(ncol(X_train))),  # keep default mtry
  splitrule     = "variance",
  min.node.size = c(1, 5, 10)             
)

# 4) Run grid search
rf_tuned <- train(
  Building.Completions ~ .,
  data      = train_df,
  method    = "ranger",
  trControl = ctrl,
  tuneGrid  = tune_grid,
  num.trees = 500,
  importance = "permutation"
)

# 5) best settings
print(rf_tuned)
best <- rf_tuned$bestTune
cat("Best max.depth:", best$max.depth,
    "– Best min.node.size:", best$min.node.size, "\n")

library(randomForest)
rf_model_best <- randomForest(x = data[data$Year < 2017, -3], y = data$Building.Completions[data$Year < 2017], ntree = 1000, mtry = 4, splitrule = "variance", min.node.size = 10)
plot(data$Year, data$Building.Completion, col = "orange", type = "b",xlim = c(2017,2023))
points(c(2017:2023) , predict(rf_model_best,newdata = data[data$Year >= 2017, -3]),col = "black",type = "b")
rf_model <- randomForest(x = X_train, y = y_train, ntree = 500)
points(c(2017:2023) , predict(rf_model,newdata = data[data$Year >= 2017, -3]),col = "green",type = "b")


# 6) Re-evaluate on the test set
final_model  <- rf_tuned$finalModel
y_pred_tuned <- predict(final_model, data = X_test)$predictions
metrics_tuned <- evaluate_model(y_test, y_pred_tuned)
print(metrics_tuned)


# visualize tuning results
plot(rf_tuned)                              
ggsave("rf_tuning_results.png", width=6, height=4, dpi=300)

