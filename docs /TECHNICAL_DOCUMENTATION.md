# Technical Documentation
## Title
Wind Farm Renewable Energy Optimization Using Data Science

## Problem statement
Wind-power output varies with wind conditions and sensor measurements. This prototype explores historical data and compares machine-learning regressors to estimate power generation and identify influential input features.

## Objectives
1. Upload and inspect a wind-farm dataset.
2. Generate six exploratory visualizations.
3. Train and compare four regression models.
4. Report regression metrics and threshold-based classification metrics.
5. Provide downloadable predictions and feature-importance results.

## Methodology
The uploaded data is converted to numeric values where appropriate; rows missing selected model inputs or target are removed for model training. The dataset is split into training and testing sets. Models are trained on the training set and evaluated on held-out test data.

## Metrics
- MAE: mean absolute prediction error.
- MSE: mean squared prediction error.
- RMSE: square root of MSE.
- R²: proportion of target variance explained by the model, relative to a mean baseline.
- Accuracy and F1: calculated after converting continuous power to high/normal classes using the training-target median.

## Reproducibility
Install packages from `requirements.txt`, run `streamlit run src/app.py`, upload the same dataset, select the same features, and use the same random seed and test percentage.

## Limitations
The data and feature choices affect model quality. Correlations and feature importance are not causal proof. This prototype does not directly control turbines.
