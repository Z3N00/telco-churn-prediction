# Telco Customer Churn Prediction

## Overview
Machine learning model to predict customer churn for a telecom company using Python. 
Helps businesses identify at-risk customers before they leave.

## Problem Statement
27% of customers churn. The goal is to predict which customers are likely to cancel 
their subscription so the business can take proactive retention actions.

## Models Used
- Logistic Regression — 78.61%
- Random Forest — 79.25% (best performer)
- XGBoost — 76.40%

## Key Findings
- Contract type is the strongest predictor of churn
- Month-to-month customers churn significantly more than long term contract customers
- New customers (tenure < 6 months) are highest risk
- Customers without Online Security and Tech Support churn more

## Tech Stack
- Python
- Pandas
- Scikit-learn
- XGBoost
- Matplotlib, Seaborn

## Key Learnings
- Label encoding for categorical variables
- Handling imbalanced datasets
- Precision-recall tradeoff
- Threshold adjustment to improve recall
- Decision Tree visualization
- Feature importance analysis
- Hyperparameter tuning with GridSearchCV
