# Enhanced Power Grid Prediction using Machine Learning

## Project Overview
This project develops a machine learning-based model for predictive analysis and anomaly detection in power grids. It compares four advanced ML models (Decision Tree, Random Forest, XGBoost, and LightGBM) optimized with hyperparameter tuning to detect power grid anomalies and cyber attacks.

## Key Features
- Data preprocessing with Yeo-Johnson Power Transformation
- Feature selection using ANOVA F-test
- Dimensionality reduction with PCA
- Class imbalance handling using ADASYN
- SQL integration for data storage
- Comprehensive model evaluation

## Dataset
The dataset (`data1.csv`) is a binary classification subset from power system simulations by Mississippi State University and Oak Ridge National Laboratory. It contains:
- 128 columns (116 PMU measurements, 12 control/relay logs, 1 target label)
- Events labeled as "Natural" (0) or "Attack" (1)

## Setup Instructions

1. Clone this repository:
   ```bash
   git clone https://github.com/haxshita/Power_Grid_Efficiency_Prediction.git
   cd Power_Grid_Efficiency_Prediction
