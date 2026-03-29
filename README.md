# Credit Risk Default Prediction

**Author:** Talia Low | **Date:** March 2026

This repository contains an **end-to-end** data science portfolio project for predicting loan defaults using the Lending Club dataset (2007–2018). The objective is to demonstrate a structured, industry-aligned approach to credit-risk modelling, including strict out-of-time validation, business-aligned threshold selection, and model explainability.



## Project Overview

- Dataset: [LendingClub loan data (Kaggle)](https://www.kaggle.com/datasets/wordsforthewise/lending-club) 
- Problem: Binary classification (Default vs Non-Default)  
- Goal: Build a robust model to rank borrower risk at origination  

The project is organised as a sequential workflow across four Jupyter notebooks, covering the full modelling pipeline from data audit to final evaluation:

| File | Description |
| :--- | :--- |
| `01_data_audit_and_target_definition.ipynb` | Data loading, target definition (Fully Paid vs Charged Off), and removal of post-origination leakage features |
| `02_eda_on_training_set.ipynb` | Temporal train/validation/holdout split, multicollinearity analysis (VIF), missing value assessment, and exploratory data analysis conducted strictly on the training set |
| `03_feature_engineering_and_preprocessing.ipynb` | Domain-driven feature engineering, missing indicator creation, and construction of a reusable scikit-learn preprocessing pipeline |
| `04_modeling_and_evaluation.ipynb` | Model benchmarking, Optuna hyperparameter tuning, validation-based model selection, business-aligned thresholding, PSI drift analysis, and SHAP/LIME explainability |
| `feature_engineering.py` | Modular Python implementation of feature engineering logic, designed for reuse in production-style pipelines |