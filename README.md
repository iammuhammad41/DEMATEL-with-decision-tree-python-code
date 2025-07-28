# DEMATEL Solver & Enhanced Analysis

A Python framework for performing DEMATEL (Decision‐Making Trial and Evaluation Laboratory) analysis on expert‐provided influence matrices, followed by advanced feature selection and machine learning model evaluation.

## Features

1. **DEMATEL Analysis**

   * Aggregate multiple expert matrices
   * Normalize direct‐influence matrix
   * Compute total‐influence matrix
   * Compute factor **Prominence** (R + C) and **Relation** (R − C)
   * Classify factors into **Cause** (net > 0) vs **Effect** (net < 0)
   * Plot Influential Relation Map (IRM) scatter chart
   * Export all matrices and R/C values to Excel

2. **Consistency Metrics**

   * Inter‐expert variance analysis
   * Mean & standard deviation of influence variance

3. **Feature Selection**

   * Recursive Feature Elimination (RFE) with logistic regression
   * Random forest feature importance

4. **Model Building & Evaluation**

   * Train & cross‐validate multiple classifiers (Decision Tree, Logistic Regression, SVM, k‑NN, Naive Bayes, Random Forest, Gradient Boosting)
   * Handle class imbalance with SMOTE
   * Standard scaling + stratified K‑fold CV
   * Print accuracy & feature importances/coefs

5. **Model Interpretation**

   * SHAP‐based summary plots for tree‐ and linear‐based models

---

## Requirements

* Python 3.7+
* numpy
* pandas
* openpyxl
* matplotlib
* scikit‑learn
* imbalanced‑learn
* shap

Install dependencies with:

```bash
pip install numpy pandas openpyxl matplotlib scikit-learn imbalanced-learn shap
```

---

## Usage

1. **Prepare Inputs**

   * `inputs/factors.txt` — one factor name per line
   * `inputs/matrices/` — CSV files, each an N×N direct‐influence matrix per expert (filename → expert ID)

2. **Run**

   ```bash
   python dml_dt8.py
   ```

   This will:

   * Load factors & expert matrices
   * Compute DEMATEL steps (Z, X, T, R, C)
   * Identify Cause vs Effect factors
   * Plot IRM scatter plot
   * Save an Excel workbook (`outputs/DEMATELAnalysis.xlsx`) with all matrices & metrics
   * Perform feature selection & build ML models on aggregated influence data
   * Display SHAP explanations

3. **Inspect Outputs**

   * Excel file in `outputs/`
   * Console logs for feature selection, model CV scores, and SHAP plots

---

## File Overview

* **dml_dt8.py** — main implementation & entry point
* **inputs/**

  * `factors.txt`
  * `matrices/` (expert CSVs)
* **outputs/**

  * `DEMATELAnalysis.xlsx`

---

## Acknowledgments

* DEMATEL methodology inspired by classic decision‐science literature
* Machine learning components leverage [scikit‑learn](https://scikit-learn.org/) and [imbalanced‑learn](https://imbalanced-learn.org/)
* SHAP interpretation uses [SHAP](https://github.com/slundberg/shap)

