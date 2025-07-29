# 🏥 Insurance Purchase Prediction with Machine Learning
This project analyzes the factors that influence whether an individual purchases insurance using logistic regression and various machine learning classification models. The dataset includes demographic and behavioral features, and the goal is to build and evaluate predictive models that classify individuals as insurance purchasers or not.

# 📊 Objective
Build and evaluate a logistic regression inferential model to learn about coefficients

Build and evaluate a range of machine learning models to predict insurance purchase.

Tune the models with cross-validated hyperparameter search.

Interpret model performance using AUC, precision, recall, F1-score, and visualizations.

Analyze feature importance and model coefficients to understand key drivers.

# 📁 Dataset
The dataset includes 2000 samples and contains both continuous and binary variables.
Target variable: insurance (1 = purchased, 0 = not purchased)

Features include:
- Age (continuous)
- Annual Income (continuous)
- Number of Family Members (continuous)
- Chronic Diseases (binary)
- Frequent Flyer (binary)
- Graduate or not (binary)
- Employment Type (binary)
- Ever Travelled Abroad (binary)


# 🧪 Methodology
A Pipeline was built to combine preprocessing and modeling. I used MinMaxScaling to base the variables, RandomizedSearch CV to tune regularization types and hyperparameters, and Stratified K-Fold for cross-validation (with AUC as scoring metric).

Models run:
- Logistic regression
- Decision tree
- Naive Bayes
- Support vector machine
- Random forest
- Voting classifier ensemble model.


# ✅ Performance Metrics
The best model at hand was arguably the random forest

- AUC = 0.771.
- Accuracy = 0.802.
- The case of **no** insurance being bought - recall of 0.92 and precision of = 0.80
- The case of insurance **being** bought - recall of 0.58 and precision of 0.81.
- Macro average F1 score = 0.77.


# 📈 Visualizations
Confusion matrix to visualize true/false positives/negatives.

ROC curve and AUC to evaluate discrimination.

Feature importance table or coefficient summary


# 🔧 Improvements for Future Analysis
Experiment with resampling techniques (SMOTE, undersampling) to improve recall for the minority class.

Improve recall by optimizing classification threshold rather than default 0.5.

Collect more balanced or larger data to improve generalization.

Run different ML models.