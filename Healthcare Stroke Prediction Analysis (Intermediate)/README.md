🧠 Stroke Prediction with Machine Learning
This project explores the factors influencing stroke occurrence using inferential statistics and several machine learning classification models. It combines statistical rigor with practical deployment to create a predictive tool that identifies individuals at high risk of stroke. Given the medical implications, particular focus was placed on maximizing recall to ensure high sensitivity in detecting potential stroke cases.

📊 Objective
Build an inferential logistic regression model to understand the impact of various health and demographic features on stroke risk.

Train a suite of machine learning classifiers to predict stroke occurrence.

Tune models using randomized cross-validation with F1 as the main scoring metric.

Interpret model performance using precision, recall, F1-score, AUC, confusion matrices, and learning curves.

Analyze feature importance and coefficients to interpret key risk factors.

Deploy the best model using Flask and joblib to demonstrate local-serving predictions via API.

📁 Dataset
The dataset includes over 5,000 samples, each representing an individual’s health and demographic profile.

Target Variable:

stroke: Binary outcome (1 = stroke occurred, 0 = no stroke)

Key Features:

Age (continuous)

Hypertension (binary)

Heart disease (binary)

Marital status (binary)

Average glucose level (continuous)

BMI (continuous)

Smoking status (categorical)

Work type (categorical)

Residence type (binary)

Gender (binary)

🧪 Methodology
A modular machine learning pipeline was built using scikit-learn, with preprocessing and model training bundled in a single flow.

Preprocessing: One-hot encoding of categorical variables, MinMax scaling for numerical ones.

Class Imbalance Handling: Class weights were used for most models; SMOTE was cautiously applied in one case.

Cross-validation: Stratified K-Fold to maintain class balance during training.

Model tuning: RandomizedSearchCV for hyperparameter optimization.

Models tested:

Logistic Regression (with class weights)

Decision Tree

Naive Bayes

Support Vector Classifier

Random Forest

XGBoost

LightGBM

Voting Classifier

✅ Performance Metrics
The model chosen for deployment was logistic regression, based on interpretability and reliable performance. While some ensemble models performed better on AUC, logistic regression offered the best balance of explainability and usability for deployment.

🧠 Key Insights
Age was by far the most significant predictor, both in statistical and machine learning models.

Average glucose level also showed a significant positive association with stroke.

Logistic regression assumptions were partially violated, but given the focus on ML prediction, these were not fully addressed.

Feature importance analysis across tree-based models aligned with inferential findings, reinforcing model validity.

🚀 Deployment
The logistic regression model was saved with joblib and deployed using a Flask API.

A simple /predict endpoint accepts JSON input and returns a stroke prediction and probability.

This served as a proof of concept; in a production setting, containerization (e.g., Docker) and secure hosting would be required.

🔧 Improvements for Future Work
More data. The extreme imbalance in stroke cases limited model reliability and generalization. A larger, more diverse dataset—especially with more positive cases—would be invaluable.

More models. Future work could explore models like BalancedRandomForest, EasyEnsemble, or neural networks, particularly those designed for imbalanced classification tasks.

Advanced imbalance handling. While class weights and SMOTE were used, newer resampling or hybrid approaches could improve results without sacrificing generalization.

Model interpretability. Tools like SHAP or LIME could further illuminate model behavior and increase transparency—especially important in medical contexts.

Deployment readiness. For real-world applications, additional deployment tooling (e.g., CI/CD, Docker, cloud services) and compliance with data privacy regulations would be essential.