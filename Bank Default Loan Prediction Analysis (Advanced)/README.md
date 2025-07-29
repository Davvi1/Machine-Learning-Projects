# 🏦 Home Credit Default Risk Analysis

## 📊 Objective  
This project aims to understand and predict defaults on bank loans, providing a risk evaluation service for retail banks. The analysis includes an inferential study to identify key factors contributing to default risk, followed by building machine learning models focused on maximizing recall and F1-score for predicting defaults.

## 📁 Dataset  
The dataset used is from the Home Credit Default Risk competition on Kaggle (https://www.kaggle.com/competitions/home-credit-default-risk/data), containing detailed consumer information, including loan default status. It is assumed to be representative of real-world data used by banks for credit risk assessment. The analysis is conducted in the notebook 'Default Risk Analysis.ipynb'.

 The 
## 🧪 Methodology  

### Exploratory Data Analysis & Inference  
Initial exploration involved understanding variable distributions and data quality. Statistical inference with logistic regression was used to identify the ten most significant predictors of loan default. Some model assumptions were violated, which were noted but not fully addressed due to scope constraints.

### Machine Learning Modeling  
To emphasize prediction performance, multiple machine learning models were developed and compared, including Logistic Regression, XGBoost, and LightGBM. The final deployed model was a voting classifier ensemble of these three, achieving a recall of 0.69 for the positive default class, with an F1 score of 0.28 (positive) and 0.82 (negative).

## ✅ Performance Metrics  
While key metrics such as recall and precision were moderate, this initial model serves as a minimum viable product demonstrating the potential to identify defaults. Limitations stem largely from computational constraints and omitted advanced tuning.

## 🧠 Key Insights  
- Recall for default prediction reached a reasonable level despite low precision, highlighting room for improvement.  
- Identified significant predictive features through inferential analysis.  
- Computational limitations prevented full exploitation of advanced methods and hyperparameter tuning.

## 🚀 Deployment  
The final model was containerized and deployed via a FastAPI application, and was hosted on Render, enabling real-time prediction requests through an accessible API endpoint.

## 🔧 Future Improvements  
Future iterations should consider:  
- Using KNN imputation and iterative methods for missing data, including non-numeric features.  
- Applying VIF-based feature elimination iteratively to reduce multicollinearity.  
- Conducting comprehensive hyperparameter tuning (Optuna, RandomizedSearch, etc.).  
- Replacing LinearSVC with SVC to leverage kernel methods.  
- Expanding ensembles with tuned hyperparameters and novel combinations.  
- Experimenting with complex neural network architectures.  
- Using SMOTE and SMOTEEN to better balance classes, including categorical data.  
- Revisiting feature selection criteria to retain high-cardinality variables when beneficial.  

## 💼 Business Context Enhancements  
- Tailoring models to maximize different metrics, e.g., focusing on predicting non-defaults where valuable.  
- Streamlining models post-training to only use strongest features for faster inference.  
- Integrating multi-table merging in pipelines for easier production deployment.  
- Researching industry-standard risk models and common data availability to tailor solutions.  
- Considering multi-output models reflecting diverse bank requirements.  

---

## Summary  
This project illustrates a proof-of-concept for bank default prediction, balancing inferential understanding with predictive modeling under resource constraints. The deployed model establishes a foundation for further refinement toward practical, robust credit risk solutions.
