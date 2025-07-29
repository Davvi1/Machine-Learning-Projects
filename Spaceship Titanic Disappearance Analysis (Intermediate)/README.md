# 🛸 Spaceship Titanic: Predicting Passenger Transport to an Alternate Dimension
## 📊 Objective
The goal of this project was to predict which passengers aboard the Spaceship Titanic were transported to an alternate dimension during the ship's collision with a spacetime anomaly. This challenge, part of the Kaggle competition, required building a machine learning model to classify passengers based on various features.

## 📁 Dataset
The dataset provided by Kaggle included passenger information with the following key columns:

PassengerId: Unique identifier for each passenger.

HomePlanet: The planet where the passenger originated.

CryoSleep: Whether the passenger was in cryosleep during the incident.

Cabin: Cabin number where the passenger stayed.

Destination: The destination planet for the passenger.

Age: Age of the passenger.

VIP: Whether the passenger was a VIP.

RoomService, FoodCourt, ShoppingMall, Spa, VRDeck: Amount spent in each service.

Name: Name of the passenger.

Transported: Target variable indicating whether the passenger was transported to an alternate dimension (1) or not (0).

## 🧪 Methodology
### Data Preprocessing
Handling Missing Values: Imputed missing values for continuous variables using median imputation and for categorical variables using the mode.

Feature Engineering: Created new features such as total spending across services and extracted titles from passenger names.

Encoding Categorical Variables: Applied one-hot encoding to categorical variables like HomePlanet, CryoSleep, Destination, and VIP.

Scaling: Standardized continuous variables to ensure uniformity across features.

### Model Selection
A variety of machine learning models were trained and evaluated:

Logistic Regression: A baseline linear model.

Decision Tree Classifier: A non-linear model capturing interactions between features.

Naive Bayes: A probabilistic model assuming independence between features.

Support Vector Classifier (SVC): A model effective in high-dimensional spaces.

Random Forest Classifier: An ensemble method using multiple decision trees.

XGBoost: A gradient boosting framework optimized for speed and performance.

LightGBM: A gradient boosting framework that uses tree-based learning algorithms.

Voting Classifier: An ensemble model combining the predictions of the above models.

### Model Evaluation
Models were evaluated using 5-fold Stratified Cross-Validation with F1-score as the primary metric. Hyperparameters were tuned using Optuna for models like XGBoost and LightGBM.

## ✅ Performance Metrics
The best-performing model achieved an F1-score of 0.80313, surpassing the competition's minimum requirement of 0.79.

## 🧠 Key Insights
Feature Importance: Age and spending across services were significant predictors of whether a passenger was transported.

Model Performance: Most models performed similarly, suggesting that the challenge's design emphasized data preprocessing and feature engineering over model complexity.

Class Imbalance: The dataset exhibited a class imbalance, with fewer passengers being transported, which posed challenges in model training.

## 🚀 Deployment
The final model was deployed using a Flask API, allowing for real-time predictions. The model was saved using joblib and served endpoints for prediction requests.

## 🔧 Improvements for Future Work
Advanced Models: Exploring neural networks and other advanced models could capture more complex patterns in the data.

Enhanced Feature Engineering: Further exploration of feature interactions and domain-specific features could improve model performance.

Data Augmentation: Techniques like SMOTE could be considered to address class imbalance, though caution is needed to avoid overfitting.

Model Interpretability: Tools like SHAP or LIME could be used to interpret model predictions and understand feature contributions.

## 📄 Submission Format
The final predictions were saved in a CSV file with the following format:

csv
Copy
Edit
PassengerId,Transported
1,True
2,False
3,True
...
This file was submitted to the Kaggle competition for evaluation.

## 📌 Conclusion
This project demonstrated the importance of data preprocessing and feature engineering in machine learning. Despite the simplicity of the models used, careful attention to data quality and feature selection led to a competitive performance in the Kaggle competition. Future work will focus on exploring more complex models and further enhancing the feature set to improve prediction accuracy.