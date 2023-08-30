# Heart Disease Predictor

## Introduction:
Cardiovascular diseases (CVDs) are the #1 cause of death globally (31% worldwide), totaling to 17.9 million lives each year. Heart failure is the most common event caused by CVDs. The dataset used for this study was obtained from 5 different hospitals with 11 data attributes.

## Overview:
* Performed exploratory data analysis on ~900 patient samples through NumPy, Pandas, and Matplotlib to identify patterns, handle missing and categorical data, and transform datasets.
* Trained 5 ML models (kNN, NN, XGBoost, DT, and SVM) using cross-validation, GridSearchCV, scikit-learn, and tensorflow to classify the presence and absence of heart disease and evaluated each model’s loss and accuracy.

## Libraries and Resources Used
**Python Version:** Python 3.9.13  
**Packages:** pandas, numpy, sklearn, matplotlib, seaborn, tensorflow, and keras  
**Dataset:** heart.csv (included in repository; from [Kaggle](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction))

## Data Cleaning/Exploratory Data Analysis
I made the following changes to improve the readability and applicability of the data:
* Reviewed for missing values (and found none!)
* Viewed distribution and possible correlations of features via histograms and pairplots (splitted values by sex), respectively. Discovered a very strong bimodal distribution for Cholesterol and FastingBS.

![](https://github.com/Max-Boonjindasup/heart_disease_predictor/blob/main/heart_pairplot.png)

## Model Building

1. Transformed the categorical variables (Sex, ChestPainType, RestingECG, ExerciseAngina, & ST_Slope) into dummy variables.
2. Standardized features
3. Splitted data into train and tests sets (test size = 20%).   

I trained 5 different models for classification of heart disease and evaluated them by loss and accuracy.

Models:
*	**KNN** – Solved first for best value of k (8), then fitted model, and graphed confusion matrix for accuracy, precision, recall, and F1-score. Also compared performance using cross validation.
*	**Neural Networks** – 5 layers starting with ReLU and ending with sigmoid activation functions, having dropout regularizations in between them (total 102 nodes); compiled with SparseCategoricalCrossentropy loss function and optimized with Adam at learning rate 0.001
*	**XGBoost** – Tested different hyperparameter combinations using GridSearchCV and used optimized hyperparameter values for model.
*	**Decision Tree** – Performed hyperparameter tuning via GridSearchCV before training model
*	**SVM** – Manually tested for best hyperparameters and trained model on optimized hyperparameters. Then, compared model to GridSearchCV optimized hyperparameters.

![](https://github.com/Max-Boonjindasup/heart_disease_predictor/blob/main/heart_knn_error_rate.png)
![](https://github.com/Max-Boonjindasup/heart_disease_predictor/blob/main/heart_confusion_matrix_knn.png)

## Model performance
Models were trained on a classification task of presence or absence of heart disease. The neural networks model far outperformed the other models based on its lower loss and perfect accuracy.

|       Model              |     Loss     | Accuracy |
|-------------------------|--------------|----------|
| Neural Networks         | 2.42e-06     |  1.000   |
| Support Vector Machine  | 4.14e-05     |  1.000   |
| XGBoost                 | 6.00e-01     |  1.000   |
| Decision Tree           |      NA      |  1.000   |
| K-Nearest Neighbors     |      NA      |  0.899   |
