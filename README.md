# Heart Disease Predictor

## Introduction:
Cardiovascular diseases (CVDs) are the #1 cause of death globally (31% worldwide), totaling to 17.9 million lives each year. Heart failure is the most common event caused by CVDs. The dataset used for this study was obtained from 5 different hospitals with 11 data attributes.

## Overview:
* Performed exploratory data analysis on ~900 patient samples through NumPy, Pandas, and Matplotlib to identify patterns, handle missing and categorical data, and transform datasets.
* Trained 5 ML models (kNN, NN, RF, DT, and SVM) using cross-validation and scikit-learn to classify the presence and absence of heart disease and evaluated each model’s  and accuracy.

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
*	**KNN** – Solved first for best value of k (8), then fitted model, and graphed confusion matrix for accuracy, precision, recall, and F1-score.
*	**Neural Networks** – 5 layers starting with ReLU and ending with sigmoid activation functions, having dropout regularizations in between them (total 102 nodes); compiled with SparseCategoricalCrossentropy loss function and optimized with Adam at learning rate 0.001
*	**XGBoost** – Tested different hyperparameter combinations using GridSearchCV and used optimized hyperparameter values for model.
*	**Decision Tree** – 
*	**SVM** – 


## Model performance
Most models were trained to predict pricing based on a variety of attributes, but changing to predicting borough (Queens, Brooklyn, Staten Island, Manhattan, and Bronx) based off price produced excellent scores.
The Logistic Regression model far outperformed the other approaches on the test sets.
*	**Linear**: RMSE = 0.825
*	**Polynomial**: RMSE = 0.773
*	**Logistic**: RMSE = 0.224 (ACC = 0.986)
*	**Decision Tree**: RMSE = 0.822
*	**Random Forest**: RMSE = 0.822

![](https://github.com/Max-Boonjindasup/airbnb_analysis/blob/main/confusion_matrix_airbnb_attributes.png)
![](https://github.com/Max-Boonjindasup/airbnb_analysis/blob/main/logistic_reg_score.png)

## Bonus Section
I performed PCA for the purpose of feature extraction by identifying the most influential features. Below are the principal components and the top 3 features (loadings) for PC1, the principal component that explains the most variance in the data. I also graphed the biplot to recast the original data onto the new PCA axes and included the top 3 PC feature vectors for reference (see notebook for 3D visualization in Plotly).

* neighbourhood_group_Manhattan (Borough - Manhattan): 0.504190
* room_type_Entire home/apt (Room Type - Entire home/apt): 0.371368
* latitude (Latitude coordinate): 0.307925

![](https://github.com/Max-Boonjindasup/airbnb_analysis/blob/main/airbnb_pca.png)
