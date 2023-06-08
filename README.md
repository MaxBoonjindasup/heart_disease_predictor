# Heart Disease Predictor

## Introduction:
Cardiovascular diseases (CVDs) are the #1 cause of death globally (31% worldwide), totaling to 17.9 million lives each year. Heart failure is the most common event caused by CVDs. The dataset used for this study was obtained from 5 different hospitals with 11 data attributes.

## Overview:
* Performed exploratory data analysis on ~900 patient samples through NumPy, Pandas, and Matplotlib to identify patterns, handle missing and categorical data, and transform datasets.
* Trained 5 ML models (kNN, NN, RF, DT, and SVM) using cross-validation and scikit-learn to classify the presence and absence of heart disease and evaluated each model’s  and accuracy.

## Libraries and Resources Used
**Python Version:** Python 3.9.13  
**Packages:** pandas, numpy, sklearn, matplotlib, seaborn, plotly, and textwrap  
**Dataset:** heart.csv (included in repository; [Kaggle dataset](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction))

## Data Cleaning/Exploratory Data Analysis
To enhance the code's overall readability and applicability to our models later, I performed the following:
*	Counted the # of listings in each borough and compared the amounts in a bar plot and the percentages in a pie graph
*	Created Revenue column and graphed the average revenue for each borough
*	Investigated for the top 3 highest revenue neighbourhoods in Manhattan, Brooklyn, & Queens
*	Made a graph that shows the distribution of room types in Manhattan, Brooklyn, & Queens. Then, identified the top performing room type in the top 3 neighbourhoods of Manhattan, Brooklyn, & Queens.

![](https://github.com/Max-Boonjindasup/airbnb_analysis/blob/main/airbnb_highlights.png)

## Model Building

1. Removed irrelavant categorical attributes and the Revenue column as to avoid multicollinearity.
2. Replaced 0's and NaN values with attribute means.
3. Checked for possible correlations between variables via correlation matrix.
4. Transformed the categorical variables into dummy variables.
5. Standardized features
6. Splitted data into train and tests sets (test size = 30%).   

I trained 5 different models and evaluated them using an array of metrics: Mean Squared Error, Root Mean Squared Error, Mean Absolute Error, and R2. For logistic regression, precision, recall, F1-score, and accuracy were reported along with the accompanying confusion matrix

Models:
*	**Linear Regression** – Baseline model; iterated on different predictor variables (ie number_of_reviews, latitude, etc.)
*	**Polynomial Regression** – Having 6 attributes, I wanted a model to use all my attributes when predicting price
*	**Multinomial Logistic Regression** – I flipped the question and made a classification model instead where we predict borough from price.
*	**Decision Tree Regression** – Testing both questions (borough predicts price vs price predicts borough) with another model
*	**Random Forest Regression** – Again, testing both questions (borough predicts price vs price predicts borough) with another model

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
