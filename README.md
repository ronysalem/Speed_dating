# Speed dating

## Problem Formulation:

* Input: All Features in the data except id & match columns
* Output: Whether the 2 persons would match or not
* Data Mining Function: Classification and prediction
* Impact: knowing if the couple is going to match or not

This Python code aims to predict whether two persons would be a match based on various features.

## Requirements:
- Python 3
- Libraries: numpy, pandas, scikit-learn, xgboost, scikit-optimize

You can install the required libraries using pip:

```
pip install numpy pandas scikit-learn xgboost scikit-optimize
```

## Data:
The data consists of various features, and the goal is to predict whether the two individuals are a match or not.

## Data Preprocessing:
- Handling missing values.
- Encoding categorical features.
- Splitting the data into training and testing sets.

## Model Selection and Hyperparameter Tuning:
The code includes multiple machine learning models:
- RandomForestClassifier
- GradientBoostingClassifier
- XGBClassifier
- Support Vector Machine (SVM) Classifier

Hyperparameters for these models are tuned using GridSearchCV and RandomizedSearchCV, and the models are trained and evaluated.

## Result Export:
The code also provides functions to save the prediction results in CSV files for each model.

## How to Use:
1. Install the required libraries using the pip command mentioned above.
2. Prepare your dataset in the same format as the provided data.
3. Adjust the data file paths in the code to point to your dataset.
4. Run the code for each model you want to evaluate and save the results.

## Running the Code:
- Ensure that your dataset is in the same directory or update the file paths accordingly.
- Run the code for each model by uncommenting the relevant sections.

## Model Comparison:
The code provides a way to compare the performance of different models and their hyperparameter combinations to find the best model for your specific dataset.
