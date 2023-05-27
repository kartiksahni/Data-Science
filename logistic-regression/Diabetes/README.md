# Exploratory Data Analysis and Logistic Regression on the Diabetes Dataset

This code performs exploratory data analysis (EDA), implements logistic regression, and compares the results with the scikit-learn implementation on the "diabetes" dataset.

## Code Description

The code performs the following tasks:

1. Imports necessary packages and libraries.
2. Loads the "diabetes" dataset from the `diabetes.csv` file.
3. Performs exploratory data analysis (EDA) on the dataset:
   - Checks for missing values.
   - Prints data types.
   - Describes the dataset.
   - Finds negative values in any column.
   - Plots histograms of each column.
   - Plots a correlation matrix.
   - Finds variables most correlated with the outcome.
4. Preprocesses the data for logistic regression:
   - Splits variables into features (X) and labels (y).
   - Converts the data into numpy arrays.
   - Standardizes the features.
   - Splits the data into training and testing sets.
5. Implements logistic regression using gradient descent:
   - Defines the sigmoid, hypothesis, cost function, gradient, and gradient descent functions.
   - Trains the logistic regression model on the training data.
   - Plots the cost function against the number of iterations.
6. Predicts the outcome on the test set using the trained model.
7. Calculates the accuracy of the logistic regression model.
8. Compares the results with the scikit-learn implementation:
   - Uses scikit-learn's LogisticRegression class.
   - Fits the model to the training data.
   - Predicts the outcome on the test set.
   - Calculates the accuracy of the scikit-learn model.

## Results

The accuracy of the logistic regression model and the scikit-learn model on the test set is printed.

## Dependencies

Make sure to have the following packages installed:

- pandas
- numpy
- matplotlib
