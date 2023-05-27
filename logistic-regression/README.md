# Logistic Regression on MNIST Dataset

This is a Jupyter notebook that demonstrates the implementation of logistic regression on the MNIST dataset. It performs binary classification by considering only the digits 0 and 1.

## Prerequisites

Make sure you have the following packages installed:

- pandas
- numpy
- matplotlib
- scikit-learn


## Dataset

The MNIST dataset is loaded from the file `mnist_test.csv`. It contains grayscale images of handwritten digits along with their corresponding labels.

## Data Preprocessing

The following steps are performed for data preprocessing:

1. Check for null values in the dataset.
2. Filter the dataset to include only the digits 0 and 1.
3. Convert the data into numpy arrays.
4. Split the data into features (X) and labels (Y).
5. Standardize the features by subtracting the mean and dividing by the standard deviation.
6. Split the data into training and test sets.

## Logistic Regression

Logistic regression is implemented using gradient descent. The following functions are defined:

- `sigmoid`: Computes the sigmoid function.
- `hypothesis`: Computes the hypothesis function for logistic regression.
- `cost_function`: Computes the cost function for logistic regression.
- `gradient`: Computes the gradient of the cost function.
- `gradient_descent`: Performs gradient descent to optimize the parameters.

The model is trained using the training data, and the cost function is plotted against the number of iterations.

## Model Evaluation

The trained model is used to make predictions on both the training and test datasets. The following functions are defined:

- `predict`: Predicts the labels based on the trained model.
- `accuracy`: Computes the accuracy of the predictions.

The accuracy of the model is calculated for both the training and test datasets.

## Results

The training accuracy and test accuracy are printed to evaluate the performance of the logistic regression model.

