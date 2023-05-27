
## Workflow

1. Data Import

The script imports the stock price data from the CSV file using the `pandas` library.

2. Data Exploration

It displays the first few rows of the data and provides information about the shape of the dataset.

3. Data Preprocessing

- Conversion of Date to Numerical Value: The script converts the date column to a numerical format.
- Encoding of Stock Symbol: The stock symbol column is encoded using `LabelEncoder` from scikit-learn.
- Standardization of Data: The numerical columns (open, close, low, high, volume) are standardized using `StandardScaler` from scikit-learn.

4. Data Splitting

The data is split into training and testing sets using the `train_test_split` function from scikit-learn. The default test size is set to 20%.

5. Model Training

The script implements linear regression using gradient descent. It defines the hypothesis function, error function, gradient function, and gradient descent function.

6. Model Evaluation

- Training Phase: The script trains the model on the training set and calculates the optimal theta (coefficients) using gradient descent.
- Prediction Phase: It predicts the stock prices for both the training and testing sets.
- R2 Score Calculation: The script calculates the R2 score, which measures the model's performance.

## Results

After running the script, you will see the following outputs:

- The optimal theta values obtained during the training phase.
- The R2 score for the training set.
- The R2 score for the testing set.

## License

This project is licensed under the MIT License. Feel free to modify and use the code as per your needs.

