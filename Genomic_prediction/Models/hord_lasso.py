# import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

# data load
X = pd.read_csv('./bar_SNP_preprocessed.csv', delimiter=',', index_col=None)
y = pd.read_csv('./bar_metabolome_preprocessed.csv', delimiter=',', index_col=None)

X = X.iloc[:, 1:]  # Drop the first column if it is non-numerical (e.g., IDs)
y = y.iloc[:, 1:]  # Same as above
# y = (y - y.mean()) / y.std()    # Reverse this for evaluation, if needed

# reducing datasets
X_sampled = X.sample(frac=0.1, random_state=42)
y_sampled = y.loc[X_sampled.index]

# creating train + test datasets
print(X_sampled)
print(y_sampled)
print('Now dividing datasets')
X_train, X_test, y_train, y_test = train_test_split(X_sampled, y_sampled, test_size=0.2, random_state=42)

# LASSO model setup
print('Now defining LASSO model')
lasso_model = Lasso(alpha=0.01, max_iter=10000, random_state=42)
print('Now fitting train data into model')
lasso_model.fit(X_train, y_train)

# making predictions on the test data
print('Now predicting')
y_pred = lasso_model.predict(X_test)

# evaluating the model using Mean Squared Error and Pearson's Correlation Coefficient
print('Now evaluating')
mse = mean_squared_error(y_test, y_pred)
print('Mean Squared Error: ', mse)

correlation, _ = pearsonr(y_test.values.flatten(), y_pred.flatten())
print('Correlation Coefficient: ', correlation)

# visualization
plt.figure()
plt.scatter(y_test.values.flatten(), y_pred.flatten(), alpha=0.7, color='b', edgecolor='k', s=40)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.axline((0, 0), slope=1, color='r', linestyle='--', linewidth=2, label='Perfect Prediction (y=x)')
plt.legend()
plt.grid(alpha=0.4)
plt.show()

