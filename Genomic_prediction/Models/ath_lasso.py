# import libraries
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

# data load
X = pd.read_csv('./ath_SNP_16_preprocessed.csv', delimiter=',', index_col=None)
y = pd.read_csv('./ath_metabolome_16_preprocessed.csv', delimiter=',', index_col=None)

# remove unnecessary columns
X = X.iloc[:, 2:]
y = y.iloc[:, 1:y.shape[1]-1]

# normalization of the metabolomic dataset
#y_normalized = StandardScaler().fit_transform(y)
y_normalized = MinMaxScaler().fit_transform(y)
y_normalized = pd.DataFrame(y_normalized, columns=y.columns, index=y.index)

# creating train + test data
print('Now dividing datasets')
X_train, X_test, y_train, y_test = train_test_split(X, y_normalized, test_size=0.2, random_state=45)

# Lasso regression setup
print('Now defining Lasso regression values')
lasso_model = Lasso(alpha=0.001, max_iter=10000, random_state=42)  # Adjust alpha for regularization strength
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
plt.scatter(y_test.values.flatten(), y_pred.flatten(), alpha=0.7, color='b', edgecolor='k', s=5)
plt.xlabel('Actual values')
plt.ylabel('Predicted values')
plt.axline((0, 0), slope=1, color='r', linestyle='--', linewidth=2, label='Perfect Prediction (y=x)')
plt.legend()
plt.grid(alpha=0.4)
plt.show()

plt.figure()
plt.scatter(y_test.values.flatten(), y_pred.flatten(), alpha=0.7, color='b', edgecolor='k', s=5)  # Smaller points
plt.xlabel('Actual values')
plt.ylabel('Predicted values')

# Set axis limits to 0.6
plt.xlim(0, 0.6)
plt.ylim(0, 0.6)

plt.axline((0, 0), slope=1, color='r', linestyle='--', linewidth=2, label='Perfect Prediction (y=x)')
plt.legend()
plt.grid(alpha=0.4)
plt.show()
