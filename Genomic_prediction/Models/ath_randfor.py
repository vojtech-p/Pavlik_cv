# import libraries
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

# data load
X = pd.read_csv('./ath_SNP_all_preprocessed.csv', delimiter=',', index_col=None)
y = pd.read_csv('./ath_metabolome_all_preprocessed.csv', delimiter=',', index_col=None)

# drop redundant columns
X = X.iloc[:, 2:]
y = y.iloc[:, 1:y.shape[1]-1]
# y = (y - y.mean()) / y.std()    # reverse for evaluation!

# normalization
# y_normalized = StandardScaler().fit_transform(y)
y_normalized = MinMaxScaler().fit_transform(y)
y_normalized = pd.DataFrame(y_normalized, columns=y.columns, index=y.index)

# creating train + test data
# print(X_sampled)
# print(y_sampled)
print('Now dividing datasets')
X_train, X_test, y_train, y_test = train_test_split(X, y_normalized, test_size=0.2, random_state=42)
# print('x train', X_train)
# print(X_train.describe())

# print('y train', y_train)
# print(len(y_train))

# random forest setup
print('Now defining random forest values')
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
print('Now fitting train data into model')
rf_model.fit(X_train, y_train)

# making predictions on the test data
print('Now predicting')
y_pred = rf_model.predict(X_test)

# print('x test', X_test)
# print(X_test.describe())

# print('y pred', y_pred)
# print(len(y_pred))

# evaluating the model using Mean Squared Error and pearsons CC
print('Now evaluating')
mse = mean_squared_error(y_test, y_pred)
print('Mean Squared Error: ', mse)

# print('y test flatten:', y_test.flatten())
# print(len(y_test.flatten()))

# print('y pred flatten:', y_pred.flatten())
# print(len(y_pred.flatten()))

correlation, _ = pearsonr(y_test.values.flatten(), y_pred.flatten())
print('Correlation Coefficient: ', correlation)

# visualization
plt.figure()
plt.scatter(y_test.values.flatten(), y_pred.flatten(), alpha=0.7, color='b', edgecolor='k', s=40)
plt.xlabel('Actual values')
plt.ylabel('Predicted values')
plt.axline((0, 0), slope=1, color='r', linestyle='--', linewidth=2, label='Perfect Prediction (y=x)')
plt.legend()
plt.grid(alpha=0.4)
plt.show()
