# import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

# data load
X = pd.read_csv('./bar_SNP_preprocessed.csv', delimiter=',', index_col=None)
y = pd.read_csv('./bar_metabolome_preprocessed.csv', delimiter=',', index_col=None)

X = X.iloc[:, 1:]
y = y.iloc[:, 1:]
# y = (y - y.mean()) / y.std()    # reverse for evaluation!

# reducing datasets
X_sampled = X.sample(frac=0.1, random_state=42)
y_sampled = y.loc[X_sampled.index]

# creating train + test data
print(X_sampled)
print(y_sampled)
print('Now dividing datasets')
X_train, X_test, y_train, y_test = train_test_split(X_sampled, y_sampled, test_size=0.2, random_state=42)

# random fores setup
print('Now defining random forest values')
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
print('Now fitting train data into model')
rf_model.fit(X_train, y_train)

# making predictions on the test data
print('Now predicting')
y_pred = rf_model.predict(X_test)

# evaluating the model using Mean Squared Error and pearson's CC
print('Now evaluating')
mse = mean_squared_error(y_test, y_pred)
print('Mean Squared Error: ', mse)

correlation, _ = pearsonr(y_test.values.flatten(), y_pred.flatten())
print('Correlation Coefficient: ', correlation)

# visualization
plt.figure()
plt.scatter(y_test.values.flatten(), y_pred.flatten(), alpha=0.7, color='b', edgecolor='k', s=40)
plt.xlabel('Actual values')
plt.ylabel('Predicted values')
plt.show()
