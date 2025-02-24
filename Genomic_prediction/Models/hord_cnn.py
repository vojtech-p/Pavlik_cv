import jax
import jax.numpy as jnp
import optax
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from flax import linen as nn
from flax.training import train_state
import numpy as np
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr


# Data loading and preprocessing
X = pd.read_csv('./bar_SNP_preprocessed.csv', delimiter=',', index_col=None)
y = pd.read_csv('./bar_metabolome_preprocessed.csv', delimiter=',', index_col=None)

X = X.iloc[:, 1:].values  # Convert to numpy array
y = y.iloc[:, 1:]  # Convert to numpy array
print(X)
print(len(X))
print(y)
print()

# Reducing datasets
# sample_indices = np.random.choice(X.shape[0], int(0.1 * X.shape[0]), replace=False)
# X_sampled = X[sample_indices]
# y_sampled = y[sample_indices]

# normalization
# y_normalized = StandardScaler().fit_transform(y)
y_normalized = MinMaxScaler().fit_transform(y)
y_normalized = pd.DataFrame(y_normalized, columns=y.columns, index=y.index).values

# Creating train + test data
X_train, X_test, y_train, y_test = train_test_split(X, y_normalized, test_size=0.2, random_state=42)


# Defining the fully connected model using Flax
class FCNN(nn.Module):
    @nn.compact
    def __call__(self, x):
        # Input layer
        x = nn.Dense(512)(x)
        x = nn.relu(x)

        # Hidden layers
        x = nn.Dense(256)(x)
        x = nn.relu(x)

        x = nn.Dense(128)(x)
        x = nn.relu(x)

        # Output layer for regression (multivariate)
        x = nn.Dense(y_train.shape[1])(x)
        return x


# Initialize the model, optimizer, and training state
model = FCNN()


# Setting up the optimizer and loss function
def create_train_state(model, learning_rate=1e-3):
    params = model.init(jax.random.PRNGKey(0), jnp.ones((1, X_train.shape[1])))  # Example input shape
    tx = optax.adam(learning_rate)
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)


train_state = create_train_state(model)


# Loss function (Mean Squared Error for regression)
def loss_fn(params, batch):
    inputs, targets = batch
    predictions = model.apply(params, inputs)  # Apply the model using the params
    return jnp.mean((predictions - targets) ** 2)


# Gradient function
@jax.jit
def update_model(state, batch):
    inputs, targets = batch
    grads = jax.grad(loss_fn)(state.params, (inputs, targets))  # Ensure batch is passed as tuple
    return state.apply_gradients(grads=grads)


# Training loop
num_epochs = 10
batch_size = 22
# enough_loss = 0.04

epoch_losses = []
for epoch in range(num_epochs):
    # Shuffle and create batches for training
    epoch_loss = 0
    idx = np.random.permutation(len(X_train))
    for i in range(0, len(X_train), batch_size):
        batch_idx = idx[i:i + batch_size]
        X_batch = jnp.array(X_train[batch_idx])
        y_batch = jnp.array(y_train[batch_idx])

        # Update model parameters
        train_state = update_model(train_state, (X_batch, y_batch))

        # Calculate loss for the batch and accumulate it
        batch_loss = loss_fn(train_state.params, (X_batch, y_batch))
        epoch_loss += batch_loss

    # Average loss for the epoch
    epoch_loss /= (len(X_train) // batch_size)
    epoch_losses.append(epoch_loss)

    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}")
#    if epoch_loss <= enough_loss:
#        break

# Make predictions using the trained model
y_pred = model.apply(train_state.params, jnp.array(X_test))

# Evaluating the model
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

correlation, _ = pearsonr(y_test.flatten(), y_pred.flatten())
print(f'Pearson Correlation Coefficient: {correlation}')

# Visualization
plt.figure()
plt.scatter(y_test.flatten(), y_pred.flatten(), alpha=0.7, color='b', edgecolor='k', s=40)
plt.xlabel('Actual values')
plt.ylabel('Predicted values')
plt.show()

# Visualization of the loss function over epochs
plt.figure()
plt.plot(range(1, num_epochs + 1), epoch_losses, marker='o', color='b', label='Loss')
plt.title('Training Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.grid(True)
plt.legend()
plt.show()
