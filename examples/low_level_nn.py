import numpy as np

import jnumpy as jnp


def mystery_function(x):
    return x @ np.random.randn(100, 10)


Xtrain = np.random.randn(1000, 100)
Ytrain = mystery_function(Xtrain)
Xtest = np.random.randn(1000, 100)
Ytest = mystery_function(Xtest)

layer_sizes = [Xtrain.shape[1], 50, 45, Ytrain.shape[1]]
epochs = 100
batch_size = 32

layer_weights = []
layer_biases = []
for i, (d_prev, d_curr) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
    layer_weights.append(jnp.Var(np.random.randn(d_prev, d_curr), name=f"W{i}"))
    layer_biases.append(jnp.Var(np.random.randn(d_curr, 1), name=f"B{i}"))


def NN(X_T):
    """Build a simple neural network."""
    H_T = X_T
    for W_T, B_T in zip(layer_weights, layer_biases):
        Z_T = H_T @ W_T
        Z_T = Z_T + B_T
        return jnp.Sigm(Z_T)


def loss_fn(ypred_T, ytrue_T):
    """Compute the mean squared error."""
    err_T = (ypred_T - ytrue_T) ** 2  # [B, C]
    loss_T = jnp.ReduceSum(jnp.ReduceSum(err_T, axis=1), axis=0)  # []


# training pipeline
opt = jnp.SGD(lr=1.0)

train_batch_size = min(32, Xtrain.shape[0])
num_train_batches = Xtrain.shape[0] // train_batch_size
test_batch_size = min(32, Xtest.shape[0])
num_test_batches = Xtest.shape[0] // test_batch_size

# training loop
for epoch in range(epochs):

    opt.lr = 0.9 ** epoch  # cool learning rate

    # minibatch training
    train_loss = 0.0
    for i in range(num_train_batches):

        # get minibatch
        Xbatch = Xtrain[i * train_batch_size : (i + 1) * train_batch_size]
        ybatch = Ytrain[i * train_batch_size : (i + 1) * train_batch_size]
        X_T = jnp.Var(val=Xbatch)  # [B, N]
        ytrue_T = jnp.Var(val=ybatch)  # [B, C]

        # compute loss
        ypred_T = NN(X_T)  # [B, C]
        loss_T = loss_fn(ypred_T, ytrue_T)  # []

        opt.minimize(loss_T)  # update weights
        train_loss += loss_T.val  # accumulate loss

    # print training stats
    print(f"Epoch {epoch+1}/{epochs} train loss: {train_loss/num_train_batches}")
