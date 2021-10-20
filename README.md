# Jumpy

Jacob's numpy library extension.

## Getting started

1. Clone this repository to your local project directory.
2. Import the `jumpy` module.

## Example

```python
from jumpy import *

def NN(X_T):
    """Build a simple neural network."""
    H_T = X_T
    for W_T, B_T in zip(layer_weights, layer_biases):
        Z_T = H_T @ W_T
        Z_T = Z_T + B_T
        return Sigm(Z_T)

def loss_fn(ypred_T, ytrue_T):
    """Compute the mean squared error."""
    err_T = (ypred_T - ytrue_T)**2  # [B, C]
    loss_T = ReduceSum(ReduceSum(err_T, axis=1), axis=0)  # [] 

# training pipeline
opt = SGD(lr=1.0)

train_batch_size = min(32, Xtrain.shape[0])
num_train_batches = Xtrain.shape[0] // train_batch_size
test_batch_size = min(32, Xtest.shape[0])
num_test_batches = Xtest.shape[0] // test_batch_size

# training loop
for epoch in range(rounds):

    opt.lr = 0.9 ** epoch # cool learning rate

    # minibatch training
    train_loss = 0.0
    for i in range(num_train_batches):

        # get minibatch
        Xbatch = Xtrain[i*train_batch_size:(i+1)*train_batch_size]
        ybatch = Ytrain[i*train_batch_size:(i+1)*train_batch_size]
        X_T = Var(val=Xbatch, trainable=False)  # [B, N]
        ytrue_T = Var(val=ybatch, trainable=False)  # [B, C]

        # compute loss
        ypred_T = NN(X_T)  # [B, C]
        loss_T = loss_fn(ypred_T, ytrue_T)  # []

        opt.minimize(loss_T) # update weights
        train_loss += loss_T.val # accumulate loss
    
    # print training stats
    print(f'Epoch {epoch+1}/{rounds} train loss: {train_loss/num_train_batches}')
```

