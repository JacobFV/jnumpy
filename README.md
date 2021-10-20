# Jnumpy

[![PyPI version](https://badge.fury.io/py/jnumpy.svg)](https://badge.fury.io/py/jnumpy)
[![](https://img.shields.io/badge/license-MIT-blue)](https://github.com/JacobFV/jnumpy/blob/main/LICENSE)


Jacob's numpy library for machine learning

## Getting started

1. Install from `pip` or clone locally:

```bash
$ pip install jnumpy
```

2. Import the `jnumpy` module.

```python
import jnumpy as jnp
```

## Example

```python
from jnumpy import *

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

## Limitations and Future Work

Version 2.0 is under development (see the dev branch) and will feature:
- static execution graphs
- a keras-style neural network API with `fit`, `evaluate`, and `predict` and premade layers
- richer collections of optimizers, metrics, and losses
- more examples

Also maybe for the future:
- custom backends (i.e.: tensorflow or pytorch instead of numpy)

## License

All code in this repository is licensed under the MIT license. No restrictions, but no warranties. See the [LICENSE](https://github.com/JacobFV/jnumpy/blob/main/LICENSE) file for details.

## Contributing

This is a small project, and I don't plan on growing it much. You are welcome to fork and contribute or email me `jacob` [dot] `valdez` [at] `limboid` [dot] `ai` if you would like to take over.