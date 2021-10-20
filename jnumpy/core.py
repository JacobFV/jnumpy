import csv
import math
import functools
import logging

from typing import Callable, Tuple, List, Optional, Union
import numpy as np
from jnumpy.graph import TensorGraph


class Tensor:
    """Tensor
    
    The `Tensor` is the base class for all constants, variables, and operations.
    """
    
    def __init__(self, val: Optional[np.array] = None, name: Optional[str] = 'Tensor'):

        self.name = Tensor._make_name_unique(name)
        graph = TensorGraph.default()
        if graph.exec_mode == 'eager':
            assert val is not None, "Eager mode requires a value"
            self.val = self.val
        elif graph.exec_mode == 'lazy':
            self.val = None
        else:
            raise ValueError('Invalid execution mode')

        graph.add_node(self)  # this may change the name of the tensor
    
    # modified from personal code: https://github.com/JacobFV/nnn/blob/main/nnn/nodes/node.py 
    _UNIQUE_NAME_COUNTER = {}
    @classmethod
    def _make_name_unique(cls, name):
        """Sequentially suffixes names. Non-idempotent method to ensure no name collisions.
        Example:
            >>> node = Tensor()
            >>> node._make_name_unique('Tensor')
            'Tensor1'
            >>> node._make_name_unique('Tensor')
            'Tensor2'
            >>> node._make_name_unique('Tensor')
            'Tensor3'
            >>> node._make_name_unique('Tensor1')
            'Tensor11'
        Args:
            name: Tensor instance name to make unique.

        Returns: unique name
        """
        if name in Tensor._UNIQUE_NAME_COUNTER:
            Tensor._UNIQUE_NAME_COUNTER[name] += 1
        else:
            Tensor._UNIQUE_NAME_COUNTER[name] = 1
        return name + str(Tensor._UNIQUE_NAME_COUNTER[name])

    def __add__(self, other):
        return eval("Add")(self, other)
    
    def __neg__(self):
        return eval('Neg')(self)
    
    def __sub__(self, other):
        return eval("Sub")(self, other)
    
    def __mul__(self, other):
        return eval("Mul")(self, other)
    
    def __pow__(self, other):
        return eval("Pow")(self, other)
    
    def __matmul__(self, other):
        return eval("MatMul")(self, other)


class Var(Tensor):
    """Variable Tensor.
    
    Represents a constant or variable numerical value."""
    def __init__(self, val: Optional[np.array] = None, trainable: bool = True, name: Optional[str] = 'Var'):
        
        self.trainable = trainable
        super().__init__(val=val, name=name)


@jnumpy.export("jnumpy.Op")
class Op(Tensor):
    """Operation-backed Tensor.
    
    Represents an operation"""
    
    def __init__(self, *input_tensors: Tensor, name: Optional[str] = 'Op'):
        """Make sure to set any variables you might need in `foreward` 
        before initializing when the graph is in eager execution mode
        """
        
        self.input_tensors = input_tensors

        graph = TensorGraph.default()
        if graph.exec_mode == 'eager':
            val = self.foreward(tuple(i.val for i in input_tensors))[0]
        elif graph.exec_mode == 'lazy':
            val = None
        else:
            raise ValueError('Invalid execution mode')
        
        super().__init__(val=val, name=name)

        for input_tensor in self.input_tensors:
            graph.add_edge(input_tensor, self)
        
    def foreward(self, input_values: Tuple[np.array]) -> np.array:
        raise NotImplementedError('subclasses should implement this method')
    
    def reverse_grad(self, input_values: Tuple[np.array], output_value: np.array, top_grad: np.array) -> Tuple[np.array]:
        raise NotImplementedError('subclasses should implement this method')


"""~~This should illustrate the imperitive execution paradigm I currently implement:~~

```python
N = 10
W_T = Var(val=np.ones(shape=(N, 1)), trainable=True)  # [N, 1]

def regressor(X_T, W_T):
    return X_T @ W_T

def compute_loss(X, ytrue, lambda_val, W_T):
    X_T = Var(val=np.array(X), trainable=False)  # [B, N]
    ytrue_T = Var(val=np.array(ytrue), trainable=False)  # [B, 1]
    lambda_T = Var(val=np.array(lambda_val), trainable=False)  # [N, 1]

    ypred_T = regressor(X_T, W_T)  # [B, 1]    
    err_T = (ypred_T - ytrue_T) ** 2  # [B, 1]
    loss_T = ReduceSum(ReduceSum(err_T, axis=1), axis=0)             + lambda_T * ReduceSum(ReduceSum(W_T, axis=1), axis=0)

    return loss_T

# train
opt = SGD(lr=0.1)
batch_size = min(32, train_arr.shape[0])
num_batches = train_arr.shape[0] // batch_size
for epoch in range(1000):

    #opt.lr = 0.001 + 0.009 * math.sin(epoch * 2 * math.pi / 1000) # 
    opt.lr = 1 / (epoch+100)

    if epoch % 100 == 0:
        #print(f'epoch %04d\tlr=%0.4f\ttrain loss=%0.4f\ttest loss=%0.4f' %
        #      (epoch, opt.lr, 
        #       compute_loss(train_arr[:,:-1], train_arr[:,-1:], lambda_val, W_T).val,
        #       compute_loss(test_arr[:,:-1], test_arr[:,-1:], lambda_val, W_T).val))
        #opt.debug=True
        pass
    else:
        opt.debug=False

    for batch_num in range(num_batches):

        loss_T = compute_loss(
            X=train_arr[batch_size*batch_num:batch_size*(batch_num+1), :-1],
            ytrue=train_arr[batch_size*batch_num:batch_size*(batch_num+1), -1:], 
            lambda_val=lambda_val,
            W_T=W_T)

        if opt.debug:
            print('weights:', W_T.val[:, 0])

        opt.minimize(loss_T)
```
"""
