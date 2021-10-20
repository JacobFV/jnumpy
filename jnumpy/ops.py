from typing import List, Optional, Tuple

import numpy as np

from jnumpy.core import Tensor, Op


class Transpose(Op):
    
    def __init__(self, t: Tensor, axes: Optional[tuple] = None):
        
        self.foreward_kwargs = dict()
        self.reverse_kwargs = dict()
        
        if axes is not None:
            self.foreward_kwargs['axes'] = axes
            self.reverse_kwargs['axes'] = tuple(reversed(axes))
            
        super().__init__(t)
    
    def foreward(self, input_values: Tuple[np.array]) -> np.array:
        X = input_values[0]
        
        Y = X.transpose(**self.foreward_kwargs)
        
        return Y
    
    def reverse_grad(self, input_values: Tuple[np.array], output_value: np.array, top_grad: np.array) -> Tuple[np.array]:
        dY = top_grad
        
        dX = dY.transpose(**self.reverse_kwargs)
        
        return (dX,)


class ReduceSum(Op):
    
    def __init__(self, t: Tensor, axis: int):
        self.sum_axis = axis
            
        super().__init__(t)
    
    def foreward(self, input_values: Tuple[np.array]) -> np.array:
        X = input_values[0]
        
        Y = X.sum(axis=self.sum_axis)
        
        return Y
    
    def reverse_grad(self, input_values: Tuple[np.array], output_value: np.array, top_grad: np.array) -> Tuple[np.array]:
        X = input_values[0]
        dY = top_grad
        
        dX = np.repeat(
            np.expand_dims(dY, axis=self.sum_axis),
            X.shape[self.sum_axis],
            axis=self.sum_axis
        )
        
        return (dX,)


class Add(Op):
    
    def foreward(self, input_values: Tuple[np.array]) -> np.array:
        X = input_values[0]
        Y = input_values[1]
        
        Z = X + Y
        
        return Z
    
    def reverse_grad(self, input_values: Tuple[np.array], output_value: np.array, top_grad: np.array) -> Tuple[np.array]:
        dZ = top_grad
        
        dX = dZ
        dY = dZ
        
        return dX, dY


class Neg(Op):
    
    def foreward(self, input_values: Tuple[np.array]) -> np.array:
        X = input_values[0]
        
        Z = -X
        
        return Z
    
    def reverse_grad(self, input_values: Tuple[np.array], output_value: np.array, top_grad: np.array) -> Tuple[np.array]:
        dZ = top_grad
        
        dX = -dZ
        
        return (dX,)


class Abs(Op):
    
    def foreward(self, input_values: Tuple[np.array]) -> np.array:
        X = input_values[0]
        
        Z = np.abs(X)
        
        return Z
    
    def reverse_grad(self, input_values: Tuple[np.array], output_value: np.array, top_grad: np.array) -> Tuple[np.array]:
        X = input_values[0]
        dZ = top_grad
        
        dX = X.sign * dZ # TODO
        
        return (dX,)


def Sub(A, B):
    return Add(A, Neg(B))


class Mul(Op):
    
    def foreward(self, input_values: Tuple[np.array]) -> np.array:
        X = input_values[0]
        Y = input_values[1]
        
        Z = X * Y
        
        return Z
    
    def reverse_grad(self, input_values: Tuple[np.array], output_value: np.array, top_grad: np.array) -> Tuple[np.array]:
        X = input_values[0]
        Y = input_values[1]
        dZ = top_grad
        
        dX = Y * dZ
        dY = X * dZ
        
        return dX, dY


class MatMul(Op):
    
    def foreward(self, input_values: Tuple[np.array]) -> np.array:
        X = input_values[0]
        Y = input_values[1]
        
        #print(f'matmul {X.shape} @ {Y.shape}')
        Z = X @ Y
        
        return Z
    
    def reverse_grad(self, input_values: Tuple[np.array], output_value: np.array, top_grad: np.array) -> Tuple[np.array]:
        X = input_values[0]  # [A,B]
        Y = input_values[1]  # [B,C]
        #Z = outputs[0]      # [A,C]
        dZ = top_grad        # [A,C]
        
        dX = dZ @ Y.transpose()
        dY = X.transpose() @ dZ
        
        return dX, dY


class Exp(Op):
    
    def foreward(self, input_values: Tuple[np.array]) -> np.array:
        X = input_values[0]
        
        Z = np.exp(X)
        
        return Z
    
    def reverse_grad(self, input_values: Tuple[np.array], output_value: np.array, top_grad: np.array) -> Tuple[np.array]:
        Z = output_value
        dZ = top_grad
        
        dX = Z * dZ
        
        return (dX,)


class Sigm(Op):
    
    def foreward(self, input_values: Tuple[np.array]) -> np.array:
        X = input_values[0]
        
        Z = 1 / (1 + np.exp(-X))
        
        return Z
    
    def reverse_grad(self, input_values: Tuple[np.array], output_value: np.array, top_grad: np.array) -> Tuple[np.array]:
        Z = output_value
        dZ = top_grad
        
        dX = Z * (1 - Z) * dZ
        
        return (dX,)


class Tanh(Op):
    
    def foreward(self, input_values: Tuple[np.array]) -> np.array:
        X = input_values[0]
        
        Z = np.tanh(X)
        
        return Z
    
    def reverse_grad(self, input_values: Tuple[np.array], output_value: np.array, top_grad: np.array) -> Tuple[np.array]:
        Z = output_value
        dZ = top_grad
        
        dX = ((1 - Z)**2) * dZ
        
        return (dX,)


class Relu(Op):
    
    def foreward(self, input_values: Tuple[np.array]) -> np.array:
        X = input_values[0]
        
        Z = (X > 0) * X
        
        return Z
    
    def reverse_grad(self, input_values: Tuple[np.array], output_value: np.array, top_grad: np.array) -> Tuple[np.array]:
        X = input_values[0]
        dZ = top_grad

        dX = (X > 0) * dZ
        
        return (dX,)


class Pow(Op):
    
    def __init__(self, x: Tensor, power: int):
        
        self.power = power
        
        super().__init__(x)
    
    def foreward(self, input_values: Tuple[np.array]) -> np.array:
        X = input_values[0]
        p = self.power
        
        Y = X ** p
        
        return Y
    
    def reverse_grad(self, input_values: Tuple[np.array], output_value: np.array, top_grad: np.array) -> Tuple[np.array]:
        X = input_values[0]
        p = self.power
        dY = top_grad
        
        dY = p * X ** (p-1) * dY
        
        return (dY,)


class StopGrad(Op):
    pass