# Jnumpy: Jacob's numpy library for machine learning
# Copyright (c) 2021 Jacob F. Valdez. Released under the MIT license.

from typing import Tuple, List, Optional, Union
import csv
import functools
import numpy as np
import logging
import math


V = np.array # V is for Value type
Vs = Tuple[V]
Vss = Union[V,Vs]


class ExecutionMode:
    EAGER=1
    STATIC=2  # STATIC execution mode not supported
    
EXECUTION_MODE = ExecutionMode.EAGER


class T:
    """Tensor"""
    
    def __init__(self, val: Optional[V] = None):
        self.val = val
        
        if val is None:
            raise 'STATIC execution mode not supported'
    
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


Ts = Tuple[T]
Tss = Union[T,Ts]


class Var(T):
    """Variable Tensor"""
    def __init__(self, val: Optional[V] = None, trainable: bool = True):
        
        self.trainable = trainable
        super().__init__(val=val)


class Op(T):
    """Operation-backed Tensor"""
    
    def __init__(self, *inputs: T):
        """Make sure to set any variables you might need in `foreward` 
        before initializing when the graph is in eager execution mode
        """
        
        self.input_ts = inputs
        
        if EXECUTION_MODE == ExecutionMode.EAGER:
            val = self.foreward(tuple(i.val for i in inputs))[0]
        else:
            val = None
        
        super().__init__(val=val)
        
    def foreward(self, inputs: Vs) -> Vs:
        raise NotImplementedError('subclasses should implement this method')
    
    def reverse_grad(self, inputs: Vs, outputs: Vs, top_grads: Vs) -> Vs:
        raise NotImplementedError('subclasses should implement this method')


class Transpose(Op):
    
    def __init__(self, t: T, axes: Optional[tuple] = None):
        
        self.foreward_kwargs = dict()
        self.reverse_kwargs = dict()
        
        if axes is not None:
            self.foreward_kwargs['axes'] = axes
            self.reverse_kwargs['axes'] = tuple(reversed(axes))
            
        super().__init__(t)
    
    def foreward(self, inputs: Vs) -> Vs:
        X = inputs[0]
        
        Y = X.transpose(**self.foreward_kwargs)
        
        return (Y,)
    
    def reverse_grad(self, inputs: Vs, outputs: Vs, top_grads: Vs) -> Vs:
        dY = top_grads[0]
        
        dX = dY.transpose(**self.reverse_kwargs)
        
        return (dX,)


class ReduceSum(Op):
    
    def __init__(self, t: T, axis: int):
        self.sum_axis = axis
            
        super().__init__(t)
    
    def foreward(self, inputs: Vs) -> Vs:
        X = inputs[0]
        
        Y = X.sum(axis=self.sum_axis)
        
        return (Y,)
    
    def reverse_grad(self, inputs: Vs, outputs: Vs, top_grads: Vs) -> Vs:
        X = inputs[0]
        dY = top_grads[0]
        
        dX = np.repeat(
            np.expand_dims(dY, axis=self.sum_axis),
            X.shape[self.sum_axis],
            axis=self.sum_axis
        )
        
        return (dX,)


class Add(Op):
    
    def foreward(self, inputs: Vs) -> Vs:
        X = inputs[0]
        Y = inputs[1]
        
        Z = X + Y
        
        return (Z,)
    
    def reverse_grad(self, inputs: Vs, outputs: Vs, top_grads: Vs) -> Vs:
        dZ = top_grads[0]
        
        dX = dZ
        dY = dZ
        
        return dX, dY


class Neg(Op):
    
    def foreward(self, inputs: Vs) -> Vs:
        X = inputs[0]
        
        Z = -X
        
        return (Z,)
    
    def reverse_grad(self, inputs: Vs, outputs: Vs, top_grads: Vs) -> Vs:
        dZ = top_grads[0]
        
        dX = -dZ
        
        return (dX,)


def Sub(A, B):
    return Add(A, Neg(B))


class Mul(Op):
    
    def foreward(self, inputs: Vs) -> Vs:
        X = inputs[0]
        Y = inputs[1]
        
        Z = X * Y
        
        return (Z,)
    
    def reverse_grad(self, inputs: Vs, outputs: Vs, top_grads: Vs) -> Vs:
        X = inputs[0]
        Y = inputs[1]
        dZ = top_grads[0]
        
        dX = Y * dZ
        dY = X * dZ
        
        return dX, dY


class MatMul(Op):
    
    def foreward(self, inputs: Vs) -> Vs:
        X = inputs[0]
        Y = inputs[1]
        X = np.nan_to_num(X, posinf=10., neginf=-10.)
        Y = np.nan_to_num(Y, posinf=10., neginf=-10.)
        
        #print(f'matmul {X.shape} @ {Y.shape}')
        Z = X @ Y
        
        Z = np.nan_to_num(Z, posinf=10., neginf=-10.)
        return (Z,)
    
    def reverse_grad(self, inputs: Vs, outputs: Vs, top_grads: Vs) -> Vs:
        X = inputs[0]  # [A,B]
        Y = inputs[1]  # [B,C]
        #Z = outputs[0]  # [A,C]
        dZ = top_grads[0]  # [A,C]
        
        X = np.nan_to_num(X, posinf=10., neginf=-10.)
        Y = np.nan_to_num(Y, posinf=10., neginf=-10.)
        dZ = np.nan_to_num(dZ, posinf=10., neginf=-10.)
        
        dX = dZ @ Y.transpose()
        dY = X.transpose() @ dZ
        
        dX = np.nan_to_num(dX, posinf=10., neginf=-10.)
        dY = np.nan_to_num(dY, posinf=10., neginf=-10.)
        return dX, dY


class Exp(Op):
    
    def foreward(self, inputs: Vs) -> Vs:
        X = inputs[0]
        
        Z = np.exp(X)
        
        return (Z,)
    
    def reverse_grad(self, inputs: Vs, outputs: Vs, top_grads: Vs) -> Vs:
        Z = outputs[0]
        dZ = top_grads[0]
        
        dX = Z * dZ
        
        return (dX,)


class Sigm(Op):
    
    def foreward(self, inputs: Vs) -> Vs:
        X = inputs[0]
        
        Z = 1 / (1 + np.exp(-X))
        
        return (Z,)
    
    def reverse_grad(self, inputs: Vs, outputs: Vs, top_grads: Vs) -> Vs:
        Z = outputs[0]
        dZ = top_grads[0]
        
        dX = Z * (1 - Z) * dZ
        
        return (dX,)


class Tanh(Op):
    
    def foreward(self, inputs: Vs) -> Vs:
        X = inputs[0]
        
        Z = np.tanh(X)
        
        return (Z,)
    
    def reverse_grad(self, inputs: Vs, outputs: Vs, top_grads: Vs) -> Vs:
        Z = outputs[0]
        dZ = top_grads[0]
        
        dX = ((1 - Z)**2) * dZ
        
        return (dX,)


class Relu(Op):
    
    def foreward(self, inputs: Vs) -> Vs:
        X = inputs[0]
        
        Z = (X > 0) * X
        
        return (Z,)
    
    def reverse_grad(self, inputs: Vs, outputs: Vs, top_grads: Vs) -> Vs:
        X = inputs[0]
        dZ = top_grads[0]
        
        dX = (X > 0) * dZ
        
        return (dX,)


class Threshold(Op):
    
    def foreward(self, inputs: Vs) -> Vs:
        X = inputs[0]
        
        Z = (X >= 0)
        
        return (Z,)
    
    def reverse_grad(self, inputs: Vs, outputs: Vs, top_grads: Vs) -> Vs:
        dZ = top_grads[0]
        
        dX = dZ
        
        return (dX,)


class Pow(Op):
    
    def __init__(self, x: T, power: int):
        
        self.power = power
        
        super().__init__(x)
    
    def foreward(self, inputs: Vs) -> Vs:
        X = inputs[0]
        p = self.power
        
        Y = X ** p
        
        return (Y,)
    
    def reverse_grad(self, inputs: Vs, outputs: Vs, top_grads: Vs) -> Vs:
        X = inputs[0]
        p = self.power
        dY = top_grads[0]
        
        dY = p * X ** (p-1) * dY
        dY = np.nan_to_num(dY, posinf=10., neginf=-10.)
        
        return (dY,)


class Optimizer:
    
    def minimize(self, t: T):
        pass


class SGD(Optimizer):
    
    def __init__(self, lr: float = 0.001):
        
        self.lr = lr
        self.debug = False
        
        super().__init__()
    
    def minimize(self, t: T):
        
        if EXECUTION_MODE == ExecutionMode.STATIC:
            raise 'STATIC execution mode not enabled'
        
        self.bprop(t_out=t, output_grad=-np.ones_like(t.val))
        
    def bprop(self, t_out: T, output_grad: V):
        
        output_grad = np.nan_to_num(output_grad, posinf=10., neginf=-10.)
        
        assert isinstance(t_out, (Var, Op))
        
        if self.debug:
            print(f'bp {t_out} output_grads:')
            print(output_grad)
        
        """
        This approach does not efficiently handle weights that are consumed by multiple nodes
        It would be better to treat backpropagation from a spreading-network-delta perspective
        than assume everything is a tree (That's also how I should do STATIC execution refresh)
        This should still work though, but it's just going to set the same weight multiple times
        for every downstream consumer.
        """
        
        # iteratively called
        if isinstance(t_out, Var):
            if t_out.trainable:
                #print('output_grad', output_grad.shape)
                if self.debug:                    
                    print('t_out.val (old)', t_out.val.shape)
                    print(t_out.val)
                
                # yucky duct tape to handle batch size differences
                if t_out.val.shape[0] == 1 and output_grad.shape[0] > 1:
                    output_grad = np.sum(output_grad, axis=0)[None, ...]

                t_out.val = t_out.val + (self.lr * output_grad)
                if self.debug:
                    print('t_out.val (new)', t_out.val.shape)
                    print(t_out.val)
            
        elif isinstance(t_out, Op):
            input_grads = t_out.reverse_grad(
                inputs=tuple(t.val for t in t_out.input_ts),
                outputs=(t_out.val,), 
                top_grads=(output_grad,))
            
            for input_t, input_grad in zip(t_out.input_ts, input_grads):
                self.bprop(t_out=input_t, output_grad=input_grad)
        
        return
