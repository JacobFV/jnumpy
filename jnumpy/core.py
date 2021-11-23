# Copyright (c) 2021 Jacob F. Valdez. Released under the MIT license.
"""Jnumpy: Jacob's numpy library for machine learning."""


from __future__ import annotations

# Many default parameters are included in jnumpy and are optional.
# I only resort to using `Optional` in the type annotations where the
# context does not make this clear.
from typing import Tuple, List, Union, Optional

import numpy as np


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

    def __getitem__(self, key):
        return eval("Index")(self, key)

    def __setitem__(self, key, value):
        raise NotImplementedError('slice assign not yet supported')
    
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

    @property
    def shape(self):
        return self.val.shape

    @property
    def ndim(self):
        return self.val.ndim

    @property
    def size(self):
        return self.val.size

    @property
    def dtype(self):
        return self.val.dtype

    @property
    def T(self, axes: Tuple[int] = None):
        return eval("Transpose")(self, axes=axes)

    def __repr__(self):
        return f"Tensor({self.val})"

    def __str__(self):
        return f"Tensor({self.val})"

    def __eq__(self, other):
        return self.val == other.val

    def __hash__(self):
        return hash(self.val)

    def __iter__(self):
        return iter(self.val)

    def __len__(self):
        return len(self.val)

    def __getstate__(self):
        return self.val.__getstate__()

    def __setstate__(self, state):
        self.val = state

    def __array__(self):
        return self.val.__array__()


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
        """Make sure to set any variables you might need in `forward` 
        before initializing when the graph is in eager execution mode
        """
        
        self.input_ts = inputs
        
        if EXECUTION_MODE == ExecutionMode.EAGER:
            val = self.forward(tuple(i.val for i in inputs))
        else:
            val = None
        
        super().__init__(val=val)
        
    def forward(self, inputs: Vs) -> V:
        raise NotImplementedError('subclasses should implement this method')
    
    def reverse_grad(self, inputs: Vs, output: V, top_grad: V) -> Vs:
        raise NotImplementedError('subclasses should implement this method')


class Transpose(Op):
    
    def __init__(self, t: T, axes: Tuple[int] = None):
        
        self.forward_kwargs = dict()
        self.reverse_kwargs = dict()
        
        if axes is not None:
            self.forward_kwargs['axes'] = axes
            self.reverse_kwargs['axes'] = tuple(reversed(axes))
            
        super().__init__(t)
    
    def forward(self, inputs: Vs) -> V:
        X = inputs[0]
        
        Y = X.transpose(**self.forward_kwargs)
        
        return Y
    
    def reverse_grad(self, inputs: Vs, output: V, top_grad: V) -> Vs:
        dY = top_grad

        dX = dY.transpose(**self.reverse_kwargs)
        
        return (dX,)


class Reshape(Op):
    
    def __init__(self, t: T, shape: Tuple[int]):
        
        self.reshape_shape = shape
            
        super().__init__(t)
    
    def forward(self, inputs: Vs) -> V:
        X = inputs[0]
        
        Y = X.reshape(self.reshape_shape)
        
        return Y
    
    def reverse_grad(self, inputs: Vs, output: V, top_grad: V) -> Vs:
        dY = top_grad
        
        dX = dY.reshape(tuple(reversed(self.reshape_shape)))
        
        return (dX,)
 

class Concat(Op):
    
    def __init__(self, ts: List[T], axis: int = 0):
        """Concatenates input tensors along an axis

        Args:
            t (T): [description]
            axis (int, optional): Axis to concatenate along. Defaults to 0.
        """
        
        self.axis = axis
        self.orig_axis_lens = [t.shape[axis] for t in ts]

        super().__init__(*ts)
    
    def forward(self, inputs: Vs) -> V:
        Xs = inputs
        
        Y = np.concatenate(Xs, axis=self.axis)
        
        return Y
    
    def reverse_grad(self, inputs: Vs, output: V, top_grad: V) -> Vs:
        dY = top_grad
        
        dXs = np.split(dY, self.orig_axis_lens, axis=self.axis)[0]
        
        return dXs


class Index(Op):
    
    def __init__(self, t: T, indices):
        """Slices a tensor along all axes.

        Args:
            t (T): The tensor to slice
            indices (Tuple[slice]):  The partial or full indices to slice on `t`.
                Can be an index, single slice, tuple of slices, or Ellipsis.
                `None` is not allowed.
        """
        if not isinstance(indices, tuple):
            indices = (indices,)

        self.indices = indices
        
        super().__init__(t)
    
    def forward(self, inputs: Vs) -> V:
        X = inputs[0]
        
        Y = X[self.indices]
        
        return Y
    
    def reverse_grad(self, inputs: Vs, output: V, top_grad: V) -> Vs:
        X = inputs[0]
        dY = top_grad
        
        dX = np.zeros(X.shape)
        dX[self.indices] = dY
        
        return (dX,)


class ReduceSum(Op):
    
    def __init__(self, t: T, axis: int):
        self.axis = axis
            
        super().__init__(t)
    
    def forward(self, inputs: Vs) -> V:
        X = inputs[0]
        
        Y = X.sum(axis=self.axis)
        
        return Y
    
    def reverse_grad(self, inputs: Vs, output: V, top_grad: V) -> Vs:
        X = inputs[0]
        dY = top_grad
        
        dX = np.repeat(
            np.expand_dims(dY, axis=self.axis),
            X.shape[self.axis],
            axis=self.axis
        )
        
        return (dX,)


class ReduceMax(Op):
    """Differentiable max operator"""
    
    def __init__(self, t: T, axis: int):
        self.axis = axis
            
        super().__init__(t)
    
    def forward(self, inputs: Vs) -> V:
        X = inputs[0]
        
        Y = X.max(axis=self.axis)
        
        return Y
    
    def reverse_grad(self, inputs: Vs, output: V, top_grad: V) -> Vs:
        X = inputs[0]
        dY = top_grad
        
        print(X.shape, dY.shape)

        dX = np.zeros_like(X)
        dX[np.argmax(X, axis=self.axis)] = dY

        print(dX.shape)
        
        return (dX,)

class ReduceMin(Op):
    """Differentiable min operator"""
    
    def __init__(self, t: T, axis: int):
        self.axis = axis
            
        super().__init__(t)
    
    def forward(self, inputs: Vs) -> V:
        X = inputs[0]
        
        Y = X.min(axis=self.axis)
        
        return Y
    
    def reverse_grad(self, inputs: Vs, output: V, top_grad: V) -> Vs:
        X = inputs[0]
        dY = top_grad
        
        dX = np.zeros_like(X)
        dX[np.argmin(X, axis=self.axis)] = dY
        
        return (dX,)


class NaN2Num(Op):
    
    def __init__(self, t: T, posinf: float = 1e3, neginf: float = -1e3):
        self.posinf = posinf
        self.neginf = neginf
            
        super().__init__(t)

    def forward(self, inputs: Vs) -> V:
        X = inputs[0]
        
        Z = np.nan_to_num(X, posinf=self.posinf, neginf=self.neginf)
        
        return Z
    
    def reverse_grad(self, inputs: Vs, output: V, top_grad: V) -> Vs:
        dZ = top_grad
        
        dX = np.nan_to_num(dZ, posinf=10., neginf=-10.)
        
        return (dX,)


class Linear(Op):
    
    def forward(self, inputs: Vs) -> V:
        X = inputs[0]
        
        Z = X
        
        return Z
    
    def reverse_grad(self, inputs: Vs, output: V, top_grad: V) -> Vs:
        dZ = top_grad
        
        dX = dZ
        
        return (dX,)


class StopGrad(Op):
    
    def forward(self, inputs: Vs) -> V:
        X = inputs[0]
        
        Z = X
        
        return Z
    
    def reverse_grad(self, inputs: Vs, output: V, top_grad: V) -> Vs:
        dZ = top_grad
        
        dX = np.zeros_like(dZ)
        
        return (dX,)


class Neg(Op):
    
    def forward(self, inputs: Vs) -> V:
        X = inputs[0]
        
        Z = -X
        
        return Z
    
    def reverse_grad(self, inputs: Vs, output: V, top_grad: V) -> Vs:
        dZ = top_grad
        
        dX = -dZ
        
        return (dX,)


class Add(Op):
    
    def forward(self, inputs: Vs) -> V:
        X = inputs[0]
        Y = inputs[1]
        
        Z = X + Y
        
        return Z
    
    def reverse_grad(self, inputs: Vs, output: V, top_grad: V) -> Vs:
        dZ = top_grad
        
        dX = dZ
        dY = dZ
        
        return dX, dY


class Sub(Op):
    
    def forward(self, inputs: Vs) -> V:
        X = inputs[0]
        Y = inputs[1]
        
        Z = X - Y
        
        return Z
    
    def reverse_grad(self, inputs: Vs, output: V, top_grad: V) -> Vs:
        dZ = top_grad
        
        dX = dZ
        dY = -dZ
        
        return dX, dY


class Mul(Op):
    
    def forward(self, inputs: Vs) -> V:
        X = inputs[0]
        Y = inputs[1]
        
        Z = X * Y
        
        return Z
    
    def reverse_grad(self, inputs: Vs, output: V, top_grad: V) -> Vs:
        X = inputs[0]
        Y = inputs[1]
        dZ = top_grad
        
        dX = Y * dZ
        dY = X * dZ
        
        return dX, dY


class MatMul(Op):
    
    def forward(self, inputs: Vs) -> V:
        X = inputs[0]
        Y = inputs[1]
        
        Z = X @ Y
        
        return Z
    
    def reverse_grad(self, inputs: Vs, output: V, top_grad: V) -> Vs:
        X = inputs[0]  # [A,B]
        Y = inputs[1]  # [B,C]
        dZ = top_grad  # [A,C]
        
        dX = dZ @ Y.transpose()
        dY = X.transpose() @ dZ
        
        return dX, dY


class Exp(Op):
    
    def forward(self, inputs: Vs) -> V:
        X = inputs[0]
        
        Z = np.exp(X)
        
        return Z
    
    def reverse_grad(self, inputs: Vs, output: V, top_grad: V) -> Vs:
        Z = output
        dZ = top_grad
        
        dX = Z * dZ
        
        return (dX,)


class Sigm(Op):
    
    def forward(self, inputs: Vs) -> V:
        X = inputs[0]
        
        Z = 1 / (1 + np.exp(-X))
        
        return Z
    
    def reverse_grad(self, inputs: Vs, output: V, top_grad: V) -> Vs:
        Z = output
        dZ = top_grad
        
        dX = Z * (1 - Z) * dZ
        
        return (dX,)


class Tanh(Op):
    
    def forward(self, inputs: Vs) -> V:
        X = inputs[0]
        
        Z = np.tanh(X)
        
        return Z
    
    def reverse_grad(self, inputs: Vs, output: V, top_grad: V) -> Vs:
        Z = output
        dZ = top_grad
        
        dX = ((1 - Z)**2) * dZ
        
        return (dX,)


class Relu(Op):
    
    def forward(self, inputs: Vs) -> V:
        X = inputs[0]
        
        Z = (X > 0) * X
        
        return Z
    
    def reverse_grad(self, inputs: Vs, output: V, top_grad: V) -> Vs:
        X = inputs[0]
        dZ = top_grad
        
        dX = (X > 0) * dZ
        
        return (dX,)


class Threshold(Op):
    
    def forward(self, inputs: Vs) -> V:
        X = inputs[0]
        
        Z = (X >= 0)
        
        return Z
    
    def reverse_grad(self, inputs: Vs, output: V, top_grad: V) -> Vs:
        dZ = top_grad
        
        dX = dZ
        
        return (dX,)


class Pow(Op):
    
    def __init__(self, x: T, power: int):
        
        self.power = power
        
        super().__init__(x)
    
    def forward(self, inputs: Vs) -> V:
        X = inputs[0]
        p = self.power
        
        Y = X ** p
        
        return Y
    
    def reverse_grad(self, inputs: Vs, output: V, top_grad: V) -> Vs:
        X = inputs[0]
        p = self.power
        dY = top_grad
        
        dX = p * X ** (p-1) * dY
        dX = np.nan_to_num(dX, posinf=1e3, neginf=-1e3)
        
        return (dX,)


class Optimizer:
    
    def minimize(self, t: T):
        pass


class SGD(Optimizer):
    
    def __init__(self, lr: float = 0.001, debug: bool = False):
        """Gradient descent optimizer.

        Args:
            lr (float, optional): Learning rate. Multiplied by gradients before applying
                them to Var nodes in `bprop` method. Defaults to 0.001.
            debug (bool, optional): Whether to print out debug information during backpropagation. 
                You can change this later by setting `self.debug`. Defaults to False.
        """
        
        self.lr = lr
        self.debug = debug
        
        super().__init__()
    
    def minimize(self, t: T):
        
        if EXECUTION_MODE == ExecutionMode.STATIC:
            raise 'STATIC execution mode not enabled'
        
        self.bprop(t_out=t, output_grad=-np.ones_like(t.val))
        
    def bprop(self, t_out: T, output_grad: V):
        """Recursively backpropagates `output_grad` starting from `t_out`.

        Args:
            t_out (T): Output tensor to start backpropagation from.
            output_grad (V): Gradient of output tensor. You should make this
                a one's array if you're just computing partial derivatives.

        NOTE for debuggers:
            I'm sorry you're reading this. This optimizer takes a depth-first search of the graph
            resulting in redundant gradient propagations. Innefficient: Yes. Works: Yes* 
            (as long as you're not tracking state outside of the jnp.T.reverse_grad function)

            Once static execution mode is implemented, the optimizer will be able to propagate 
            gradients in a propper topological order. However, the main thesis of gradient descent 
            is about *small* updates to the parameters, so it's not too *big* a deal. ;)
        """
        
        if self.debug:
            print('bprop', t_out, np.mean(output_grad), output_grad)
        
        output_grad = np.nan_to_num(output_grad, posinf=10., neginf=-10.)
        
        # iteratively called
        if isinstance(t_out, Var):
            if t_out.trainable:
                #print('output_grad', output_grad.shape)
                if self.debug:                    
                    print('t_out.val (old)', t_out.shape)
                    print(t_out.val)
                
                # yucky duct tape to handle batch size differences
                if t_out.shape[0] == 1 and output_grad.shape[0] > 1:
                    output_grad = np.sum(output_grad, axis=0)[None, ...]

                t_out.val = t_out.val + (self.lr * output_grad)
                if self.debug:
                    print('t_out.val (new)', t_out.shape)
                    print(t_out.val)
            
        elif isinstance(t_out, Op):
            input_grads = t_out.reverse_grad(
                inputs=tuple(t.val for t in t_out.input_ts),
                output=t_out.val, top_grad=output_grad)
            
            for input_t, input_grad in zip(t_out.input_ts, input_grads):
                self.bprop(t_out=input_t, output_grad=input_grad)
        
        return