# Copyright (c) 2021 Jacob F. Valdez. Released under the MIT license.
"""Jnumpy: Jacob's numpy library for machine learning."""


from __future__ import annotations

# Many default parameters are included in jnumpy and are optional.
# I only resort to using `Optional` in the type annotations where the
# context does not make this clear.
from typing import Tuple, List, Union, Optional

import numpy as np
from numpy.lib.arraysetops import isin


V = np.array  # Value type
Vs = Tuple[V]  # tuple of value types
Vss = Union[V, Vs]  # single value or tuple of value types


class Graph:

    EAGER_EXECUTION = 1
    STATIC_EXECUTION = 2  # STATIC execution mode not supported
    EXECUTION_MODE = EAGER_EXECUTION

    @staticmethod
    def set_execution_mode(mode: int) -> None:
        Graph.EXECUTION_MODE = mode

    @staticmethod
    def get_execution_mode() -> int:
        return Graph.EXECUTION_MODE

    @staticmethod
    def print_ancestors(t: T, ord: str = "post", n: int = -2) -> None:
        """Prints all ancestors of a Tensor pre/post-ordered.

        Args:
            t (T): the Tensor to print ancestors of.
            ord (str): the order to print the ancestors in. Either 'pre' or
                'post'. Defaults to 'post'.
            n (int): the number of ancestors to print. If -2, prints all.
        """
        if n == -1:
            return
        if isinstance(t, Op):
            if ord == "pre":
                print(
                    f"{', '.join([input_t.name for input_t in t.input_ts])} -> {t.name} : {t.shape}"
                )
            for input_t in t.input_ts:
                Graph.print_ancestors(input_t, ord, n - 1)
            if ord == "post":
                print(
                    f"{', '.join([input_t.name for input_t in t.input_ts])} -> {t.name} : {t.shape}"
                )
        else:
            print(f"{t.name} : {t.shape}")


class NameScope:

    _GLOBAL_PREFIXES = list()
    _NAMES_REGISTRY = dict()

    @staticmethod
    def get_name(namespace, name):
        """Gets a unique and name-scoped identifier name.
        Makes `name` unique by adding or increasing its integer suffix.
        Separates prefixes from the body using underscores as appropriate.

        Examples:
            >>> make_unique('tensor', 'Add')
            ... 'Add'
            >>> make_unique('tensor', 'Add')
            ... 'Add1'
            >>> make_unique('tensor', 'Add')
            ... 'Add2'
            >>> make_unique('tensor', 'Add1')
            ... 'Add3'
            >>> make_unique('tensor', 'Sub')
            ... 'Sub'

        Args:
            namespace (str): What namespace to track uniqueness under.
            name (str): The name to make unique

        Returns:
            str: `name` but with the suffix possibly changed.
        """
        # get or make namespace
        if namespace not in NameScope._NAMES_REGISTRY:
            NameScope._NAMES_REGISTRY[namespace] = dict()
        namespace_registry = NameScope._NAMES_REGISTRY[namespace]

        # add prefixes
        for prefix in reversed(NameScope._GLOBAL_PREFIXES):
            name = f"{prefix}:{name}"

        # make `name` unique
        basename = name.strip("0123456789")
        name_ext = name[len(basename) :]

        # maybe create slot to store new `basename`
        if basename not in namespace_registry:
            namespace_registry[basename] = list()

        # if the variable name is truely unique, just store it and move on
        if name == basename and -1 not in namespace_registry[basename]:
            # special case, no extension flag
            namespace_registry[basename].append(-1)
            return name

        # otherwise find the next free integer extension to `basename`
        if len(name_ext) == 0:
            name_ext = 1
        name_ext = int(name_ext)
        while name_ext in namespace_registry[basename]:
            name_ext += 1
        namespace_registry[basename].append(name_ext)
        return f"{basename}{name_ext}"

    def __init__(self, *prefixes):
        self.prefixes = prefixes

    def __enter__(self):
        for prefix in self.prefixes:
            NameScope._GLOBAL_PREFIXES.append(prefix)

    def __exit__(self, *args, **kwargs):
        for prefix in self.prefixes:
            NameScope._GLOBAL_PREFIXES.pop()


class T:
    """Tensor.

    Base class for all tensors.

    Args:
        val (V, optional): Value of the variable. Optional if `EXECUTION_MODE`
            is `STATIC`. Otherwise you should supply a value. `Op` tensors should
            call their `forward` method to compute the value before calling the
            base __init__ function.
    """

    def __init__(self, val: Optional[V] = None, name: str = "T"):

        self.name = NameScope.get_name("TENSOR_NAMES", name)
        self.val = val

        if val is None:
            raise "STATIC execution mode not supported"

    def __getitem__(self, key):
        return eval("Index")(self, key)

    def __setitem__(self, key, value):
        raise NotImplementedError("slice assign not yet supported")

    def __add__(self, other):
        return eval("Add")(self, other)

    def __neg__(self):
        return eval("Neg")(self)

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

    def __str__(self):
        return f"Tensor(name={self.name}, shape={self.shape}, {self.val})"

    def __repr__(self):
        self.__str__()

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


Ts = Tuple[T]  # tuple of tensors
Tss = Union[T, Ts]  # single or tuple of tensors


class Var(T):
    """Variable Tensor.
    A variable is a tensor that can be trained.

    Args:
        val (V, optional): Value of the variable. Optional if `EXECUTION_MODE`
            is `STATIC`. Otherwise you should supply a value.
        trainable (bool, optional): Whether the variable is trainable.
            Read by the optimizer when backpropagating gradients.
            The default is False (Changed since jnumpy 1.0).
    """

    def __init__(
        self, val: Optional[V] = None, trainable: bool = False, name: str = "Var"
    ):

        if not isinstance(val, np.ndarray):
            val = np.array(val)

        self.trainable = trainable
        super().__init__(val=val, name=name)


class Op(T):
    """Operator Tensor

    Make sure to set any variables you might need to use in `forward`
    before leaving __init__ when the graph is in eager execution mode.

    Args:
        inputs (Tuple[T]): Tuple of input Tensors (each is only a single T).
    """

    def __init__(self, *inputs: T, name: str = "Op"):

        self.input_ts = inputs

        if Graph.get_execution_mode() == Graph.EAGER_EXECUTION:
            val = self.forward(tuple(i.val for i in inputs))
        else:
            val = None

        super().__init__(name=name, val=val)

    def forward(self, inputs: Vs) -> V:
        raise NotImplementedError("subclasses should implement this method")

    def reverse_grad(self, inputs: Vs, output: V, top_grad: V) -> Vs:
        raise NotImplementedError("subclasses should implement this method")


class Transpose(Op):
    """Transpose operator

    Args:
        t (T): Tensor to transpose
        axes (Tuple[int]): Axes to transpose over
    """

    def __init__(self, t: T, axes: Tuple[int] = None, name: str = "Transpose"):

        self.forward_kwargs = dict()
        self.reverse_kwargs = dict()

        if axes is not None:
            self.forward_kwargs["axes"] = axes
            self.reverse_kwargs["axes"] = tuple(reversed(axes))

        super().__init__(t, name=name)

    def forward(self, inputs: Vs) -> V:
        X = inputs[0]

        Y = X.transpose(**self.forward_kwargs)

        return Y

    def reverse_grad(self, inputs: Vs, output: V, top_grad: V) -> Vs:
        dY = top_grad

        dX = dY.transpose(**self.reverse_kwargs)

        return (dX,)


class Reshape(Op):
    """Tensor reshaping operator.

    Args:
        t (T): Tensor to reshape
        shape (Tuple[int]): New shape of the tensor
    """

    def __init__(self, t: T, shape: Tuple[int], name: str = "Reshape"):

        self.reshape_shape = shape

        super().__init__(t, name=name)

    def forward(self, inputs: Vs) -> V:
        X = inputs[0]

        Y = X.reshape(self.reshape_shape)

        return Y

    def reverse_grad(self, inputs: Vs, output: V, top_grad: V) -> Vs:
        X = inputs[0]
        dY = top_grad

        dX = dY.reshape(X.shape)
        # dX = dY.reshape(tuple(reversed(self.reshape_shape)))

        return (dX,)


class Concat(Op):
    """Concatenates input tensors along an axis

    Args:
        t (T): [description]
        axis (int, optional): Axis to concatenate along. Defaults to 0.
    """

    def __init__(self, ts: List[T], axis: int = 0, name: str = "Concat"):

        self.axis = axis
        self.orig_axis_lens = [t.shape[axis] for t in ts]

        super().__init__(*ts, name=name)

    def forward(self, inputs: Vs) -> V:
        Xs = inputs

        Y = np.concatenate(Xs, axis=self.axis)

        return Y

    def reverse_grad(self, inputs: Vs, output: V, top_grad: V) -> Vs:
        dY = top_grad

        cuts = np.cumsum(self.orig_axis_lens[:-1])
        dXs = np.split(dY, cuts, axis=self.axis)

        return dXs


class Index(Op):
    """Slices a tensor along all axes.

    Args:
        t (T): The tensor to slice
        indices (Tuple[slice]):  The partial or full indices to slice on `t`.
            Can be an index, single slice, tuple of slices, or Ellipsis.
            `None` is not allowed.
    """

    def __init__(self, t: T, indices, name: str = "Index"):
        if not isinstance(indices, tuple):
            indices = (indices,)

        self.indices = indices

        super().__init__(t, name=name)

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
    """Reduce sum oeprator"""

    def __init__(self, t: T, axis: int, name: str = "ReduceSum"):
        self.axis = axis

        super().__init__(t, name=name)

    def forward(self, inputs: Vs) -> V:
        X = inputs[0]

        Y = X.sum(axis=self.axis)

        return Y

    def reverse_grad(self, inputs: Vs, output: V, top_grad: V) -> Vs:
        X = inputs[0]
        dY = top_grad

        dX = np.repeat(
            np.expand_dims(dY, axis=self.axis), X.shape[self.axis], axis=self.axis
        )

        return (dX,)


class ReduceMax(Op):
    """Differentiable max operator"""

    def __init__(self, t: T, axis: int, name: str = "ReduceMax"):
        self.axis = axis

        super().__init__(t, name=name)

    def forward(self, inputs: Vs) -> V:
        X = inputs[0]

        Y = X.max(axis=self.axis)

        return Y

    def reverse_grad(self, inputs: Vs, output: V, top_grad: V) -> Vs:
        X = inputs[0]
        dY = top_grad

        # transpose the input and gradients to make the max indexing on the right
        XT = np.transpose(
            X,
            axes=(
                *range(self.axis),
                *range(self.axis + 1, X.ndim),
                self.axis,
            ),
        )
        # [leading_dims..., axis_len, trailing_dims...]
        # -> [leading_dims..., trailing_dims, axis]

        dYT = dY

        # recompute max indices
        max_indices = np.argmax(XT, axis=-1)  # [leading_dims..., trailing_dims]
        dXT = np.zeros_like(XT)  # [leading_dims..., trailing_dims, axis]

        # completely flatten the indices and gradients
        dXT_flat = np.reshape(dXT, (-1, X.shape[self.axis]))
        #  [?, axis_len] = [leading_dims_1 * leading_dim_2 * ... * trailing_dim_1 * ..., axis_len]
        max_indices_flat = np.reshape(max_indices, (-1,))
        # [?] = [leading_dims_1 * leading_dim_2 * ... * trailing_dim_1 * ...]
        dYT_flat = np.reshape(dYT, (-1,))
        # [?,] = [leading_dims_1 * leading_dim_2 * ... * trailing_dim_1 * ...]

        # set the gradients to the correct indices
        dXT_flat[:, max_indices_flat] = dYT_flat

        # unflatten the gradients
        dXT = np.reshape(dXT_flat, XT.shape)

        # undo the transpose
        dX = np.transpose(
            dXT,
            axes=(
                *range(self.axis),
                dXT.ndim - 1,
                *range(self.axis, dXT.ndim - 1),
            ),
        )

        return (dX,)


class ReduceMin(Op):
    """Differentiable min operator"""

    def __init__(self, t: T, axis: int, name: str = "ReduceMin"):
        self.axis = axis

        super().__init__(t, name=name)

    def forward(self, inputs: Vs) -> V:
        X = inputs[0]

        Y = X.min(axis=self.axis)

        return Y

    def reverse_grad(self, inputs: Vs, output: V, top_grad: V) -> Vs:
        X = inputs[0]
        dY = top_grad

        # transpose the input and gradients to make the max indexing on the right
        XT = np.transpose(
            X,
            axes=(
                *range(self.axis),
                *range(self.axis + 1, X.ndim),
                self.axis,
            ),
        )
        # [leading_dims..., axis_len, trailing_dims...]
        # -> [leading_dims..., trailing_dims, axis]

        dYT = dY

        # recompute max indices
        min_indices = np.argmin(XT, axis=-1)  # [leading_dims..., trailing_dims]
        dXT = np.zeros_like(XT)  # [leading_dims..., trailing_dims, axis]

        # completely flatten the indices and gradients
        dXT_flat = np.reshape(dXT, (-1, X.shape[self.axis]))
        #  [?, axis_len] = [leading_dims_1 * leading_dim_2 * ... * trailing_dim_1 * ..., axis_len]
        min_indices_flat = np.reshape(min_indices, (-1,))
        # [?] = [leading_dims_1 * leading_dim_2 * ... * trailing_dim_1 * ...]
        dYT_flat = np.reshape(dYT, (-1,))
        # [?,] = [leading_dims_1 * leading_dim_2 * ... * trailing_dim_1 * ...]

        # set the gradients to the correct indices
        dXT_flat[:, min_indices_flat] = dYT_flat

        # unflatten the gradients
        dXT = np.reshape(dXT_flat, XT.shape)

        # undo the transpose
        dX = np.transpose(
            dXT,
            axes=(
                *range(self.axis),
                dXT.ndim - 1,
                *range(self.axis, dXT.ndim - 1),
            ),
        )

        return (dX,)


class NaN2Num(Op):
    """Not a number correction operator:

    y = x
    neginf < y < posinf
    neginf < dx(dy) < posinf
    """

    def __init__(
        self, t: T, posinf: float = 1e3, neginf: float = -1e3, name: str = "NaN2Num"
    ):
        self.posinf = posinf
        self.neginf = neginf

        super().__init__(t, name=name)

    def forward(self, inputs: Vs) -> V:
        X = inputs[0]

        Z = np.nan_to_num(X, posinf=self.posinf, neginf=self.neginf)

        return Z

    def reverse_grad(self, inputs: Vs, output: V, top_grad: V) -> Vs:
        dZ = top_grad

        dX = np.nan_to_num(dZ, posinf=10.0, neginf=-10.0)

        return (dX,)


class Linear(Op):
    """Linear operator: y = x"""

    def __init__(self, t: T, name: str = "Linear"):
        super().__init__(t, name=name)

    def forward(self, inputs: Vs) -> V:
        X = inputs[0]

        Z = X

        return Z

    def reverse_grad(self, inputs: Vs, output: V, top_grad: V) -> Vs:
        dZ = top_grad

        dX = dZ

        return (dX,)


class StopGrad(Op):
    """Stop gradient operator: y = x. dx(dy) = 0"""

    def __init__(self, t: T, name: str = "StopGrad"):
        super().__init__(t, name=name)

    def forward(self, inputs: Vs) -> V:
        X = inputs[0]

        Z = X

        return Z

    def reverse_grad(self, inputs: Vs, output: V, top_grad: V) -> Vs:
        dZ = top_grad

        dX = np.zeros_like(dZ)

        return (dX,)


class Neg(Op):
    """Negation operator: y = -x"""

    def __init__(self, t: T, name: str = "Neg"):
        super().__init__(t, name=name)

    def forward(self, inputs: Vs) -> V:
        X = inputs[0]

        Z = -X

        return Z

    def reverse_grad(self, inputs: Vs, output: V, top_grad: V) -> Vs:
        dZ = top_grad

        dX = -dZ

        return (dX,)


class Add(Op):
    """Addition operator: z = x + y"""

    def __init__(self, x: T, y: T, name: str = "Add"):
        super().__init__(x, y, name=name)

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
    """Subtraction operator: z = x - y"""

    def __init__(self, x: T, y: T, name: str = "Sub"):
        super().__init__(x, y, name=name)

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
    """Multipulcation operator: z = xy"""

    def __init__(self, x: T, y: T, name: str = "Mul"):
        super().__init__(x, y, name=name)

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
    """Matrix multipulcation operator: Z = XY

    for sizes:
        X: [B1..Bn, l, m],
        Y: [m, n],
        Z: [B1..Bn, l, n]
    """

    def __init__(self, x: T, y: T, name: str = "MatMul"):

        assert (
            x.shape[-1] == y.shape[-2]
        ), "Incompatible shapes: Inner dimensions not equal."
        assert y.ndim == 2, "Incompatible shapes: Y must be a matrix."

        super().__init__(x, y, name=name)

    def forward(self, inputs: Vs) -> V:
        X = inputs[0]
        Y = inputs[1]

        Z = X @ Y

        return Z

    def reverse_grad(self, inputs: Vs, output: V, top_grad: V) -> Vs:
        X = inputs[0]  # [B1..Bn, l, m]
        Y = inputs[1]  # [m, n]
        dZ = top_grad  # [B1..Bn, l, n]

        dX = dZ @ Y.T
        # [B1..Bn, l, m] = [B1..Bn, l, n] x [n, m]

        dY = np.einsum("... l m , ... l n -> ... m n ", X, dZ)
        while dY.ndim > 2:
            dY = np.sum(dY, axis=0)
        # [m, n] <- [B1..Bn, l, m] x [B1..Bn, l, n]

        return dX, dY


class Exp(Op):
    """Exponential operator: y = e^x"""

    def __init__(self, x: T, name: str = "Exp"):
        super().__init__(x, name=name)

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
    """Sigmoid operator: y = S(x) = 1 / (1 + e^-x)"""

    def __init__(self, x: T, name: str = "Sigm"):
        super().__init__(x, name=name)

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
    """Hyperbolic tangent: y = tanh(x)"""

    def __init__(self, x: T, name: str = "Tanh"):
        super().__init__(x, name=name)

    def forward(self, inputs: Vs) -> V:
        X = inputs[0]

        Z = np.tanh(X)

        return Z

    def reverse_grad(self, inputs: Vs, output: V, top_grad: V) -> Vs:
        Z = output
        dZ = top_grad

        dX = ((1 - Z) ** 2) * dZ

        return (dX,)


class Relu(Op):
    """Rectified linear unit: y = ReLU(x)"""

    def __init__(self, x: T, name: str = "ReLU"):
        super().__init__(x, name=name)

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
    """Threshold operator:

    y = 1 if x >= 0
    y = 0 if x < 0"""

    def __init__(self, x: T, name: str = "Threshold"):
        super().__init__(x, name=name)

    def forward(self, inputs: Vs) -> V:
        X = inputs[0]

        Z = X >= 0

        return Z

    def reverse_grad(self, inputs: Vs, output: V, top_grad: V) -> Vs:
        dZ = top_grad

        dX = dZ

        return (dX,)


class Pow(Op):
    """Power operator: y = x^N

    Args:
        x (T): base
        N (int): power
    """

    def __init__(self, x: T, power: int, name: str = "Pow"):

        self.power = power

        super().__init__(x, name=name)

    def forward(self, inputs: Vs) -> V:
        X = inputs[0]
        p = self.power

        Y = X ** p

        return Y

    def reverse_grad(self, inputs: Vs, output: V, top_grad: V) -> Vs:
        X = inputs[0]
        p = self.power
        dY = top_grad

        dX = p * X ** (p - 1) * dY
        dX = np.nan_to_num(dX, posinf=1e3, neginf=-1e3)

        return (dX,)


class Optimizer:
    """Base class for optimizers."""

    def minimize(self, t: T):
        pass


class SGD(Optimizer):
    """Gradient descent optimizer.

    NOTE for debuggers:
        `SGD` recursively backpropagates in depth-first fashion resulting in potential
        redundant gradient propagations. For example, if you have a graph like:
        X1 -> X2 -> ... -> Xn
        Xn -> Y1
        Xn -> Y2
        (Y1, Y2) -> Z
        Then backpropagating from Z down results in 2n+3 backpropagation step since
        gradients are backpropagated down X1...Xn twice. If you want to maximize efficiency
        you will have to manually take the mean of dXn from Y1 and Y2. Innefficient: Yes. Works: Yes*
        (as long as you're not tracking state outside of the jnp.T.reverse_grad function)
        Once static execution mode is implemented, the optimizer will be able to propagate
        gradients in a propper topological order. However, the main thesis of gradient descent
        is about *small* updates to the parameters, so it's not too *big* a deal. ;)

    Args:
        lr (float, optional): Learning rate. Multiplied by gradients before applying
            them to Var nodes in `bprop` method. Defaults to 0.001.
        debug (bool, optional): Whether to print out debug information during backpropagation.
            You can change this later by setting `self.debug`. Defaults to False.
    """

    def __init__(self, lr: float = 0.001, debug: bool = False):

        self.lr = lr
        self.debug = debug

        super().__init__()

    def minimize(self, t: T):

        if Graph.get_execution_mode() == Graph.STATIC_EXECUTION:
            raise "STATIC execution mode not enabled"

        if self.debug:
            print(64 * "=")
            print(
                "minimizing loss for the following tensors (only a depth of 3 are displayed):"
            )
            Graph.print_ancestors(t, n=3)
            print(64 * "=")

        self.bprop(t_out=t, output_grad=-np.ones_like(t.val))

    def bprop(self, t_out: T, output_grad: V):
        """Backpropagates `output_grad` down `t_out`.
        If t_out is an `Op`, this method recursively backpropagates down
            each of t_out's parents in depth-first fashion.
        If t_out is a `Var`, this method applies `self.lr*output_grad` to the Var.

        Args:
            t_out (T): Output tensor to start backpropagation from.
            output_grad (V): Gradient of output tensor. You should make this
                a one's array if you're just computing partial derivatives.
        """

        # if self.debug:
        #     print("bprop", t_out, np.mean(output_grad))

        output_grad = np.nan_to_num(output_grad, posinf=10.0, neginf=-10.0)

        # iteratively called
        if isinstance(t_out, Var):
            if t_out.trainable:
                # print('output_grad', output_grad.shape)
                if self.debug:
                    print(
                        f"{t_out.name} old shape={t_out.shape}, grad shape={output_grad.shape}"
                    )
                # print(t_out.val)

                # yucky duct tape to handle batch size differences
                if t_out.shape[0] == 1 and output_grad.shape[0] > 1:
                    output_grad = np.sum(output_grad, axis=0)[None, ...]

                t_out.val = t_out.val + (self.lr * output_grad)
                if self.debug:
                    print(f"{t_out.name} new shape={t_out.shape}")
                    # print(t_out.val)

        elif isinstance(t_out, Op):

            if self.debug:
                print(
                    f"{t_out.name} shape={t_out.shape}, grad shape={output_grad.shape}"
                )

            input_grads = t_out.reverse_grad(
                inputs=tuple(t.val for t in t_out.input_ts),
                output=t_out.val,
                top_grad=output_grad,
            )

            if self.debug:
                print("ancestors: ")
                for input_t, input_grad in zip(t_out.input_ts, input_grads):
                    print(f" - {input_t.name}, output_grad.shape={input_grad.shape}")

            for input_t, input_grad in zip(t_out.input_ts, input_grads):
                self.bprop(t_out=input_t, output_grad=input_grad)

        return
