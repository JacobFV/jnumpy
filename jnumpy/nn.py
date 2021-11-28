# Copyright (c) 2021 Jacob F. Valdez. Released under the MIT license.
"""High-level abstractions for neural network design."""


from __future__ import annotations

import functools
import itertools
from typing import Tuple, List, Union, Callable

import numpy as np

import jnumpy.core as jnp


class Layer:
    """Base class for all layers (including Sequential).

    If you want to implement a new layer, you should inherit from this class.
    You'll probabbly only need to override the `build`, `forward`, and
    `trainable_variables` methods.

    Internally tracks the loss of the layer in `self._loss`.
    This variable is reset before each forward pass unless you
    override the `__call__` method.
    """

    def __init__(self, name: str = "Layer") -> None:
        self.name = jnp.NameScope.get_name("LAYER_NAMES", name)
        self._built = False
        with jnp.NameScope(self.name):
            self._reset_loss()

    @property
    def loss(self) -> jnp.T:
        return self._loss

    def _reset_loss(self):
        self._loss = jnp.Var(np.zeros(()), name=f"initial_loss")

    @property
    def trainable_variables(self) -> List[jnp.T]:
        return []

    def build(self, input_shape: Tuple[int]) -> None:
        """Initialize the layer with the input shape.

        Args:
            input_shape (Tuple[int]): The shape of the input.
        """
        pass

    def forward(self, X_T: jnp.T) -> jnp.T:
        """Forward pass of the layer. Should be implemented by subclasses.

        Args:
            X_T (jnp.T): The input to the layer.

        Returns:
            jnp.T: The output of the layer.
        """
        pass

    def __call__(self, X_T: jnp.T) -> jnp.T:
        with jnp.NameScope(self.name):
            if not self._built:
                self.build(X_T.shape)
                self._built = True

            # reset the loss for downstream regularizers to accumulate onto
            self._reset_loss()

            return self.forward(X_T)


class Dense(Layer):
    """Just your standard fully connected layer.

    Args:
        units (int): Number of output units in the layer.
        activation (Op, optional): Activation function to apply
            to the output. Defaults to Linear.
        use_bias (bool, optional): Whether to use a bias. Defaults to True.
        activity_L2 (float, optional): L2 regularization coefficient
            for the activity. Default is None.
        weight_L2 (float, optional): L2 regularization coefficient
            for the weights. Default is None.
        bias_L2 (float, optional): L2 regularization coefficient
            for the bias. Default is None.

    Example:
        >>> layer = Dense(10, Relu, 0.1, 0.1, 0.1)
        >>> X_T = jnp.Var(np.random.uniform(0, 1, size=(3, 5)), trainable=False)
        >>> Y_T = layer(X_T)
        >>> X_T.val, Y_T.val, layer.loss, layer.trainable_variables
    """

    def __init__(
        self,
        units: int,
        activation: jnp.Op = None,
        use_bias: bool = True,
        activity_L2: float = None,
        weight_L2: float = None,
        bias_L2: float = None,
        name: str = "Dense",
    ):
        super().__init__(name=name)

        if activation is None:
            activation = jnp.Linear

        self.units = units
        self.activation = activation
        self.use_bias = use_bias

        self.activity_L2 = (
            jnp.Var(activity_L2, name="activity_L2")
            if activity_L2 is not None
            else None
        )
        self.weight_L2 = (
            jnp.Var(weight_L2, name="weight_L2") if weight_L2 is not None else None
        )
        self.bias_L2 = jnp.Var(bias_L2, name="bias_L2") if bias_L2 is not None else None

    @property
    def trainable_variables(self) -> List[jnp.T]:
        return [self.W_T] + ([self.B_T] if self.use_bias else [])

    def build(self, input_shape):
        W = np.random.uniform(low=-0.05, high=0.05, size=(input_shape[-1], self.units))
        self.W_T = jnp.Var(val=W, trainable=True, name="weights")
        if self.use_bias:
            B = np.random.uniform(low=-0.05, high=0.05, size=(1, self.units))
            self.B_T = jnp.Var(val=B, trainable=True, name="bias")

    def forward(self, X_T: jnp.T) -> jnp.T:

        # compute presynaptic input
        Z_T = X_T @ self.W_T

        # maybe add bias
        if self.use_bias:
            Z_T = Z_T + self.B_T

        # apply activation
        Y_T = self.activation(Z_T)

        # track regularization losses
        if self.activity_L2 is not None:
            self._loss += self.activity_L2 * jnp.ReduceSum(
                jnp.ReduceSum(Y_T ** 2, 1), 0
            )
        if self.weight_L2 is not None:
            self._loss += self.weight_L2 * jnp.ReduceSum(
                jnp.ReduceSum(self.W_T ** 2, 1), 0
            )
        if self.use_bias and self.bias_L2 is not None:
            self._loss += self.bias_L2 * jnp.ReduceSum(
                jnp.ReduceSum(self.B_T ** 2, 1), 0
            )

        return Y_T


class Conv2D(Layer):
    """Standard 2D Conv layer.
    I.E. convolves over Tensors shaped [B, H, W, D]
    to produce [B, H-2*kernel_size, W-2*kernel_size, filters]

    Args:
        filters (int): Number of filters to apply
        kernel_size (Union[int, Tuple[int, int]], optional):
            The size of the convolution kernel. Default is 3x3.
        strides (Union[int, Tuple[int, int]], optional):
            The stride of the convolution. Default is 1.
        padding (str, optional): The padding type. Can be
            'valid' or 'same'. Default is 'same'.
        activation (jnp.Op, optional): The activation function.
            Default is Linear.
        use_bias (bool, optional): Whether to use a bias. Default
            is False.
        activity_L2 (float, optional): L2 regularization coefficient
            for the activity. Default is None.
        weight_L2 (float, optional): L2 regularization coefficient
            for the weights. Default is None.
        bias_L2 (float, optional): L2 regularization coefficient
            for the bias. Default is None.

    Example:
        >>> layer = Conv2D(filters=64, kernel_size=5, strides=5, padding='same', weight_L2=0.1)
        >>> X_T = jnp.Var(np.random.uniform(0, 1, size=(2, 7, 70, 5)), trainable=False)
        >>> Y_T = layer(X_T)
        >>> X_T.shape, Y_T.shape, layer.loss, layer.trainable_variables
    """

    def __init__(
        self,
        filters: int,
        kernel_size: Union[int, Tuple[int, int]] = 3,
        strides: Union[int, Tuple[int, int]] = 1,
        padding: str = "valid",  # 'valid' or 'same'
        activation: jnp.Op = None,
        use_bias: bool = False,
        activity_L2: float = None,
        weight_L2: float = None,
        bias_L2: float = None,
        name: str = "Conv",
    ):
        super().__init__(name=name)

        if activation is None:
            activation = jnp.Linear
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        assert (
            kernel_size[0] % 2 == 1 and kernel_size[1] % 2 == 1
        ), "kernel_size must be odd"
        if isinstance(strides, int):
            strides = (strides, strides)
        assert strides[0] > 0 and strides[1] > 0, "strides must be positive"
        padding = padding.lower()
        assert padding in ("valid", "same"), "padding must be valid or same"

        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.activation = activation
        self.use_bias = use_bias

        self.activity_L2 = (
            jnp.Var(activity_L2, name="activity_L2")
            if activity_L2 is not None
            else None
        )
        self.weight_L2 = (
            jnp.Var(weight_L2, name="weight_L2") if weight_L2 is not None else None
        )
        self.bias_L2 = jnp.Var(bias_L2, name="bias_L2") if bias_L2 is not None else None

    @property
    def trainable_variables(self) -> List[jnp.T]:
        return [self.W_T] + ([self.B_T] if self.use_bias else [])

    def build(self, input_shape):
        W = np.random.uniform(
            low=-0.05,
            high=0.05,
            size=(
                self.kernel_size[0] * self.kernel_size[1] * input_shape[-1],
                self.filters,
            ),
        )
        self.W_T = jnp.Var(val=W, trainable=True, name="weights")

        if self.use_bias:
            B = np.random.uniform(low=-0.05, high=0.05, size=(1, self.filters))
            self.B_T = jnp.Var(val=B, trainable=True, name="bias")

    def forward(self, X_T: jnp.T) -> jnp.T:

        # maybe pad input
        if self.padding == "same":

            # various padding sizes, strides, and offsets
            # 0   1   2   3   4
            #     0 1 2 3 4
            #         0 1 2 3 4 5 6 7 8 9
            #         0 1 2 3 4
            #         0   1   2   3   4

            pad_top = self.strides[0] * (self.kernel_size[0] - 1) // 2
            pad_bottom = pad_top
            pad_left = self.strides[1] * (self.kernel_size[1] - 1) // 2
            pad_right = pad_left
            B, H_orig, W_orig, C = X_T.shape

            # pad height
            X_T = jnp.Concat(
                [
                    jnp.Var(np.zeros((B, pad_top, W_orig, C)), trainable=False),
                    X_T,
                    jnp.Var(np.zeros((B, pad_bottom, W_orig, C)), trainable=False),
                ],
                axis=1,
                name="PadHeight",
            )

            # pad width
            X_T = jnp.Concat(
                [
                    jnp.Var(
                        np.zeros((B, H_orig + pad_top + pad_bottom, pad_left, C)),
                        trainable=False,
                    ),
                    X_T,
                    jnp.Var(
                        np.zeros((B, H_orig + pad_top + pad_bottom, pad_right, C)),
                        trainable=False,
                    ),
                ],
                axis=2,
                name="PadWidth",
            )

        elif self.padding == "valid":
            pass

        # stack the input tensor along the channel axis
        # but shifted by all possible kernel shifts
        stack = []
        for shift in itertools.product(
            range(0, self.strides[0] * self.kernel_size[0], self.strides[0]),
            range(0, self.strides[1] * self.kernel_size[1], self.strides[1]),
        ):
            stack.append(X_T[:, shift[0] :, shift[1] :, :])

        # clip stack to greatest common shape
        min_shape = np.min(np.array([s.shape for s in stack]), axis=0)
        stack = [s[:, : min_shape[1], : min_shape[2], :] for s in stack]

        # stack the shifted tensors along the channel axis
        stacked = jnp.Concat(stack, axis=3)  # [B, H-k_h//2, W-k_w//2, C*k_h*k_w]

        # convolve over the stacked tensors
        Z_T = stacked @ self.W_T

        # maybe add bias
        if self.use_bias:
            Z_T = Z_T + self.B_T

        # apply the activation function
        Y_T = self.activation(Z_T)

        # track regularization losses
        if self.activity_L2 is not None:
            self._loss += self.activity_L2 * jnp.ReduceSum(
                jnp.ReduceSum(Y_T ** 2, 1), 0
            )
        if self.weight_L2 is not None:
            self._loss += self.weight_L2 * jnp.ReduceSum(
                jnp.ReduceSum(self.W_T ** 2, 1), 0
            )
        if self.use_bias and self.bias_L2 is not None:
            self._loss += self.bias_L2 * jnp.ReduceSum(
                jnp.ReduceSum(self.B_T ** 2, 1), 0
            )

        return Y_T


class GlobalMaxPooling(Layer):
    """Performs max pooling along an entire axis."""

    def __init__(self, axis: int, name: str = "GlobalMaxPooling"):
        super().__init__(name=name)
        self.axis = axis

    def forward(self, X_T: jnp.T) -> jnp.T:
        return jnp.ReduceMax(X_T, self.axis)


class Lambda(Layer):
    """Lambda layer.

    NOTE: Unlike keras' lambda layer, the lambda function here
    recieves the full tensor (batch axis included) and currently
    all transforms must be performed using jnumpy core ops."""

    def __init__(self, fn: Callable[[jnp.T], jnp.T], name: str = "Lambda"):
        super().__init__(name=name)
        self.fn = fn

    def forward(self, X_T: jnp.T) -> jnp.jnp.T:
        return self.fn(X_T)


class Flatten(Layer):
    """Flattens all non-batch dimensions into a single axis

    Example:
        >>> layer = Flatten()
        >>> X_T = jnp.Var(np.random.uniform(0, 1, size=(2, 3, 4, 5)))
        >>> Y_T = layer(X_T)
        >>> X_T.shape, Y_T.shape, layer.loss, layer.trainable_variables

    """

    def __init__(self, name: str = "Flatten"):
        super().__init__(name=name)

    @property
    def trainable_variables(self) -> List[jnp.T]:
        return []

    def forward(self, X_T: jnp.T) -> jnp.T:
        flat_dims = functools.reduce(lambda x, y: x * y, X_T.shape[1:])
        Y_T = jnp.Reshape(X_T, (X_T.shape[0], flat_dims))
        return Y_T


class Sequential(Layer):
    """A sequential layer is a list of layers that are applied sequentially.

    Example (dense network):
        >>> X_T = jnp.Var(np.random.uniform(0, 1, size=(2, 5)))
        >>> dense_net = Sequential([
                Dense(10, jnp.Relu),
                Dense(128, jnp.Relu),
                Dense(512, jnp.Tanh),
                Dense(1, jnp.Sigm)
            ])
        >>> dense_net(X_T)

    Example (convolutional network):
        >>> img_T = jnp.Var(np.random.uniform(0, 1, size=(1, 28, 28, 1)), trainable=False)
        >>> conv_net = Sequential([
                Conv2D(32, 3, 2, activation=jnp.Relu),
                Conv2D(64, 3, 2, activation=jnp.Relu),
                Flatten(),
                Dense(512, jnp.Sigm),
                Dense(1, jnp.Linear)
            ])
        >>> conv_net(img_T)
    """

    def __init__(self, layers, name="Sequential"):
        self.layers = layers
        name = jnp.NameScope.get_name("MODEL_NAMES", name)
        super().__init__(name=name)

    def forward(self, X_T: jnp.T) -> jnp.T:
        for layer in self.layers:
            X_T = layer(X_T)
        return X_T

    @property
    def loss(self) -> jnp.T:
        with jnp.NameScope(self.name):
            loss = functools.reduce(
                lambda x, y: x + y, [layer.loss for layer in self.layers]
            )
        return loss

    @property
    def trainable_variables(self) -> List[jnp.T]:
        trainable_vars = []
        for layer in self.layers:
            trainable_vars += layer.trainable_variables
        return trainable_vars
