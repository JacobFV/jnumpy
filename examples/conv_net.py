import numpy as np

import jnumpy as jnp
import jnumpy.nn as jnn


img_T = jnp.Var(np.random.uniform(0, 1, size=(1, 28, 28, 1)), trainable=False)
conv_net = jnn.Sequential(
    [
        jnn.Conv2D(32, 3, 2, activation=jnp.Relu),
        jnn.Conv2D(64, 3, 2, activation=jnp.Relu),
        jnn.Flatten(),
        jnn.Dense(512, jnp.Sigm),
        jnn.Dense(1, jnp.Linear),
    ]
)
y_T = conv_net(img_T)

print(f"{img_T.shape} --conv_net--> {y_T.shape}")
