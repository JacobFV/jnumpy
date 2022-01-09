![](https://github.com/JacobFV/jnumpy/raw/main/content/logo.png)

[![PyPI version](https://badge.fury.io/py/jnumpy.svg)](https://badge.fury.io/py/jnumpy)
[![](https://img.shields.io/badge/license-MIT-blue)](https://github.com/JacobFV/jnumpy/blob/main/LICENSE)
[![Codacy Badge](https://api.codacy.com/project/badge/Grade/e1cb295484424f36acf2c813fae6f57e)](https://app.codacy.com/gh/JacobFV/jnumpy?utm_source=github.com&utm_medium=referral&utm_content=JacobFV/jnumpy&utm_campaign=Badge_Grade_Settings)

Jacob's numpy library for machine learning

## Getting started

1. Install from `pip` or clone locally:

```bash
$ pip install jnumpy
# or
$ git clone https://github.com/JacobFV/jnumpy.git
$ cd jnumpy
$ pip install .
```

2. Import the `jnumpy` module.

```python
import jnumpy as jnp
```

## Examples

### Low-level stuff

```python
import jnumpy as jnp

W = jnp.Var(np.random.randn(5, 3), trainable=True, name='W')
b_const = jnp.Var(np.array([1., 2., 3.]), name='b')  # trainable=False by default

def model(x):
    return x @ W + b_const

def loss(y, y_pred):
    loss = (y - y_pred)**2
    loss = jnp.ReduceSum(loss, axis=1)
    loss = jnp.ReduceSum(loss, axis=0)
    return loss

opt = jnp.SGD(0.01)

for _ in range(10):
    # make up some data
    x = jnp.Var(np.random.randn(100, 5))
    y = jnp.Var(np.random.randn(100, 3))

    # forward pass
    y_pred = model(x)
    loss_val = loss(y, y_pred)

    # backpropagation
    opt.minimize(loss)
```

### Neural networks

```python
import jnumpy as jnp
import jnumpy.nn as jnn

conv_net = jnn.Sequential(
    [
        jnn.Conv2D(32, 3, 2, activation=jnp.Relu),
        jnn.Conv2D(64, 3, 2, activation=jnp.Relu),
        jnn.Flatten(),
        jnn.Dense(512, jnp.Sigm),
        jnn.Dense(1, jnp.Linear),
    ]
)
```

### Reinforcement learning

```python
import jnumpy as jnp
import jnumpy.rl as jrl

shared_encoder = conv_net  # same archiecture as the conv_net above

# agents
agentA_hparams = {...}
agentB_hparams = {...}
agentC_hparams = {...}

# categorical deep Q-network:
#   <q0,q1,..,qn> = dqn(o)
#   a* = arg_i max qi
agentA = jrl.agents.CategoricalDQN(
    num_actions=agentA_hparams['num_actions'],
    encoder=shared_encoder,
    hparams=agentA_hparams,
    name='agentA'
    )

# standard deep Q-network:
#   a* = arg_a max dqn(o, a)
agentB = jrl.agents.RealDQN(
    num_actions=agentB_hparams['num_actions'],
    encoder=shared_encoder,
    hparams=agentB_hparams,
    name='agentB'
    )

# random agent:
#   pick a random action
agentC = jrl.agents.RandomAgent(agentC_hparams['num_actions'], name='agentC')

# init enviroments
train_env = jrl.ParallelEnv(
    batch_size=32,
    env_init_fn=lambda: MyEnv(...),  # `jrl.Environment` subclass. Must have `reset` and `step` methods.
)
dev_env = jrl.ParallelEnv(
    batch_size=8,
    env_init_fn=lambda: MyEnv(...),
)
test_env = jrl.ParallelEnv(
    batch_size=8,
    env_init_fn=lambda: MyEnv(...),
)

# train
trainer = jrl.ParallelTrainer(callbacks=[
    jrl.PrintCallback(['epoch', 'agent', 'collect_reward', 'q_train', 'q_test']),
    jrl.QEvalCallback(eval_on_train=True, eval_on_test=True),
])
trainer.train(
    agents={'agentA': agentA, 'agentB': agentB},
    all_hparams={'agentA': agentA_hparams, 'agentB': agentB_hparams},
    env=train_env,
    test_env=dev_env,
    training_epochs=10,
)

# test
driver = ParallelDriver()
trajs = driver.drive(
    agents={'agentA': agentA, 'agentB': agentB},
    env=test_env
)
per_agent_rewards = {
    agent_name: sum(step.reward for step in traj)
    for agent_name, traj in trajs.items()}
print('cumulative test rewards:', per_agent_rewards)
```

## Limitations and Future Work

Future versions will feature:

- add `fit`, `evaluate`, and `predict` to `jnp.Sequential`
- recurrent network layers
- static execution graphs allowing breadth-first graph traversal
- more optimizers, metrics, and losses
- io loaders for csv's, images, and models (maybe also for graphs)
- more examples

Also maybe for the future:

- custom backends (i.e.: tensorflow or pytorch instead of numpy)

## License

All code in this repository is licensed under the MIT license. No restrictions, but no warranties. See the [LICENSE](https://github.com/JacobFV/jnumpy/blob/main/LICENSE) file for details.

## Contributing

This is a small project, and I don't plan on growing it much. You are welcome to fork and contribute or email me `jacob` [dot] `valdez` [at] `limboid` [dot] `ai` if you would like to take over. You can add your name to the copyright if you make a PR or your own branch.

The codebase is kept in only a few files, and I have tried to minimize the use of module prefixes because my CSE 4308/4309/4392 classes require the submissions to be stitched togethor in a single file.
