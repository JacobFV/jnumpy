# Copyright (c) 2021 Jacob F. Valdez. Released under the MIT license.
"""Experiment code."""

import random
import itertools
from copy import deepcopy

import jnumpy as jnp
import jnumpy.nn as jnn
import jnumpy.rl as jrl

from common import BoardEnv


hparams = dict(
    hidden_size=64,  # hidden layer size for RealDQN
    categorical_hidden_size=32,  # hidden layer size for CategoricalDQN
    activation=jnp.Relu,  # activation function for networks
    optimizer=jnp.SGD(0.001),  # optimizer for networks
    epsilon_start=1.0,  # Starting value for epsilon
    epsilon_decay=0.9,  # Decay rate for epsilon per epoch
    min_epsilon=0.01,  # Final value for epsilon
    discount=0.9,  # Discount factor
    epoch=0,  # Current epoch (different for each agent)
    batch_size=2,  # Number of samples per training batch
    train_batch_size=32,  # Batch size for training
    board_size=10,  # Board size
    train_win_length=4,  # Number of pieces in a row needed to win in training
    test_win_length=5,  # Number of pieces in a row needed to win in testing
    min_steps_per_epoch=2,  # Minimum number of steps per epoch
    num_steps_replay_coef=0.5,  # How much to upweight longer episodes
    success_replay_coef=1.5,  # How much to upweight successful experience
    age_replay_coef=0.5,  # How much to downweight older trajectories
)

encoder = jnn.Sequential(
    [
        jnn.Conv2D(8, 3, 1, "same", jnp.Relu),
        jnn.Conv2D(16, 3, 1, "same", jnp.Relu),
        jnn.GlobalMaxPooling(1),
    ],
    name="Encoder",
)  # [B, H, W, 2] -> [B, W, d_enc]

agents = [
    jrl.agents.RandomAgent(hparams["board_size"], name="Ally"),
    jrl.agents.RealDQN(hparams["board_size"], encoder,
                       deepcopy(hparams), name="Bob"),
    jrl.agents.RealDQN(hparams["board_size"], encoder,
                       deepcopy(hparams), name="Cara"),
    jrl.agents.CategoricalDQN(
        hparams["board_size"], encoder, deepcopy(hparams), name="Dan"
    ),
    jrl.agents.CategoricalDQN(
        hparams["board_size"], encoder, deepcopy(hparams), name="Emma"
    ),
]
all_hparams = {
    agent.name: agent.hparams if hasattr(
        agent, "hparams") else deepcopy(hparams)
    for agent in agents
}

train_env = jrl.ParallelEnv(
    hparams["batch_size"],
    lambda: BoardEnv(
        initial_board_size=hparams["board_size"],
        win_length=hparams["train_win_length"],
        reward_mode="dense_stateless",
    ),
)

test_env = jrl.ParallelEnv(
    hparams["batch_size"],
    lambda: BoardEnv(
        initial_board_size=hparams["board_size"],
        win_length=hparams["test_win_length"],
        reward_mode="dense_stateless",
    ),
)

trainer = jrl.ParallelTrainer(
    callbacks=[
        jrl.QEvalCallback(eval_on_collect=True,
                          eval_on_train=True, eval_on_test=True),
        jrl.PrintCallback(
            keys=[
                "epoch",
                "agent",
                "collect_reward",
                "train_reward",
                "test_reward",
                "q_collect",
                "q_train",
                "q_test",
            ],
        ),
    ],
)

for i in range(100):
    agents_iter_list = list(itertools.combinations(agents, 2))
    random.shuffle(agents_iter_list)
    for agentA, agentB in agents_iter_list:
        all_hparams = trainer.train(
            agents=[agentA, agentB],
            all_hparams=all_hparams,
            env=train_env,
            test_env=test_env,
            training_epochs=1,
        )
