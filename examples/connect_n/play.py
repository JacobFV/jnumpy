# Copyright (c) 2021 Jacob F. Valdez. Released under the MIT license.
"""Interactive Connect-N game."""

from __future__ import annotations

import datetime
import pickle
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

import jnumpy as jnp
import jnumpy.nn as jnn
import jnumpy.rl as jrl
import jnumpy.utils as jutils

from common import BoardEnv, MaxConnect4Agent, get_human_move
from maxconnect4.jnumpy.jnumpy.rl.agents import RandomAgent, RealDQN, CategoricalDQN

AGENT_TYPES = {
    "random: takes random actions at each step": RandomAgent,
    "max, maxconnect4: takes greedy action with n step lookahead": MaxConnect4Agent,
    "dqn: standard deep Q-network; a* = arg_a max dqn(o, a)": RealDQN,
    "categorical dqn: DQN with categorical action space; <q0,q1,..,qn> = dqn(o); a* = arg_i max qi": CategoricalDQN,
}

default_hparams = dict(
    hidden_size=64,  # hidden layer size for RealDQN
    categorical_hidden_size=32,  # hidden layer size for CategoricalDQN
    activation="Relu",  # activation function for networks
    optimizer=None,  # optimizer for networks (filled in later)
    learning_rate=0.001,  # learning rate for optimizer
    epsilon_start=1.0,  # Starting value for epsilon
    epsilon_decay=0.9,  # Decay rate for epsilon per epoch
    min_epsilon=0.01,  # Final value for epsilon
    discount=0.9,  # Discount factor
    epoch=0,  # Current epoch (different for each agent)
    batch_size=8,  # Number of samples per training batch
    train_batch_size=32,  # Batch size for training
    board_size=10,  # Board size
    train_win_length=4,  # Number of pieces in a row needed to win in training
    test_win_length=5,  # Number of pieces in a row needed to win in testing
    min_steps_per_epoch=64,  # Minimum number of steps per epoch
    num_steps_replay_coef=0.5,  # How much to upweight longer episodes
    success_replay_coef=1.5,  # How much to upweight successful experience
    age_replay_coef=0.5,  # How much to downweight older trajectories
)


encoder_pool = list()  # [Encoder] encoders can be shared across agents
agent_pool = list()  # [Agent], not shared by definition
hparam_pool = dict()  # {name: hparams}  not shared. Updated in `Trainer.train`

simple_encoder = jnn.Sequential(
    [
        jnn.Conv2D(8, 3, 1, "same", jnp.Relu),
        jnn.Conv2D(16, 3, 1, "same", jnp.Relu),
        jnn.GlobalMaxPooling(1),
    ],
    name='simple_encoder',
)  # [B, H, W, 2] -> [B, W, d_enc]
encoder_pool.append(simple_encoder)


def new_encoder() -> jnn.Sequential:
    print("\nNew encoder:")
    layers = []
    num_layers = jutils.get_input(
        "Enter number of convolutional layers: ", int, default_val=2)
    for i in range(num_layers):
        print(f"\nConv Layer {i}:")
        filters = jutils.get_int("Enter number of filters", default_val=8)
        kernel_size = jutils.get_input(
            "Enter kernel size (odd, int or 2-tuple): ", eval, default_val="3")
        stride = jutils.get_input(
            "Enter stride (int or 2-tuple): ", eval, default_val="1")
        activation = jutils.select_option(
            'Enter activation function', jnn.Activations)
        layers.append(jnn.Conv2D(filters, kernel_size,
                      stride, "same", activation))
    layers.append(jnn.GlobalMaxPooling(1))
    encoder_name_candidate = input(
        "\nPlease provide a name for this encoder: ").strip()
    encoder = jnn.Sequential(layers, name=encoder_name_candidate)
    encoder_pool.append(encoder)
    return encoder


def get_encoder() -> jnn.Sequential:
    options = {
        f'{encoder.name}: conv_layers={len(encoder.layers)-1}': encoder
        for encoder in encoder_pool
    }
    NEW_ENCODER = 0
    options['new'] = NEW_ENCODER
    encoder = jutils.select_option(
        "Select an encoder", options, default_val=simple_encoder)
    if encoder == NEW_ENCODER:
        return new_encoder()
    else:
        return encoder


def get_hparams(agent_name: str) -> dict:
    print(f"\nHyperameters for {agent_name}:")
    hparams = dict()
    hparams.update(default_hparams)
    for key, value in default_hparams.items():
        for type in [int, float, str]:
            if isinstance(value, type):
                hparams[key] = jutils.get_input(
                    f"Enter {key}", type, default_val=value)
    # these are special hyperparameters (not int, float, or str)
    hparams["optimizer"] = jnp.SGD(hparams["learning_rate"])
    hparams["activation"] = jutils.select_option(
        'Enter activation function', jnn.Activations,
        default_val=hparams["activation"])
    return hparams


def new_agent() -> jrl.Agent:
    print()

    agent_type = jutils.select_option("Select an agent type", AGENT_TYPES)
    agent_name_candidate = input("Please name this agent: ").strip().lower()
    num_actions = jutils.get_int("Enter num actions (int): ")
    if agent_type == RandomAgent:
        agent = jrl.agents.RandomAgent(
            num_actions=num_actions, name=agent_name_candidate
        )
        hparam_pool[agent.name] = dict(
            epoch=0,  # Current epoch (not used for random agent)
            min_steps_per_epoch=64,  # Minimum number of steps per epoch
            num_steps_replay_coef=0.5,  # How much to upweight longer episodes
            success_replay_coef=1.5,  # How much to upweight successful experience
            age_replay_coef=0.5,  # How much to downweight older trajectories
        )
    elif agent_type == MaxConnect4Agent:
        look_ahead = jutils.get_int(
            "Enter lookahead depth (int, default=1): ", default_val=1)
        agent = MaxConnect4Agent(num_actions, look_ahead)
        hparam_pool[agent.name] = dict(
            epoch=0,  # Current epoch (not used for random agent)
            min_steps_per_epoch=64,  # Minimum number of steps per epoch
            num_steps_replay_coef=0.5,  # How much to upweight longer episodes
            success_replay_coef=1.5,  # How much to upweight successful experience
            age_replay_coef=0.5,  # How much to downweight older trajectories
        )
    elif agent_type == RealDQN:
        encoder = get_encoder()
        hparams = get_hparams(agent_name=agent_name_candidate)
        agent = jrl.agents.RealDQN(
            num_actions=num_actions,
            encoder=encoder,
            hparams=hparams,
            name=agent_name_candidate,
        )
        hparam_pool[agent.name] = hparams
    elif agent_type == CategoricalDQN:
        encoder = get_encoder()
        hparams = get_hparams(agent_name=agent_name_candidate)
        agent = jrl.agents.CategoricalDQN(
            num_actions=num_actions,
            encoder=encoder,
            hparams=hparams,
            name=agent_name_candidate,
        )
        hparam_pool[agent.name] = hparams
    agent_pool.append(agent)
    return agent


def get_agent() -> jrl.Agent:
    options = {
        f" - {agent.name}: type={type(agent).__name__} num_actions={agent.num_actions} epoch={hparam_pool[agent.name]['epoch']}":
        agent for agent in agent_pool
    }
    NEW_AGENT = 0
    options['new'] = NEW_AGENT
    agent = jutils.select_option(
        "Select an agent", options, default_val=NEW_AGENT)
    if agent == NEW_AGENT:
        return new_agent()
    else:
        return agent


def get_env(env_type=None) -> Tuple[str, dict]:
    print()
    if env_type:
        print(f"{env_type}:")

    rows = jutils.get_int(
        "Enter number of board rows (int, default=6): ", default_val=6)
    cols = jutils.get_int(
        "Enter number of board columns (int, default=7): ", default_val=7)
    win_length = jutils.get_int("Enter win length (int, default=4): ", default_val=4))
    reward_mode=jutils.select_option(
        "Select reward mode", BoardEnv.reward_modes, default_val = 'simple')
    batch_size=jutils.get_int(
        "Enter environment batch size (int, default=4)", default_val = 4)

    return jrl.ParallelEnv(
        batch_size = batch_size,
        env_init_fn = lambda: BoardEnv(
            initial_board_size=(rows, cols),
            win_length=win_length,
            reward_mode=reward_mode,
        ),
    )


def train():

    train_env=get_env("Training environment")
    test_env=get_env("Test environment")

    while train_env.env.board_size != test_env.env.board_size:
        print(
            f"Board sizes must match: {train_env.env.board_size} vs {test_env.env.board_size}"
        )
        print(
            f"Please choose a test environment with a board size of {train_env.env.board_size}"
        )
        test_env = get_env("Test environment")

    agent1 = get_agent()
    agent2 = get_agent()

    while agent1.num_actions != agent2.num_actions:
        print(
            "Agents must have the same number of actions: "
            f"{agent1.name} has {agent1.num_actions} actions but "
            f"{agent2.name} has {agent2.num_actions} actions."
        )
        agent1 = get_agent()
        agent2 = get_agent()

    trainer = jrl.ParallelTrainer(
        callbacks=[
            jrl.QEvalCallback(
                eval_on_collect=True, eval_on_train=True, eval_on_test=True
            ),
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
    global hparam_pool
    training_epochs = jutils.get_int(
        "\nEnter number of training epochs (int, default=10)", default_val=10)
    train_start_time = datetime.datetime.now()
    print(f"\nTraining {agent1.name} with {agent2.name}...")
    hparam_pool = trainer.train(
        agents=[agent1, agent2],
        all_hparams=hparam_pool,
        env=train_env,
        test_env=test_env,
        training_epochs=training_epochs,
    )
    train_finish_time = datetime.datetime.now()
    print(
        f"Training finished. Elapsed time = {train_finish_time-train_start_time}")


def play():

    human_name = input("Enter your name: ").strip()
    agent = get_agent()
    first_play = jutils.select_option(
        "Who goes first?", [human_name, agent.name], default_val=human_name)

    rows = jutils.get_int(
        "Enter number of board rows (int, default=6): ", default_val=6)
    cols = jutils.get_int(
        "Enter number of board columns (int, default=7): ", default_val=7)
    win_length = jutils.get_int("Enter win length (int, default=4): ", default_val=4))
    reward_mode=jutils.select_option(
        "Select reward mode", BoardEnv.reward_modes, default_val = 'simple')
    env = BoardEnv(
        initial_board_size=(rows,cols),
        win_length=win_length,
        reward_mode=reward_mode,
    )
    batch_env = jrl.Batch1Env(env)  # to handle executing single actions

    step = batch_env.reset()
    if first_play == human_name:
        env.render()
        action = get_human_move(env.cols)
        step = batch_env.step(action)
    while True:
        action = agent.forward(step)
        step = batch_env.step(action)
        if step.done:
            winner = agent.name if step.reward > 0 else human_name
            break
        env.render()
        action = get_human_move(env.cols)
        step = batch_env.step(action)
        if step.done:
            winner = human_name if step.reward > 0 else agent.name
            break

    # announce winner
    print(f"{winner} wins!\n")

    # play again?
    if input("Play again? (y/N):").strip().lower() == "y":
        play()


def main_menu():
    print("1. Train")
    print("2. Play")
    print("3. Exit")
    print("\n")

    choice = jutils.get_input("Enter choice (1, 2, or 3): ",
                              lambda x: int(x),
                              lambda x: int(x) in [1, 2, 3])
    if choice == "1":
        train()
    elif choice == "2":
        play()
    elif choice == "3":
        exit()

    _ = input("\nPress enter to continue...")


def main():
    print("Connect 4 with Reinforcement Learning")
    print("Copyright 2022 Jacob Valdez. Released under the MIT License")
    print("Code at https://github.com/JacobFV/jnumpy")
    print()
    while True:
        try:
            main_menu()
        except Exception as e:
            print(e)
            print("\n")
            print("An error occurred. Please try again.")
            print(
                "Please report issues at https://github.com/JacobFV/jnumpy/issues/new"
            )
            print("\n")
            _ = input("\nPress enter to continue...")

        print(128 * "\n")


if __name__ == "__main__":
    main()
