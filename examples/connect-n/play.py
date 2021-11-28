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

from common import BoardEnv

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

encoder_pool.append(
    jnn.Sequential(
        [
            jnn.Conv2D(8, 3, 1, "same", jnp.Relu),
            jnn.Conv2D(16, 3, 1, "same", jnp.Relu),
            jnn.GlobalMaxPooling(1),
        ],
        name="simple_encoder",
    )
)  # [B, H, W, 2] -> [B, W, d_enc]


def maybe_input(prompt: str, default: str) -> str:
    answer = input(f"{prompt} (default: {default}): ").strip()
    if answer == "":
        return default
    return answer


def get_activation_fn(name: str) -> jnn.Activation:
    if name == "Relu":
        return jnp.Relu
    elif name == "Sigm":
        return jnp.Sigm
    elif name == "Tanh":
        return jnp.Tanh
    elif name == "Linear":
        return jnp.Linear
    else:
        raise ValueError(f"Invalid activation: {name}")


def new_encoder() -> jnn.Sequential:
    layers = []
    print("\n")
    num_layers = int(maybe_input("Enter number of conv layers", 2))
    for _ in range(num_layers):
        filters = int(maybe_input("Enter number of filters", 8))
        kernel_size = eval(
            input("Enter kernel size (odd, int or 2-tuple; default: 3): ").strip()
        )
        stride = eval(input("Enter stride (int|2-tuple; default: 1): ").strip())
        activation = (
            input("Enter activation function (`Relu`, `Sigm`, `Tanh`, `Linear`): ")
            .strip()
            .lower()
        )
        activation = get_activation_fn(activation)
        layers.append(jnn.Conv2D(filters, kernel_size, stride, "same", activation))
    layers.append(jnn.GlobalMaxPooling(1))
    encoder_name_candidate = input("Enter encoder name: ").strip()
    encoder = jnn.Sequential(layers, name=encoder_name_candidate)
    encoder_pool.append(encoder)
    return encoder


def get_encoder() -> jnn.Sequential:
    print("\n")
    print("All encoders: " + ", ".join([enc.name for enc in encoder_pool]))
    print("\n")
    encoder_name = input('Enter encoder name (or enter "new"): ').strip().lower()
    if encoder_name == "new":
        encoder = new_encoder()
    else:
        encoder = [enc for enc in encoder_pool if enc.name == encoder_name][0]
    return encoder


def new_hparams() -> dict:
    hparams = dict()
    hparams.update(default_hparams)
    print("\n")
    for key, value in default_hparams.items():
        if isinstance(value, float):
            hparams[key] = float(maybe_input(f"Enter {key}", value))
        elif isinstance(value, int):
            hparams[key] = int(maybe_input(f"Enter {key}", value))
        elif isinstance(value, str):
            hparams[key] = maybe_input(f"Enter {key}", value)
        else:
            continue
    hparams["optimizer"] = jnp.SGD(hparams["learning_rate"])
    hparams["activation"] = get_activation_fn(hparams["activation"])
    return hparams


def new_agent() -> jrl.Agent:
    print("\n")
    agent_type = (
        input(
            "Enter agent type (Random, RealDQN, or CategoricalDQN; default: 'RealDQN'): "
        )
        .strip()
        .lower()
    )
    agent_name_candidate = input("Please name this agent: ").strip().lower()
    if agent_type == "random":
        num_actions = int(input("Enter num_actions (int): ").strip())
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
    elif agent_type == "realdqn" or agent_type == "dqn":
        num_actions = int(input("Enter num_actions (int): ").strip())
        encoder = get_encoder()
        hparams = new_hparams()
        agent = jrl.agents.RealDQN(
            num_actions=num_actions,
            encoder=encoder,
            hparams=hparams,
            name=agent_name_candidate,
        )
        hparam_pool[agent.name] = hparams
    elif agent_type == "categoricaldqn":
        num_actions = int(input("Enter num_actions (int): ").strip())
        encoder = get_encoder()
        hparams = new_hparams()
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
    print("\n")
    print("All agents:")
    for agent in agent_pool:
        print(
            f" - {agent.name}: type={type(agent).__name__} num_actions={agent.num_actions} epoch={hparam_pool[agent.name]['epoch']}"
        )
    agent_name = input('Enter agent name (or enter "new"): ').strip().lower()
    if agent_name == "new":
        return new_agent()
    else:
        return [agent for agent in agent_pool if agent.name == agent_name][0]


def get_env(env_type) -> Tuple[str, dict]:
    print(f"\n{env_type}:")

    board_size = int(maybe_input("Enter board size (int)", 8))
    win_length = int(maybe_input("Enter win length (int)", 4))
    reward_mode = maybe_input(
        "Enter reward mode ('sparse', 'dense_stateless', or 'dense_advantage')",
        "sparse",
    ).lower()
    batch_size = int(maybe_input("Enter environment batch size (int)", 4))

    return jrl.ParallelEnv(
        batch_size=batch_size,
        env_init_fn=lambda: BoardEnv(
            board_size=board_size,
            win_length=win_length,
            reward_mode=reward_mode,
        ),
    )


def train():

    train_env = get_env("Training environment")
    test_env = get_env("Test environment")

    agent1 = get_agent()
    agent2 = get_agent()

    if agent1.num_actions != agent2.num_actions:
        raise ValueError(
            "Agents must have the same number of actions: "
            f"{agent1.name} has {agent1.num_actions}, "
            f"{agent2.name} has {agent2.num_actions}."
        )

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
    training_epochs = int(maybe_input("Enter number of training epochs (int)", 10))
    train_start_time = datetime.datetime.now()
    print(f"Training {agent1.name} with {agent2.name}...")
    hparam_pool = trainer.train(
        agents=[agent1, agent2],
        all_hparams=hparam_pool,
        env=train_env,
        test_env=test_env,
        training_epochs=training_epochs,
    )
    train_finish_time = datetime.datetime.now()
    print(f"Training finished. Elapsed time = {train_finish_time-train_start_time}")


def play():
    def is_human_first(human_name, agent_name) -> bool:
        first_agent = input(f"Who goes first ({human_name}, {agent_name})? ").strip()
        if first_agent == human_name:
            return True
        elif first_agent == agent_name:
            return False
        else:
            print(f"{first_agent} is not a valid option.")
            return is_human_first(human_name, agent_name)

    human_name = input("Enter your name: ").strip()
    agent = get_agent()
    human_first = is_human_first(human_name, agent.name)

    board_size = int(maybe_input("Enter board size (int)", 8))
    win_length = int(maybe_input("Enter win length (int)", 4))
    reward_mode = maybe_input(
        "Enter reward mode ('sparse', 'dense_stateless', or 'dense_advantage')",
        "sparse",
    ).lower()
    env = BoardEnv(
        board_size=board_size,
        win_length=win_length,
        reward_mode=reward_mode,
    )
    batch_env = jrl.Batch1Env(env)  # to handle executing single actions

    step = env.reset()
    if human_first:
        env.render()
        step.action = int(input(f"Enter move (0-{env.board_size}): ").strip())
        step = batch_env.step(step)
    while True:
        step.action = agent.act(step)
        step = batch_env.step(step)
        if step.done:
            winner = agent.name if step.reward > 0 else human_name
            break
        env.render()
        step.action = int(input(f"Enter move (0-{env.board_size}): ").strip())
        step = batch_env.step(step)
        if step.done:
            winner = human_name if step.reward > 0 else agent.name
            break

    # announce winner
    print(f"{winner} wins!\n")

    # play again?
    if input("Play again? (y/N):").strip().lower() == "y":
        play()


def main_menu():
    print(128 * "\n")
    print("1. Train")
    print("2. Play")
    print("3. Exit")
    print("\n")

    choice = input("Enter your choice: ")
    if choice == "1":
        train()
    elif choice == "2":
        play()
    elif choice == "3":
        exit()
    else:
        print("Invalid choice")

    _ = input("\nPress enter to continue...")


def main():
    print("CSE 4309 Machine Learning Project 7++")
    print("Copyright 2021 Jacob Valdez. Released under the MIT License")
    print("Code at https://github.com/JacobFV/jnumpy")
    print("\n")
    while True:
        main_menu()
        continue

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


main()
