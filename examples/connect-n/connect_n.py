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

from .board_env import BoardEnv
from .board_env import hparams as HPARAMS


encoder_pool = dict()
agent_pool = dict()
env_pool = dict()

jnn.Sequential(
    [
        jnn.Conv2D(32, 3, 2, "same", jnp.Relu),
        jnn.Conv2D(64, 3, 2, "same", jnp.Relu),
        jnn.AxisMaxPooling(1),
    ]
)


def new_encoder() -> jnn.Sequential:
    layers = []
    print("\n")
    num_layers = int(input("Enter number of conv layers (int): ").strip())
    for _ in range(num_layers):
        filters = eval(input("Enter number of filters (int): ").strip())
        kernel_size = eval(input("Enter kernel size (int|2-tuple, odd): ").strip())
        stride = eval(input("Enter stride (int|2-tuple): ").strip())
        activation = input(
            "Enter activation function (`Relu`, `Sigm`, `Tanh`, `Linear`): "
        ).strip()
        if activation == "Relu":
            activation = jnp.Relu
        elif activation == "Sigm":
            activation = jnp.Sigm
        elif activation == "Tanh":
            activation = jnp.Tanh
        elif activation == "Linear":
            activation = jnp.Linear
        else:
            raise ValueError(f"Invalid activation: {activation}")
        layers.append(jnn.Conv2D(filters, kernel_size, stride, "same", activation))
    layers.append(jnn.AxisMaxPooling(1))
    encoder = jnn.Sequential(layers)

    encoder_name = input("Enter encoder name: ").strip()
    encoder_pool[encoder_name] = encoder
    return encoder


def get_encoder() -> jnn.Sequential:
    print("\n")
    print("All encoders:")
    for name, encoder in encoder_pool.items():
        print(f"{name}: {encoder}")
    print("\n")
    encoder_name = input('Enter encoder name (or enter "new"): ').strip().lower()
    if encoder_name == "new":
        encoder = new_encoder()
    else:
        encoder = encoder_pool[encoder_name]
    return encoder


def new_agent() -> Tuple[str, jrl.Agent]:
    print("\n")
    agent_type = (
        input("Enter agent type (Random, RealDQN, or CategoricalDQN): ").strip().lower()
    )
    if agent_type == "Random":
        num_actions = eval(input("Enter num_actions (int): ").strip())
        agent = jrl.agents.RandomAgent(num_actions=num_actions)
    elif agent_type == "RealDQN":
        num_actions = eval(input("Enter num_actions (int): ").strip())
        _, encoder = get_encoder()
        agent = jrl.agents.RealDQN(
            num_actions=num_actions, encoder=encoder, hparams=HPARAMS
        )
    elif agent_type == "CategoricalDQN":
        num_actions = eval(input("Enter num_actions (int): ").strip())
        _, encoder = get_encoder()
        agent = jrl.agents.CategoricalDQN(
            num_actions=num_actions, encoder=encoder, hparams=HPARAMS
        )

    agent_name = input("Please name this agent: ").strip().lower()
    agent_pool[agent_name] = agent
    return agent_name, agent


def get_agent() -> Tuple[str, jrl.Agent]:
    print("\n")
    print("All agents:")
    for name, agent in agent_pool.items():
        print(f"{name}: {agent} (made for board size = {agent.num_actions})")
    print("\n")
    agent_name = input('Enter agent name (or enter "new"): ').strip().lower()
    if agent_name == "new":
        agent_name, agent = new_agent()
    else:
        agent = agent_pool[agent_name]
    return agent_name, agent


def new_env() -> Tuple[str, dict]:
    print("\n")

    board_size = int(input("Enter board size (int): ").strip())
    win_length = int(input("Enter train win length (int): ").strip())
    reward_mode = (
        input("Enter reward mode (`sparse`, `dense_stateless`, or `dense_advantage`): ")
        .strip()
        .lower()
    )
    batch_size = int(input("Enter number of batches (int): ").strip())

    psuedo_env = dict(
        board_size=board_size,
        win_length=win_length,
        reward_mode=reward_mode,
        batch_size=batch_size,
    )

    env_name = input("Please name this environment: ").strip().lower()
    env_pool[env_name] = psuedo_env
    return env_name, psuedo_env


def get_env() -> Tuple[str, dict]:
    print("\n")
    print("All environments:")
    for name, env in env_pool.items():
        print(f"{name}: {env}")  # list psuedo_env attributes
    print("\n")
    env_name = input('Enter environment name (or enter "new"): ').strip().lower()
    if env_name == "new":
        env_name, env = new_env()
    else:
        env = env_pool[env_name]
    return env_name, env


def train():

    train_env_name, train_env_dict = get_env()
    test_env_name, test_env_dict = get_env()

    if train_env_dict["board_size"] != test_env_dict["board_size"]:
        raise ValueError("Train and test environments must have the same board size.")

    # convert psuedo env's to full parallel env
    train_env = jrl.ParallelEnv(
        batch_size=train_env_dict["batch_size"],
        env_init_fn=lambda: BoardEnv(
            board=train_env_dict["board_size"],
            win_length=train_env_dict["win_length"],
            reward_mode=train_env_dict["reward_mode"],
        ),
    )
    test_env = jrl.ParallelEnv(
        batch_size=test_env_dict["batch_size"],
        env_init_fn=lambda: BoardEnv(
            board=test_env_dict["board_size"],
            win_length=test_env_dict["win_length"],
            reward_mode=test_env_dict["reward_mode"],
        ),
    )

    agent1_name, agent1 = get_agent()
    agent2_name, agent2 = get_agent()

    if agent1.num_actions != agent2.num_actions:
        raise ValueError(
            "Agents must have the same number of actions: "
            "{agent1_name} has {agent1.num_actions}, "
            "{agent2_name} has {agent2.num_actions}."
        )

    trainer = jrl.ParallelTrainer(
        hparams=HPARAMS,
        callbacks=[
            jrl.PrintCallback(
                hparams=HPARAMS, print_hparam_keys=["epoch"], print_data_keys=["reward"]
            ),
            jrl.QEvalCallback(
                eval_on_collect=True, eval_on_train=True, eval_on_test=True
            ),
        ],
    )

    train_start_time = datetime.datetime.now()
    print(
        f"Training {agent1_name} against {agent2_name} with train_env={train_env_name} test_env={test_env_name}..."
    )
    history = trainer.train(agent1, agent2, train_env, test_env)
    train_finish_time = datetime.datetime.now()
    print(f"Training finished. Elapsed time = {train_finish_time-train_start_time}")
    print(history)

    print("\n")
    if input("Save history? (y/N):").strip().lower() == "y":
        fname = input("Enter filename: ").strip()
        with open(fname, "wb") as f:
            pickle.dump(history, f)
        print(f"Saved pickle dump to {fname}")


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
    agent_name, agent = get_agent()
    human_first = is_human_first(human_name, agent_name)

    _, env_dict = get_env()
    env = BoardEnv(
        board=env_dict["board_size"],
        win_length=env_dict["win_length"],
        reward_mode="sparse",
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
            winner = agent_name if step.reward > 0 else human_name
            break
        env.render()
        step.action = int(input(f"Enter move (0-{env.board_size}): ").strip())
        step = batch_env.step(step)
        if step.done:
            winner = human_name if step.reward > 0 else agent_name
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
    print("Code at https://github.com/JacobFV/assignment-rl")
    print("\n")
    while True:
        try:
            main_menu()
        except Exception as e:
            print(e)
            print("\n")
            print("An error occurred. Please try again.")
            print(
                "Please report issues on github at https://github.com/JacobFV/assignment-rl"
            )
            print("\n")


main()
