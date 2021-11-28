# Copyright (c) 2021 Jacob F. Valdez. Released under the MIT license.
"""High-level abtractions for reinforcement learning."""


from __future__ import annotations

import itertools
from typing import Tuple, List, Mapping, Union, Callable
from copy import deepcopy

import numpy as np

import jnumpy.core as jnp
from jnumpy.rl.types import Step, BatchStep, NoBatchStep, Traj
from jnumpy.rl.agents import Agent


class Environment:
    """RL environment."""

    def __init__(self):
        pass

    def reset(self) -> Step:
        """Resets the environment

        Returns:
            Step: Initial step. The `next_obs` attribute should be set
                with an initial observation. `done` should be False.
                `obs` and `action` should not be used.
        """
        pass

    def step(self, action: np.ndarray) -> Step:
        """Computes one logical step in the environment

        Args:
            action (np.ndarray): The action to take.

        Returns:
            Step: Step resulting from taking `action`. The `next_obs` attribute
                should be set with the observation resulting from taking the `action`
                in the current environment state. `obs` should not be used. If the
                environment is turn-based, then the reward should correspond to the
                agent that just acted (not the next agent in line to act).
        """
        pass

    def render(self):
        pass


class NoBatchEnv(Environment):
    """Environment with no batch dimension."""

    def reset(self) -> NoBatchStep:
        pass

    def step(self, action: np.ndarray) -> NoBatchStep:
        pass


class BatchEnv(Environment):
    """Environment with batch dimension."""

    def reset(self) -> BatchStep:
        pass

    def step(self, action: np.ndarray) -> BatchStep:
        pass


class Batch1Env(BatchEnv):
    """Adds a batch axis to all outgoing Steps and strips it off incoming Steps."""

    def __init__(self, env: NoBatchEnv):
        self.env = env

    def reset(self) -> BatchStep:
        return Step.from_no_batch_axis(self.env.reset())

    def step(self, action: np.array) -> BatchStep:
        return Step.from_no_batch_axis(self.env.step(action[0]))

    def render(self):
        self.env.render()


class ParallelEnv(BatchEnv):
    """Keeps reseting the same environment in a batch.

    Declares itself done when a total of `batch_size` individual environment
    dones are experienced.

    NOTE: Individual environments should be `NoBatchEnv`'s.
    """

    def __init__(self, batch_size: int, env_init_fn: Callable[[], NoBatchEnv]):
        self.batch_size = batch_size
        self.env_init_fn = env_init_fn

    def reset(self) -> Step:
        self.dones = 0
        self.envs = [self.env_init_fn() for _ in range(self.batch_size)]
        steps = [env.reset() for env in self.envs]
        steps = [Step.from_no_batch_axis(step) for step in steps]
        return Step.batch(steps)

    def step(self, action: np.array) -> Step:
        steps = []
        for i, (env, single_action) in enumerate(zip(self.envs, action)):
            step = env.step(single_action)
            if step.done:
                self.envs[i] = self.env_init_fn()
                new_step = env.reset()
                step.next_obs = new_step.next_obs
                self.dones += 1
            steps.append(Step.from_no_batch_axis(step))

        batched_step = Step.batch(steps)
        batched_step.done = self.dones >= self.batch_size
        return batched_step

    def render(self):
        for env in self.envs:
            env.render()


class ReplayBuffer:
    """Replay buffer.

    Expects the following hyperparameters:
        - `epoch`: The current epoch.
        - `batch_size`: Number of trajectories to return at each call.
        - `min_sample_len`: Minimum length of trajectories to sample.
        - `max_sample_len`: Maximum length of trajectories to sample.
        - `num_steps_replay_coef`: Sampling coefficient based on trajectory length.
        - `success_replay_coef`: Sampling coefficient based on trajectory success.
        - `age_replay_coef`: Sampling coefficient based on trajectory age.
    """

    def __init__(self, hparams: dict):
        self.hparams = hparams
        self.trajs = dict()

    @property
    def flat_traj(self):
        return [step for traj in self.trajs for step in traj]

    def add(self, traj: Traj):
        """Add a new trajectory to the buffer.

        Args:
            traj (Traj): the trajectory to add.
        """
        epoch = self.hparams["epoch"]
        if epoch not in self.trajs:
            self.trajs[epoch] = []
        self.trajs[epoch] += traj

    def sample(self) -> Traj:
        """Samples a batched trajectory from the buffer stochastically based on:
            - the number of steps in the trajectory (num_steps_replay_coef)
            - how well the agent did in the trajectory (success_replay_coef)
            - how long ago the trajectory was experienced. Positive favors newer,
                negative favors older experiences. (age_replay_coef)

        Returns:
            Traj: a trajectory of batched steps experienced.
        """

        weights = {
            epoch: self.hparams["num_steps_replay_coef"] * len(traj)
            + self.hparams["success_replay_coef"]
            * sum(np.sum(step.reward) for step in traj)
            + self.hparams["age_replay_coef"] * (epoch - self.hparams["epoch"])
            for epoch, traj in self.trajs.items()
        }
        weights_arr = np.array(list(weights.values()))
        weights_arr = weights_arr - np.min(weights_arr) + 1e3
        epoch = np.random.choice(
            list(weights.keys()), p=weights_arr / np.sum(weights_arr)
        )
        return self.trajs[epoch]


class ParallelDriver:
    """Drives batched turn-based `BatchEnv` environments with multiple agents.
    Also supports single-agent environments as a special case.
    """

    def __init__(self):
        pass

    def drive(self, agents: Mapping[str, Agent], env: BatchEnv) -> Mapping[str, Traj]:
        """Drives a batched environment with multiple agents.

        Args:
            agents (Mapping[str, Agent]): A dictionary of agents to drive.
            env (BatchEnv): The environment to drive.

        Returns:
            Mapping[str, Traj]: A dictionary of trajectories for each agent.
                Each trajectory is completely disengaged from the other agent's.
                (i.e.: the obs, next_obs, action, reward, done attributes are
                individual to each agent for each trajectory.)
        """

        names_it = itertools.cycle(agents.keys())
        trajs = {agent_name: [] for agent_name in agents}
        prev_rewards = {agent_name: 0.0 for agent_name in agents}

        step = env.reset()
        while not step.done:
            agent_name = next(names_it)

            # `Agent.forward` only looks at `step.next_obs` and `step.reward`
            # but I'm assigning defaults just to be safe.
            action = agents[agent_name].forward(
                Step(
                    obs=step.obs,  # what the previous agent saw before acting
                    action=step.action,  # what the previous agent did
                    next_obs=step.next_obs,  # what the current agent sees before acting
                    reward=prev_rewards[
                        agent_name
                    ],  # the reward this agent experienced following its last action
                    done=step.done,  # whether the environment was done after the previous agent acted
                    info=step.info,  # any extra information the environment might have output
                )
            )

            prev_step = step
            step = env.step(
                action
            )  # `Environment.step` produces a Step with all fields except `step.obs` set
            step.obs = (
                prev_step.next_obs
            )  # the current agent's observation is the previous agent's next observation
            prev_rewards[
                agent_name
            ] = step.reward  # the reward for the action the current agent just took
            trajs[agent_name].append(
                step
            )  # Step completely corresponding to this agent (obs before action, obs after action, action, reward, done, info)

        # remove the agents' first trajectories since they don't carry Markovian information
        trajs = {n: traj[1:] for n, traj in trajs.items()}
        return trajs


class ParallelTrainer:
    """Trains `BatchEnv` environments and mutliple agents
    (with N=1 single-agent supported as a special case).
    """

    def __init__(self, callbacks: List[Callable]):
        self.callbacks = callbacks

    def train(
        self,
        agents: Union[List[Agent], Mapping[str, Agent]],
        all_hparams: Mapping[str, Mapping[str, any]],
        env: BatchEnv,
        test_env: BatchEnv = None,
        collect_driver: ParallelDriver = None,
        test_driver: ParallelDriver = None,
        training_epochs: int = 1,
    ) -> Mapping[str, Mapping[str, any]]:
        """Train a group of agents on a batched environment.

        Args:
            agents (Union[List[Agent], Mapping[str, Agent]]): The agents to train.
            all_hparams (Mapping[str, Mapping[str, any]]): Data (including hyperparameters) for each agent. If only
                a single agent is provided, it will be used for all agents. Can also include hyperparameters for
                agents that are not being trained. Each agent's hyperparameters will be updated as the training
                progresses. Agent hyperparameters should include the following keys:
                - `epoch`: the current training epoch for that agent.
                - `min_steps_per_epoch`: the minimum number of steps to train each epoch for in a `train` call.
                - `num_steps_replay_coef`: the weight of the number of steps in the trajectory in the loss function.
                - `success_replay_coef`: the weight of the success of the agent in the loss function.
                - `age_replay_coef`: the weight of the age of the trajectory in the loss function.
                as well as any other hyperparameters that the agent needs. NOTE: hparam dicts should not be
                shared between agents since they are updated individually.
            env (BatchEnv): The environment to train on.
            test_env (BatchEnv, optional): The environment to test on. Defaults to `env`.
            collect_driver (ParallelDriver, optional): Train environment driver. Defaults to `ParallelDriver`.
            test_driver (ParallelDriver, optional): Test environment driver. Defaults to `collect_driver`.
            training_epochs (int, optional): How many epochs to train for. Defaults to 1.

        Returns:
            Mapping[str, Mapping[str, any]]: updated hyperparameters for each agent.
        """

        if isinstance(agents, list):
            agents = {agent.name: agent for agent in agents}

        agent_names = list(agents.keys())

        if len(all_hparams) == 1:
            single_hparam = next(iter(all_hparams.values()))
            all_hparams = {
                agent_name: deepcopy(single_hparam) for agent_name in agent_names
            }

        # initialize defaults
        if test_env is None:
            test_env = env
        if collect_driver is None:
            collect_driver = ParallelDriver()
        if test_driver is None:
            test_driver = collect_driver

        # build uninitialized agent-specific objects
        for name in agent_names:
            if "epoch" not in all_hparams[name]:
                all_hparams[name]["epoch"] = 0
            if "history" not in all_hparams[name]:
                all_hparams[name]["history"] = dict()  # {epoch: {...data}}
            if "buffer" not in all_hparams[name]:
                all_hparams[name]["buffer"] = ReplayBuffer(hparams=all_hparams[name])
            if "steps" not in all_hparams[name]:
                all_hparams[name]["steps"] = 0

        # run training loop
        for train_epoch in range(training_epochs):

            # collect trajectories
            for name in agent_names:
                all_hparams[name]["steps"] = 0
            while any(
                all_hparams[name]["steps"] < all_hparams[name]["min_steps_per_epoch"]
                for name in agent_names
            ):
                collect_trajs = collect_driver.drive(agents, env)
                for name in agent_names:
                    all_hparams[name]["buffer"].add(collect_trajs[name])
                    all_hparams[name]["steps"] += len(collect_trajs[name])

            # train
            train_trajs = {
                name: all_hparams[name]["buffer"].sample() for name in agent_names
            }
            for name in agent_names:
                agents[name].train(train_trajs[name])

            # test
            test_trajs = test_driver.drive(agents, env)

            # record history and run callbacks
            for name in agent_names:
                agent_epoch = all_hparams[name]["epoch"]
                collect_traj = all_hparams[name]["buffer"].trajs[agent_epoch]
                all_hparams[name]["history"][agent_epoch] = {
                    "epoch": agent_epoch,
                    "agent": agents[name],
                    "all_agents": agents,
                    "env": env,
                    "test_env": test_env,
                    "collect_traj": collect_traj,
                    "train_traj": train_trajs[name],
                    "test_traj": test_trajs[name],
                    "collect_reward": agents[name].reward(collect_traj),
                    "train_reward": agents[name].reward(train_trajs[name]),
                    "test_reward": agents[name].reward(test_trajs[name]),
                }
                for callback in self.callbacks:
                    callback(all_hparams[name]["history"][agent_epoch])

            # increment individual agent epoch's
            for name in agent_names:
                all_hparams[name]["epoch"] += 1

        return all_hparams


class PrintCallback:
    def __init__(self, keys: List[str]):
        self.keys = keys

    def __call__(self, data: Mapping[str, any]):
        for key in self.keys:
            if key in data and data[key] is not None:
                if isinstance(data[key], float):
                    print(f"{key}: {data[key]:+6.4f}", end="\t")
                else:
                    print(f"{key}: {data[key]}", end="\t")
        print()


class QEvalCallback:
    def __init__(
        self,
        eval_on_collect: bool = True,
        eval_on_train: bool = False,
        eval_on_test: bool = False,
    ):

        self.eval_on_collect = eval_on_collect
        self.eval_on_train = eval_on_train
        self.eval_on_test = eval_on_test

    def __call__(self, data: Mapping[str, any]):
        agent = data["agent"]
        if not hasattr(agent, "q_eval"):
            return

        if self.eval_on_collect:
            traj = data["collect_traj"]
            cat_obs = np.concatenate([s.obs for s in traj], axis=0)
            cat_actions = np.concatenate([s.action for s in traj], axis=0)
            q_vals = agent.q_eval(cat_obs, cat_actions)
            q_val = np.mean(q_vals)
            data["q_collect"] = q_val

        if self.eval_on_train:
            traj = data["train_traj"]
            cat_obs = np.concatenate([s.obs for s in traj], axis=0)
            cat_actions = np.concatenate([s.action for s in traj], axis=0)
            q_vals = agent.q_eval(cat_obs, cat_actions)
            q_val = np.mean(q_vals)
            data["q_train"] = q_val

        if self.eval_on_test:
            traj = data["test_traj"]
            cat_obs = np.concatenate([s.obs for s in traj], axis=0)
            cat_actions = np.concatenate([s.action for s in traj], axis=0)
            q_vals = agent.q_eval(cat_obs, cat_actions)
            q_val = np.mean(q_vals)
            data["q_test"] = q_val


# TODO
# - also make a recurrent DQN agent (estimate q function of a sequence of states)
# - make a simple greedy connect4 agent
# - train the preprocessor on an auxillary objective to estimate the max connected for each length for self and for oponent
