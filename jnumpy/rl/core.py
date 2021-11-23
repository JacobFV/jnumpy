# Copyright (c) 2021 Jacob F. Valdez. Released under the MIT license.
"""High-level abtractions for reinforcement learning."""


from __future__ import annotations

import itertools
from typing import Tuple, List, Mapping, Union, Callable

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
        epoch = self.hparams['epoch']
        if epoch not in self.trajs:
            self.trajs[epoch] = []
        self.trajs[epoch] += traj

    def sample(self) -> Traj:
        """Samples a batched trajectory from the buffer stochastically based on:
            - the number of steps in the trajectory (num_steps_replay_coef)
            - how well the agent did in the trajectory (success_replay_coef)
            - how long ago the trajectory was experienced (age_replay_coef)

        Returns:
            Traj: a trajectory of batched steps experienced.
        """
        
        weights = {
            epoch: self.hparams['num_steps_replay_coef'] * len(traj) +
                   self.hparams['success_replay_coef'] * sum(np.sum(step.reward) for step in traj) +
                   self.hparams['age_replay_coef'] * (epoch - self.hparams['epoch'])
            for epoch, traj in self.trajs.items()
        }
        epoch = np.random.choice(list(weights.keys()), p=list(weights.values())/sum(weights.values()))
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
        prev_rewards = {agent_name: 0. for agent_name in agents}

        step = env.reset()
        while not step.done:
            agent_name = next(names_it)
            
            # `Agent.forward` only looks at `step.next_obs` and `step.reward`
            # but I'm assigning defaults just to be safe.
            action = agents[agent_name].forward(Step(
                obs=step.obs,  # what the previous agent saw before acting
                action=step.action,  # what the previous agent did
                next_obs=step.next_obs,  # what the current agent sees before acting
                reward=prev_rewards[agent_name],  # the reward this agent experienced following its last action
                done=step.done,  # whether the environment was done after the previous agent acted
                info=step.info  # any extra information the environment might have output
            )) 

            prev_step = step
            step = env.step(action)  # `Environment.step` produces a Step with all fields except `step.obs` set
            step.obs = prev_step.next_obs  # the current agent's observation is the previous agent's next observation
            prev_rewards[agent_name] = step.reward  # the reward for the action the current agent just took
            trajs[agent_name].append(step)  # Step completely corresponding to this agent (obs before action, obs after action, action, reward, done, info)
        
        # remove the agents' first trajectories since they don't carry Markovian information
        trajs = {n: traj[1:] for n, traj in trajs.items()}  
        return trajs


class ParallelTrainer:
    """Trains `BatchEnv` environments and mutliple agents 
    (with N=1 single-agent supported as a special case).
    
    Uses following hyperparameters:
    - `epoch`: the current epoch. Reads and writes to this variable.
    - `epochs`: the number of epochs to train for.
    - `min_steps_per_epoch`: the minimum number of steps to train for each epoch.
    """

    def __init__(self, hparams: dict, callbacks: List[Callable]):
        self.hparams = hparams
        self.callbacks = callbacks
        
    def train(self, 
        agents: Mapping[str, Agent], 
        env: BatchEnv,
        test_env: BatchEnv = None,
        buffers: Mapping[str, ReplayBuffer] = None,
        collect_driver: ParallelDriver = None,
        test_driver: ParallelDriver = None,
        histories: Mapping[str, Mapping[int, Mapping[str, any]]] = None,
        ) -> Mapping[int, Mapping[str, any]]:

        agent_names = list(agents.keys())
        
        # initialize defaults
        if test_env is None:
            test_env = env
        if buffers is None:
            buffers = dict()
        if collect_driver is None:
            collect_driver = ParallelDriver()
        if test_driver is None:
            test_driver = collect_driver
        if histories is None:
            histories = dict()  # {agent_name: {epoch: {...data}}}

        # build uninitialized agent-specific objects
        for agent_name in agent_names:
            if agent_name not in buffers:
                buffers[agent_name] = ReplayBuffer(hparams=self.hparams)
            if agent_name not in histories:
                histories[agent_name] = dict()

        # run training loop
        for epoch in range(self.hparams['epoch'], self.hparams['epochs']):
            self.hparams['epoch'] = epoch

            # collect trajectories
            steps = 0
            while steps < self.hparams['min_steps_per_epoch']:
                collect_trajs = collect_driver.drive(agents, env)
                steps += min(len(traj) for _, traj in collect_trajs.items())
                for agent_name in agent_names:
                    buffers[agent_name].add(collect_trajs[agent_name])

            # train
            train_trajs = {agent_name: buffers[agent_name].sample() for agent_name in agent_names}
            for agent_name in agent_names:
                agents[agent_name].train(train_trajs[agent_name])
                
            # test
            test_trajs = test_driver.drive(agents, env)

            # record history and run callbacks
            for agent_name in agent_names:
                histories[agent_name][epoch] = {
                    'epoch': epoch,
                    'agent': agents[agent_name],
                    'all_agents': agents,
                    'env': env,
                    'test_env': test_env,
                    'collect_traj': collect_trajs[agent_name],
                    'train_traj': train_trajs[agent_name],
                    'test_traj': test_trajs[agent_name],
                    'buffer': buffers[agent_name],
                }
                for callback in self.callbacks:
                    callback(histories[agent_name][epoch])

        return histories


class PrintCallback:

    def __init__(self, hparams: dict, print_hparam_keys: List[str] = None, print_data_keys: List[str] = None):
        if print_hparam_keys is None:
            print_hparam_keys = ['epoch']
        if print_data_keys is None:
            print_data_keys = []
        
        self.hparams = hparams
        self.print_hparam_keys = print_hparam_keys
        self.print_data_keys = print_data_keys

    def __call__(self, data: Mapping[str, any]):
        for key in self.print_hparam_keys:
            print(f'{key}: {self.hparams[key]}', end='\t')
        for key in self.print_data_keys:
            print(f'{key}: {data[key]}', end='\t')


class QEvalCallback:

    def __init__(self, 
        eval_on_collect: bool = True, 
        eval_on_train: bool = False, 
        eval_on_test: bool = False):

        self.eval_on_collect = eval_on_collect
        self.eval_on_train = eval_on_train
        self.eval_on_test = eval_on_test

    def __call__(self, data: Mapping[str, any]):
        agent = data['agent']
        if not hasattr(agent, 'q_eval'):
            return

        if self.eval_on_collect:
            traj = data['collect_traj']
            q_val = agent.q_eval(traj)
            data['q_collect'] = q_val

        if self.eval_on_train:
            traj = data['train_traj']
            q_val = agent.q_eval(traj)
            data['q_train'] = q_val

        if self.eval_on_test:
            traj = data['test_traj']
            q_val = agent.q_eval(traj)
            data['q_test'] = q_val

# TODO
# - also make a recurrent DQN agent (estimate q function of a sequence of states)
# - make a simple greedy connect4 agent
# - train the preprocessor on an auxillary objective to estimate the max connected for each length for self and for oponent
