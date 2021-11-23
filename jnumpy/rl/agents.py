# Copyright (c) 2021 Jacob F. Valdez. Released under the MIT license.
"""Agent implementations for reinforcement learning."""


from __future__ import annotations

import random
from typing import Tuple, List, Mapping, Union, Callable

import numpy as np

import jnumpy.core as jnp
import jnumpy.nn as jnn
from jnumpy.rl.types import BatchStep, Traj


class Agent:
    """Base class for agents.
    See docstrings on `forward`, `reward`, `train` for more information."""

    def __init__(self, policy: Callable):
        self.policy = policy

    def forward(self, step: BatchStep) -> np.ndarray:
        """Generates an action for a given observation using `self.policy`.
        Override if you want to give your policy more information such as
        recurrent state or previous reward.

        Args:
            step (Step): last step output by the environment. This means the agent
                should feed `step.next_obs`, not `step.obs` to its policy. If the
                environment is multi-agent, then the `reward` attribute has already
                updated to reflect the reward for this agent by the driver.

        Returns:
            np.ndarray: The action to take
        """
        return self.policy(step.next_obs)

    def reward(self, traj: Traj) -> float:
        """Evaluates the cumulative reward for your agent as the sum of
        individual rewards experienced.

        If your agent uses intrinsic rewards, be sure to add them in here.
        Do not introduce Q-values or predicted rewards here.

        Args:
            traj (Traj): timestep trajectory.

        Returns:
            float: Cumulative (sum) reward over the entire sequence.
        """
        return sum(step.reward for step in traj)

    def train(self, traj: Traj):
        """Train your agent on a sequence of Timestep's.

        Args:
            traj (Traj): timestep trajectory.
        """
        raise NotImplemented("Method `train` must be implemented by subclass")


class RandomAgent(Agent):
    """Takes a random action on each timestep.

    Example:
        >>> agent = RandomAgent(num_actions=5)
        >>> obs = jnp.Var(np.array([[1, 2], [3, 4]]))
        >>> step = Step(obs=None, next_obs=obs, action=None, reward=None, done=None, info=None)
        >>> agent.forward(step)

    Args:
        num_actions (int): The number of actions the agent can take.
    """

    def __init__(self, num_actions: int):
        super(RandomAgent, self).__init__(policy=self._policy)
        self.num_actions = num_actions

    def _policy(self, obs: np.ndarray) -> np.ndarray:
        choices = np.random.randint(0, self.num_actions, (obs.shape[0],))
        onehots = np.eye(self.num_actions)[choices]
        return onehots

    def train(self, traj: Traj):
        """Train your agent on a sequence of Timestep's.

        Args:
            traj (Traj): timestep trajectory.
        """
        pass


class RealDQN(Agent):
    """'Classic' Deep Q-learning agent.
    Implements the approach in https://arxiv.org/pdf/1312.5602.pdf.

    NOTE: This agent expects its encoder to output a per-column vector.
        I.E.: [B, H, W, C] --encoder--> [B, W, d_enc]

    This agent uses the following hyperparameters:
        - activation: activation function to use for the hidden layers.
        - hidden_size: hidden layer size.
        - discount: discount factor.
        - optimizer: optimizer to use.
        - epsilon_start: initial epsilon value.
        - min_epsilon: minimum epsilon value.
        - epsilon_decay: epsilon decay rate.
        - epoch: current epoch.

    Example:
        >>> test_batch_size = 5
        >>> test_num_actions = 7
        >>> test_encoder = jnn.Sequential([
                jnn.Conv2D(32, 3, 2, 'same', jnp.Relu),
                jnn.Conv2D(64, 3, 2, 'same', jnp.Relu),
                jnn.AxisMaxPooling(1),
            ])  # [B, H, W, 2] -> [B, W, d_enc]
        >>> test_step = Step(
                obs=np.random.rand(test_batch_size, test_num_actions, test_num_actions, 2),
                next_obs=np.random.rand(test_batch_size, test_num_actions, test_num_actions, 2),
                action=np.random.rand(test_batch_size, test_num_actions),
                reward=np.random.rand(test_batch_size),
                done=False,
                info={})
        >>> test_hparams = {
                'hidden_size': 256,
                'activation': jnp.Relu,
                'optimizer': jnp.SGD(1e-3),
                'epsilon_start': 1.0,
                'epsilon_decay': 0.9,
                'min_epsilon': 0.1,
                'discount': 0.99,
                'epoch': 0}
        >>> test_agent = RealDQN(num_actions=test_num_actions, encoder=test_encoder, hparams=test_hparams)
        >>> test_agent.forward(test_step)

    Args:
        num_actions (int): The number of actions the agent can take.
        encoder (jnn.Layer): An image encoder [B, H, W, C] -> [B, W, d_enc]
        hparams (dict): Hyperparameters.
    """

    def __init__(self, num_actions: int, encoder: jnn.Layer, hparams: dict):

        self.num_actions = num_actions
        self.encoder = encoder  # [B, H, W, C] -> [B, W, d_enc]
        self.neck = jnn.Flatten()
        self.head = jnn.Sequential(
            [
                jnn.Dense(hparams["hidden_size"], hparams["activation"]),
                jnn.Dense(1, jnp.Linear),
            ]
        )  # [B, L+|A|] -> [B, 1]
        self.hparams = hparams

        super(RealDQN, self).__init__(policy=self._policy)

    def _policy(self, obs: np.ndarray) -> np.ndarray:

        B = obs.shape[0]

        # Maybe take greedy step
        epsilon = (
            self.hparams["epsilon_start"]
            * self.hparams["epsilon_decay"] ** self.hparams["epoch"]
        )
        epsilon = max(epsilon, self.hparams["min_epsilon"])
        if random.random() < epsilon:
            indeces = np.random.randint(0, self.num_actions, (B,))
            return np.eye(self.num_actions)[indeces]

        # Otherwise take the action with the highest Q-value
        q_vals = np.zeros((B, self.num_actions))  # [B, self.num_actions]

        # prepare inputs
        obs_T = jnp.Var(obs)  # [B, H, W, 2]
        enc_T = self.neck(self.encoder(obs_T))  # [B, W*d_enc]

        for action_index in range(self.num_actions):

            # prepare action
            action_T = jnp.Var(
                np.repeat(np.array([action_index])[None, :], repeats=B, axis=0)
            )  # [B, 1]

            # run the network
            cat_T = jnp.Concat([enc_T, action_T], axis=1)  # [B, W*d_enc+A]
            q_T = self.head(cat_T)  # [B, 1]

            # store q-values
            q_vals[:, action_index] = q_T.val[:, 0]

        # select the action with the highest Q-value
        action_indeces = np.argmax(q_vals, axis=1)  # [B]
        onehots = np.eye(self.num_actions)[action_indeces]  # [B, self.num_actions]
        return onehots

    def train(self, traj: Traj):
        """Train your agent on a sequence of Timestep's.

        Args:
            traj (Traj): batched timestep trajectory.
        """

        optimizer = self.hparams["optimizer"]
        discount_T = jnp.Var(self.hparams["discount"], trainable=False)  # []
        for step in traj:

            obs_T = jnp.Var(step.obs, trainable=False)  # [B, H, W, 2]
            action_T = jnp.Var(step.action, trainable=False)  # [B, A]
            next_obs_T = jnp.Var(step.next_obs, trainable=False)  # [B, H, W, 2]
            action_next_T = jnp.Var(
                self.policy(step.next_obs), trainable=False
            )  # [B, A]
            r_T = jnp.Var(step.reward, trainable=False)  # [B]

            # compute previous Q value using the actual (not necesarily optimal) action selected
            enc_T = self.neck(self.encoder(obs_T))  # [B, W*d_enc]
            cat_T = jnp.Concat([enc_T, action_T], axis=1)  # [B, d_enc+A]
            Qnow_T = self.head(cat_T)[:, 0]  # [B]
            reg_loss_now_T = self.encoder.loss + self.head.loss  # []
            #                       `int`             `Var`

            # compute the maximum possible next step Q-value
            next_enc_T = self.neck(self.encoder(next_obs_T))  # [B, W*d_enc]
            next_cat_T = jnp.Concat([next_enc_T, action_next_T], axis=1)  # [B, d_enc+A]
            Qnext_T = self.head(next_cat_T)[:, 0]  # [B]
            reg_loss_next_T = self.encoder.loss + self.head.loss  # []

            # train the current policy to estimate new Q-value
            # i.e.: approx Qnew_T <= (1-lr)*Qnow_T + lr*(r_T+discount_T*Qnext_T)  # [B, 1]
            # using gradient descent (let lr=1 in the above eq; small updates are
            # ensured by small SGD lr instead)
            loss_T = (
                jnp.ReduceSum(
                    ((r_T + discount_T * jnp.StopGrad(Qnext_T)) - Qnow_T) ** 2, axis=0
                )
                + reg_loss_now_T
                + reg_loss_next_T
            )  # []
            optimizer.minimize(loss_T)

    def q_eval(self, obs: np.ndarray, action: np.ndarray) -> np.ndarray:
        """Evaluate the Q-value of a given state-action pair.

        Args:
            obs (np.ndarray): observation.
            action (np.ndarray): action.

        Returns:
            q_val (np.ndarray): Q-value of the given state-action pair.
        """

        # prepare inputs
        obs_T = jnp.Var(obs)  # [B, H, W, 2]
        action_T = jnp.Var(action)  # [B, A]
        enc_T = self.neck(self.encoder(obs_T))  # [B, W*d_enc]
        cat_T = jnp.Concat([enc_T, action_T], axis=1)  # [B, W*d_enc+A]
        Qnow_T = self.head(cat_T)[:, 0]  # [B]

        return Qnow_T.val


class CategoricalDQN(Agent):
    """Categorical deep Q-network agent.
    I never read the paper for this architecture, so my implementation may
    be different from the origonal researchers.

    NOTE: This agent expects its encoder to output a per-column vector.
        I.E.: [B, H, W, C] --encoder--> [B, W, d_enc]

    This agent uses the following hyperparameters:
        - activation: activation function to use for the hidden layers.
        - categorical_hidden_size: hidden layer size.
        - discount: discount factor.
        - optimizer: optimizer to use.
        - epsilon_start: initial epsilon value.
        - min_epsilon: minimum epsilon value.
        - epsilon_decay: epsilon decay rate.
        - epoch: current epoch.

    Example:
        >>> test_batch_size = 5
        >>> test_num_actions = 7
        >>> test_encoder = jnn.Sequential([
                jnn.Conv2D(32, 3, 2, 'same', jnp.Relu),
                jnn.Conv2D(64, 3, 2, 'same', jnp.Relu),
                jnn.AxisMaxPooling(1),
            ])  # [B, H, W, 2] -> [B, W, d_enc]
        >>> test_step = Step(
                obs=np.random.rand(test_batch_size, test_num_actions, test_num_actions, 2),
                next_obs=np.random.rand(test_batch_size, test_num_actions, test_num_actions, 2),
                action=np.random.rand(test_batch_size, test_num_actions),
                reward=np.random.rand(test_batch_size),
                done=False,
                info={})
        >>> test_hparams = {
                'categorical_hidden_size': 32,
                'activation': jnp.Relu,
                'optimizer': jnp.SGD(1e-3),
                'epsilon_start': 1.0,
                'epsilon_decay': 0.9,
                'min_epsilon': 0.1,
                'discount': 0.99,
                'epoch': 0}
        >>> test_agent = CategoricalDQN(num_actions=test_num_actions, encoder=test_encoder, hparams=test_hparams)
        >>> test_agent.forward(test_step)

    Args:
        num_actions (int): The number of actions the agent can take.
        encoder (jnn.Layer): An image encoder [B, H, W, C] -> [B, W, d_enc]
        hparams (dict): Hyperparameters.
    """

    def __init__(self, num_actions: int, encoder: jnn.Layer, hparams: dict):

        self.num_actions = num_actions
        self.encoder = encoder  # [B, H, W, C] -> [B, W, d_enc]
        self.neck = jnn.Flatten()
        self.head = jnn.Sequential(
            [
                jnn.Dense(hparams["categorical_hidden_size"], hparams["activation"]),
                jnn.Dense(1, jnp.Linear),
            ]
        )  # [B, W, d_enc] -> [B, A, 1]
        self.hparams = hparams

        super(CategoricalDQN, self).__init__(policy=self._policy)

    def _policy(self, obs: np.ndarray) -> np.ndarray:

        B = obs.shape[0]

        # Maybe take greedy step
        epsilon = (
            self.hparams["epsilon_start"]
            * self.hparams["epsilon_decay"] ** self.hparams["epoch"]
        )
        epsilon = max(epsilon, self.hparams["min_epsilon"])
        if random.random() < epsilon:
            indeces = np.random.randint(0, self.num_actions, (B,))
            return np.eye(self.num_actions)[indeces]

        # Otherwise estimate Q-values for all actions
        obs_T = Var(obs, trainable=False)  # [B, H, W, C]
        enc_T = self.encoder(obs_T)  # [B, W, d_enc]
        qvals_T = self.head(enc_T)  # [B, W, 1]
        return qvals_T[..., 0].val  # [B, W]

    def train(self, traj: Traj):
        """Train your agent on a sequence of Timestep's.

        Args:
            traj (Traj): batched timestep trajectory.
        """

        optimizer = self.hparams["optimizer"]
        discount_T = jnp.Var(self.hparams["discount"], trainable=False)  # []
        for step in traj:

            obs_T = jnp.Var(step.obs, trainable=False)  # [B, H, W, 2]
            action_indeces = np.argmax(step.action, axis=1)  # [B]
            obs_next_T = jnp.Var(step.next_obs, trainable=False)  # [B, H, W, 2]
            r_T = jnp.Var(step.reward, trainable=False)  # [B]

            # compute previous Q value using the actual (not necesarily optimal) action selected
            enc_T = self.encoder(obs_T)  # [B, W, d_enc]
            qvals_T = self.head(enc_T)[..., 0]  # [B, W]
            Q_now_T = qvals_T[action_indeces]  # [B]
            reg_loss_now_T = self.encoder.loss + self.head.loss  # []

            # compute the maximum possible next step Q-value
            enc_next_T = self.encoder(obs_next_T)  # [B, W, d_enc]
            qvals_T = self.head(enc_next_T)[..., 0]  # [B, W]
            Q_next_T = jnp.ReduceMax(qvals_T, axis=1)  # [B]
            # equivalent to: Q_next_T = ReduceMax(qvals_T, axis=1)
            reg_loss_next_T = self.encoder.loss + self.head.loss  # []

            # train the current policy to estimate new Q-value
            # i.e.: approx Qnew_T <= (1-lr)*Qnow_T + lr*(r_T+discount_T*Qnext_T)  # [B, 1]
            # using gradient descent (let lr=1 in the above; small updates are handled in the SGD step)
            # but only update the targets that were actually selected for action at `step_now`.
            loss_T = (
                jnp.ReduceSum(
                    ((r_T + discount_T * jnp.StopGrad(Q_next_T)) - Q_now_T) ** 2, axis=0
                )
                + reg_loss_now_T
                + reg_loss_next_T
            )  # []
            optimizer.minimize(loss_T)

    def q_eval(self, obs: np.ndarray, action: np.ndarray) -> np.ndarray:
        """Evaluate the Q-value of a given state-action pair.

        Args:
            obs (np.ndarray): observation.
            action (np.ndarray): action.

        Returns:
            q_val (np.ndarray): Q-value of the given state-action pair.
        """

        # prepare inputs
        obs_T = jnp.Var(obs)  # [B, H, W, 2]
        action_indeces = np.argmax(action, axis=1)  # [B]

        enc_T = self.encoder(obs_T)  # [B, W, d_enc]
        qvals_T = self.head(enc_T)[..., 0]  # [B, W]

        return qvals_T.val[action_indeces]  # [B]
