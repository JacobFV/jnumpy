# Copyright (c) 2021 Jacob F. Valdez. Released under the MIT license.
"""Common implementations for Connect-N."""

from __future__ import annotations
from ctypes import Union
import itertools

import math
import random
from typing import Optional, Tuple

import numpy as np

import jnumpy as jnp
import jnumpy.rl as jrl
import jnumpy.utils as jutils


class Board:
    """Drafted by copilot with minor human edits"""

    def __init__(self, size=(6, 7), win_length=4):
        self.size: Tuple[int, int] = size  # (rows, cols)
        self.win_length = win_length
        self.board = np.zeros(self.size)
        self.turn = 1
        self.winner = 0

    def __str__(self) -> str:
        board_str = [''.join(str(c) for c in row) + '\r\n'
                     for row in self.board.tolist()]
        return f"{board_str}\n{self.turn}\r\n"

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        return self.board == other.board

    def __hash__(self):
        return hash(self.board.tostring())

    def is_full(self):
        return np.count_nonzero(self.board) == math.prod(self.size)

    def is_empty(self, col):
        return self.board[0, col] == 0

    def is_valid_move(self, col):
        return 0 <= col < self.size and self.is_empty(col)

    def make_move(self, col):
        if self.is_valid_move(col):
            highest_row = np.where(self.board[:, col] == 0)[0][-1]
            self.board[highest_row, col] = self.turn
            self.turn *= -1

    def undo_move(self, col):
        if self.is_valid_move(col) and self.board[0, col] != 0:
            highest_row = np.where(self.board[:, col] == 0)[0][-2]
            self.board[highest_row, col] = 0
            self.turn *= -1

    def check_win(self) -> int:
        for turn in [-1, 1]:
            if self.num_connected(self.win_length, turn) > 0:
                self.winner = turn
                return True
        return self.winner

    def num_connected(self, length, turn):
        num_connected = 0
        for row, col in itertools.product(
                range(self.size[0]-length),
                range(self.size[1]-length)):
            # horizontal match
            if np.all(self.board[row, col: col + length] == turn):
                num_connected += 1
            # vertical match
            if np.all(self.board[row: row + length, col] == turn):
                num_connected += 1
            # diagonal match
            if all(self.board[row + i, col + i] == turn for i in range(length)):
                num_connected += 1
            if all(self.board[row + i, col - i] == turn for i in range(length)):
                num_connected += 1
        return num_connected

    @staticmethod
    def from_file(self, filename, win_length=4) -> Board:
        with open(filename, "r") as f:
            int_arr = [list(map(int, line.strip())) for line in f]
            board_arr = np.array(int_arr[:-1])
            board = Board(size=board_arr.size, win_length=win_length)
            board.board = board_arr
            board.turn = int_arr[-1][0]
            return board

    def to_file(self, filename):
        with open(filename, "w") as f:
            f.write(self.__str__())


class BoardEnv(jrl.NoBatchEnv):
    """RL environment for Connect 4.
    See docstrings under `__init__`, `reset`, and `step` for more information.

    Example:
        >>> board_size = 10
        >>> env = BoardEnv(board_size=board_size, win_length=5, reward_mode="dense_advantage")
        >>> step = env.reset()
        >>> while not step.done:
                # random move
                action = np.random.uniform(0, 1, (board_size,))
                step = env.step(action)
                # show game state
                print(f"Action: {action}")
                env.render()
                print(f"Reward: {step.reward}\n")
        >>> print(f"Winner: {env.board.winner}")
    """

    reward_modes = ['simple', 'sparse', 'dense_stateless', 'dense_advantage']

    def __init__(self, initial_board_size: Optional[Union[int, Tuple[int, int]]] = None,
                 win_length: int = None, reward_mode: str = "simple", finish_on_win: bool = True,
                 initial_board: Board = None):
        """RL environment for Connect 4.

        Args:
            initial_board_size (int or tuple, optional): The size (rows, cols) of the board.
            win_length (int, optional): The minimum connected length to win. Defaults to 4.
            reward_mode (str, optional): One of 'sparse', 'dense_stateless', 'dense_advantage'.
                - For 'simple', the reward is the number of `win_length`-length connected
                    pieces in a row.
                - For 'sparse', the reward is 1 if the player has attained a connect `win_length`,
                    and is 0 otherwise.
                - For 'dense_stateless', the reward increases linearly with the number of N-in-a-row's
                    for all values of N from 0 to board_size weighted logarithmically by N.
                - For 'dense_advantage', the reward is determined by the difference between the
                    previous and current dense reward for each player individually.
                `reward_mode` defaults to 'simple'.
            finish_on_win (bool, optional): Whether to finish the episode when a player has won.
                Defaults to True.
            initial_board (Board, optional): The initial board. Defaults to None.
        """
        if initial_board_size and isinstance(initial_board_size, int):
            initial_board_size = (initial_board_size, initial_board_size)

        self.initial_board_size = initial_board_size
        self.win_length = win_length
        self.reward_mode = reward_mode
        self.finish_on_win = finish_on_win
        self.intial_board = initial_board

        self._prev_reward = None

    @property
    def rows(self):
        return self.board.size[0]

    @property
    def cols(self):
        return self.board.size[1]

    def reset(self) -> jrl.NoBatchStep:
        self.board = self.intial_board if self.intial_board \
            else Board(self.initial_board_size, self.win_length)

        if self.reward_mode == "dense_advantage":
            self.prev_dense_reward = [0.0, 0.0]

        return jrl.NoBatchStep(
            obs=np.zeros_like(self._make_obs()),
            action=np.zeros(self.board_size),
            next_obs=self._make_obs(),
            reward=np.array(0.0),
            done=False,
            info=dict(),
        )

    def step(self, action: np.ndarray) -> jrl.NoBatchStep:
        """Apply agent X's action to the board and
        returns the next agent's timestep.

        Args:
            action (np.ndarray): array shaped (board_size,). The arg max
                action index is the column where next piece is placed.

        Returns:
            tuple: NoBatchStep with values:
                obs (np.ndarray[H, W, 2]): None
                next_obs (np.ndarray[H, W, 2]): the next board state with
                    self's entered squares represented in channel 0 and
                    opponent's squares represented in channel 1
                action (np.ndarray[board_size]): None
                reward (float): the reward for the agent
                    If sparse_reward is True, then reward is -1, 0, or +1.
                    If sparse_reward is False, then reward is:
                        ego_dense_reward - opponent_dense_reward.
                done (bool): whether the game is over
                info (dict): extra information
        """
        # Apply action
        action_index = np.argmax(action)  # []
        self.board.make_move(action_index)  # this flips `board.turn`
        # Temporarily unflip `board.turn`
        self.board.turn *= -1

        # Make egocentric observation
        obs = self._make_obs()

        # Compute reward
        # This is the lazy way to do it, but it's fast enough

        # Sparse reward
        winner = self.board.check_win()
        sparse_reward = self.board.turn * winner

        # Dense reward
        def dense_reward_for_turn(board, turn):
            r = 0
            for length in range(2, min(self.board.size)):
                r += math.log(length) * board.num_connected(length, turn)
            return r

        ego_dense_reward = dense_reward_for_turn(self.board, self.board.turn)
        opponent_dense_reward = dense_reward_for_turn(
            self.board, -self.board.turn)
        dense_reward = ego_dense_reward - opponent_dense_reward

        reward: float
        if self.reward_mode == "simple":
            reward = self.board.num_connected(self.win_length, self.board.turn)
        elif self.reward_mode == "sparse":
            reward = sparse_reward
        elif self.reward_mode == "dense_stateless":
            reward = dense_reward
        elif self.reward_mode == "dense_advantage":
            turn_index = (self.board.turn + 1) // 2
            reward = dense_reward - self.prev_dense_reward[turn_index]
            self.prev_dense_reward[turn_index] = dense_reward
        else:
            raise ValueError(f"Invalid reward_mode: {self.reward_mode}")

        # Evaluate whether game is over
        winner = self.board.check_win()
        done = winner != 0 and self.finish_on_win

        # Record debugging info
        info = dict()

        # Revert temporary flip on `board.turn`
        self.board.turn *= -1

        # save the reward for rendering
        self._prev_reward = reward

        return jrl.NoBatchStep(
            obs=np.zeros_like(obs),
            next_obs=obs,
            action=action,
            reward=np.array(reward),
            done=done,
            info=info,
        )

    def render(self):
        print(str(self.board)[:-2] + self._prev_reward + '\r\n')

    def _make_obs(self) -> np.ndarray:
        """Only show ego values on first channel and opponent values on second channel"""
        obs = np.stack(
            [self.board.turn * self.board.board, -
                self.board.turn * self.board.board],
            axis=-1,
        )  # [board_size, board_size, 2]
        obs[obs < 0] = 0  # rectify negative values
        return obs


def get_human_move(board_cols: int) -> np.ndarray:
    return jutils.get_input(
        prompt=f"Enter move (0-{board_cols}): ",
        parser=lambda x: np.eye(board_cols)[int(x)][None, :],
        validator=lambda x: 0 <= x < board_cols)


class MaxConnect4Agent(jrl.Agent):
    def __init__(self, num_actions: int, look_ahead: int = 1):
        self.num_actions = num_actions
        self.look_ahead = look_ahead
        super(MaxConnect4Agent, self).__init__(policy=self._policy)

    def _policy(self, obs: np.ndarray) -> np.ndarray:
        B = obs.shape[0]
        action = np.zeros(B, dtype=np.int32)
        for b in range(B):
            o = obs[b]
            # TODO: make a greedy agent
            action[b] = random.randint(0, self.board_size - 1)
        return action

    def train(self, traj: jrl.Traj):
        print('Greedy agents do not train')
