# Playing Connect-N with Jnumpy Reinforcement Learning

This example shows how `jnumpy.rl` can be used to train a Connect-N agent. Core modules are in `common.py`. Code for the experiments below is in `experiment.py`. For an interactive connect-N game, run `python3 play.py`. Please make sure you have `jnumpy` installed and your working directory is at `examples/connect-n` before running the experiments.

## Experiments

I train 5 agents (`Ally`, `Bob`, `Cara`, `Dan`, `Emma`) at competitive group-play with separate weights for each agent. Following [Mnih et. al 2013](https://arxiv.org/pdf/1312.5602.pdf), agents were trained offline using an experience replay buffer. Different from their approach, agents had separate replay buffers and experiences were weighted by the sum of episode reward and length to encourage their Q-functions with positive experiences. Agents also maintained individualized epoch counts reflecting different stages of maturity in their multi-agent development. Agents were trained in two experiments:
 - Experiment 1: Sample a new random pair of agents every 3 epochs for training.
 - Experiment 2: Sample a new random pair of agents at every epoch for training.

The following hyperparameters were used (same for all 5 agents):
```python
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
```

## Results

Both experiments took about 1 hour to complete. Several other experiments were also conducted but are not reported due to training divergence (2 cases), numerical overflow (1 case), and failure to save logs (3+ cases).

`collect_reward` is the mean reward obtained by an agent when collecting new experience. `train_reward` is the mean reward of the trajectories that the agent trained on. `test_reward` is the mean reward of the trajectories that the agent collected from the test environment. `q_collect/train/test` refers to the mean Q-value of each agent over corresponding trajectories. 

### Experiment 1
```
epoch: 0        agent: Ally     collect_reward: +0.8308 train_reward: +0.8308   test_reward: +1.0574    q_collect: +0.9673      q_train: +0.9673        q_test: +0.9674
epoch: 0        agent: Dan      collect_reward: +0.1864 train_reward: +0.1864   test_reward: -0.0242    q_collect: +0.1705      q_train: +0.1705        q_test: +0.1705
epoch: 1        agent: Ally     collect_reward: +1.0100 train_reward: +1.0100   test_reward: +0.7937    q_collect: +2.5260      q_train: +2.5260        q_test: +2.5233
epoch: 1        agent: Dan      collect_reward: +0.0247 train_reward: +0.0247   test_reward: +0.1961    q_collect: +0.1931      q_train: +0.1931        q_test: +0.1932
epoch: 2        agent: Ally     collect_reward: +0.7770 train_reward: +0.7770   test_reward: +0.5026    q_collect: +3.9085      q_train: +3.9085        q_test: +3.8937
epoch: 2        agent: Dan      collect_reward: +0.2334 train_reward: +0.0247   test_reward: +0.6025    q_collect: +0.2157      q_train: +0.2167        q_test: +0.2168
epoch: 3        agent: Dan      collect_reward: +0.8685 train_reward: +0.8685   test_reward: +0.8460    q_collect: +1.5881      q_train: +1.5881        q_test: +1.5852
epoch: 0        agent: Emma     collect_reward: +0.2447 train_reward: +0.2447   test_reward: +0.2121    q_collect: +0.3704      q_train: +0.3704        q_test: +0.3676
epoch: 4        agent: Dan      collect_reward: +0.7014 train_reward: +0.2334   test_reward: +1.7234    q_collect: +1.2409      q_train: +1.2360        q_test: +1.2232
epoch: 1        agent: Emma     collect_reward: +0.3448 train_reward: +0.3448   test_reward: -0.6870    q_collect: +2.5168      q_train: +2.5168        q_test: +2.7928
epoch: 5        agent: Dan      collect_reward: +1.5189 train_reward: +1.5189   test_reward: +1.7944    q_collect: +9.3871      q_train: +9.3871        q_test: +9.3622
epoch: 2        agent: Emma     collect_reward: -0.4036 train_reward: +0.2447   test_reward: -0.6933    q_collect: +1.2033      q_train: +1.2046        q_test: +1.2008
epoch: 3        agent: Ally     collect_reward: +1.0194 train_reward: +0.8308   test_reward: +1.4696    q_collect: +6.2479      q_train: +6.2454        q_test: +6.2634
epoch: 0        agent: Cara     collect_reward: +0.0296 train_reward: +0.0296   test_reward: -0.4650    q_collect: +0.0256      q_train: +0.0256        q_test: +0.0256
epoch: 4        agent: Ally     collect_reward: +0.5440 train_reward: +0.5440   test_reward: -0.1834    q_collect: +6.7465      q_train: +6.7465        q_test: +6.7203
epoch: 1        agent: Cara     collect_reward: +0.4800 train_reward: +0.4800   test_reward: +1.1749    q_collect: +0.5851      q_train: +0.5851        q_test: +0.5860
epoch: 5        agent: Ally     collect_reward: +0.8596 train_reward: +1.0100   test_reward: +1.4455    q_collect: +10.8626     q_train: +10.8795       q_test: +10.9179
epoch: 2        agent: Cara     collect_reward: +0.2382 train_reward: +0.4800   test_reward: -0.3850    q_collect: +1.1182      q_train: +1.1183        q_test: +1.1190
epoch: 0        agent: Bob      collect_reward: +0.9509 train_reward: +0.9509   test_reward: +0.2762    q_collect: +1.2698      q_train: +1.2698        q_test: +1.2654
epoch: 6        agent: Dan      collect_reward: +0.1252 train_reward: +1.5189   test_reward: +0.7476    q_collect: +28.8336     q_train: +28.9175       q_test: +28.7053
```

### Experiment 2
```
epoch: 0        agent: Bob      collect_reward: +1.2229 train_reward: +1.2229   test_reward: +0.1697       q_collect: +1.6110      q_train: +1.6110        q_test: +1.6099
epoch: 0        agent: Dan      collect_reward: -0.1890 train_reward: -0.1890   test_reward: +1.0630       q_collect: -0.2440      q_train: -0.2440        q_test: -0.2442
epoch: 1        agent: Dan      collect_reward: +1.0742 train_reward: +1.0742   test_reward: +1.1045       q_collect: +1.1573      q_train: +1.1573        q_test: +1.1567
epoch: 0        agent: Emma     collect_reward: -0.0866 train_reward: -0.0866   test_reward: -0.0813       q_collect: -0.1203      q_train: -0.1203        q_test: -0.1197
epoch: 1        agent: Bob      collect_reward: +0.7236 train_reward: +1.2229   test_reward: +0.4376       q_collect: +5.5387      q_train: +5.5587        q_test: +5.5151
epoch: 0        agent: Cara     collect_reward: +0.4005 train_reward: +0.4005   test_reward: +0.4520       q_collect: +0.4906      q_train: +0.4906        q_test: +0.4905
epoch: 0        agent: Ally     collect_reward: +1.1479 train_reward: +1.1479   test_reward: +1.0789       q_collect: +1.3787      q_train: +1.3787        q_test: +1.3786
epoch: 2        agent: Bob      collect_reward: -0.0879 train_reward: -0.0879   test_reward: +0.0781       q_collect: +3.4035      q_train: +3.4035        q_test: +3.4038
epoch: 1        agent: Cara     collect_reward: +1.2075 train_reward: +0.4005   test_reward: +0.7563       q_collect: +0.9473      q_train: +0.9468        q_test: +0.9464
epoch: 1        agent: Emma     collect_reward: -0.1779 train_reward: -0.1779   test_reward: +0.2953       q_collect: -0.3017      q_train: -0.3017        q_test: -0.3005
epoch: 2        agent: Cara     collect_reward: +0.8869 train_reward: +0.8869   test_reward: +0.4128       q_collect: +3.8562      q_train: +3.8562        q_test: +3.8727
epoch: 2        agent: Dan      collect_reward: +0.1139 train_reward: +1.0742   test_reward: +0.7359       q_collect: +2.4112      q_train: +2.3903        q_test: +2.4152
epoch: 1        agent: Ally     collect_reward: +0.7223 train_reward: +0.7223   test_reward: +0.6523    q_collect: +1.5575      q_train: +1.5575        q_test: +1.5596
epoch: 2        agent: Emma     collect_reward: +0.3760 train_reward: +0.3760   test_reward: +0.3822    q_collect: -0.2053      q_train: -0.2053        q_test: -0.2057
epoch: 3        agent: Bob      collect_reward: +0.3311 train_reward: +0.3311   test_reward: +0.9607    q_collect: +3.9185      q_train: +3.9185        q_test: +3.8993
epoch: 3        agent: Emma     collect_reward: +0.7839 train_reward: +0.3760   test_reward: +0.1323    q_collect: +0.3022      q_train: +0.3017        q_test: +0.3023
epoch: 2        agent: Ally     collect_reward: +0.7667 train_reward: +0.7667   test_reward: +1.2142    q_collect: +2.5045      q_train: +2.5045        q_test: +2.5043
epoch: 3        agent: Cara     collect_reward: +0.2782 train_reward: +0.2782   test_reward: -0.1929    q_collect: +2.3590      q_train: +2.3590        q_test: +2.3654
epoch: 3        agent: Ally     collect_reward: +1.0699 train_reward: +0.7667   test_reward: +0.9112    q_collect: +3.1780      q_train: +3.1779        q_test: +3.1786
epoch: 3        agent: Dan      collect_reward: -0.0017 train_reward: +0.1139   test_reward: +0.1046    q_collect: +2.2367      q_train: +2.2386        q_test: +2.2386
```

## Analysis

