# Reinforcement Learning 😌

## 1. Q-Learning on Gymnasium FrozenLake-v1 (8x8 Tiles)

Frozen lake involves crossing a frozen lake from start to goal without falling into any holes by walking over the frozen lake. The player may not always move in the intended direction due to the slippery nature of the frozen lake. More details about the environment, including the observation and action spaces can be found at [Frozen Lake](https://gymnasium.farama.org/environments/toy_text/frozen_lake/).

**Behaviour Policy**

The *Epsilon-Greedy* algorithm is used for both exploration (choosing random actions in the environment) and exploitation (choosing the best actions). It follows the *Greedy in the Limit with Infinite Exploration* theorem where there is a higher exploration rate at the beginning of training and decays as the model converges to an optimal policy. 

$$
a_t =
\begin{cases}
\text{random action} & \text{with probability } \epsilon_t \\
\arg\max_{a} Q(s_t, a) & \text{with probability } 1 - \epsilon_t
\end{cases}
$$

$$
\epsilon_t = \epsilon_{min} + (\epsilon_{max} - \epsilon_{min}) \, e^{-\lambda t}
$$

**Q-Learning Update Rule**

The Q-Learning update rule is used to update the Q-lookup table during training.

$$
Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha \left[ r_{t+1} + \gamma \max_{a} Q(s_{t+1}, a) - Q(s_t, a_t) \right]
$$

**Results**

The following parameters were set for training:
- `episodes`=`100_000`
- `learning_rate`=`0.9`
- `discount_factor`=`0.9`
- `is_slippery`=`True` this introduces the transition probabilities into the environment

**Code reference**
- [frozenlake.py](https://github.com/gbonsu673-ui/reinforcement-learning/blob/main/frozenlake.py)


## 2. SARSA(λ) on Gymnasium FrozenLake-v1 (8x8 Tiles)
Here, I use the SARSA policy control algorithm to train the FrozenLake-v1 8x8 environment. The implementation uses the backward-view of SARSA, which utilises the eligibility trace equation below to weight the contribution of states to a TD-error. The TD update rule used to learn environment is also provided below.

**TD Target:**

$$\delta_t = r_{t+1} + \gamma Q(s_{t+1}, a_{t+1}) - Q(s_t, a_t)$$

**Eligibility trace:**

$$
e_t(s,a) =
\begin{cases}
\gamma \lambda \, e_{t-1}(s,a) + 1 & \text{if } s=s_t, a=a_t \\
\gamma \lambda \, e_{t-1}(s,a) & \text{otherwise}
\end{cases}
$$

**Update rule:**

$$Q(s,a) \leftarrow Q(s,a) + \alpha \, \delta_t \, e_t(s,a) \quad \forall s,a$$

**Code reference**
- [frozenlake_sarsa_lambda.py](https://github.com/gbonsu673-ui/reinforcement-learning/blob/main/frozenlake_sarsa_lambda.py)


## 3. Deep Q-Learning on Gymnasium FrozenLake-v1 (Function Approximation with Neural Network)

This is Deep Reinforcement Learning project that uses the Deep Q-Learning (DQL) algorithm. It uses two neural networks: a Policy Deep Q-Network (DQN) and a Target DQN, to train the FrozenLake-v1 4x4 environment. The Epsilon-Greedy algorithm and the Experience Replay technique are also used as part of DQL to help train the learning agent. PyTorch is used to build the DQNs.

**Code reference**
- [frozenlake_dql.py](https://github.com/gbonsu673-ui/reinforcement-learning/blob/main/frozenlake_dql.py)


## Observation

In all three algorithms, when there are transition probabilities in the environment (i.e. when the slippery flag is set to true), the agent acts in the environment as shown in the gif below. It takes more episodes to train this probabilistic environment than when it is deterministic (i.e when slippery flag is set to false).

<p align="center">
  <img src="https://github.com/gbonsu673-ui/reinforcement-learning/blob/main/assets/frozenlake_slippery.gif" alt="animated" />
</p>

