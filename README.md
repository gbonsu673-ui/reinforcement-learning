# Reinforcement Learning

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

The Q-Learning update rule is used to update the Q-lookup table during training (it is the same rule used in subsequent environments)

$$
Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha \left[ r_{t+1} + \gamma \max_{a} Q(s_{t+1}, a) - Q(s_t, a_t) \right]
$$

**Results**

The following parameters were set for training:
- `episodes`=`100_000`
- `learning_rate`=`0.9`
- `discount_factor`=`0.9`
- `is_slippery`=`True` this introduces the transition probabilities into the environment

 ![demo](https://github.com/gbonsu673-ui/reinforcement-learning/blob/main/assets/frozenlake_slippery.gif)

**Code reference**
- [frozenlake.py](https://github.com/gbonsu673-ui/reinforcement-learning/blob/main/frozenlake.py)


## 2. Q-Learning on Gymnasium Taxi-v3 (Multiple Objectives)

