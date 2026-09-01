import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle

def run(episodes, is_training=True, render=False, learning_rate=0.9, discount_factor=0.9):

    # initialize the FrozenLake environment
    env = gym.make('FrozenLake-v1', map_name='8x8', is_slippery=False, render_mode='human' if render else None)

    # initialize a Q lookup table only when we are training mode. Load otherwise
    if (is_training):
        # initialize the Q lookup table
        q = np.zeros((env.observation_space.n, env.action_space.n)) # a 64 X 4 state-action array
    else:
        # loading an already trained Q-table
        f = open('frozen_lake8x8.pkl', 'rb')
        q = pickle.load(f)
        f.close()

    # hyperparameters for Q-Learning
    learning_rate_a = learning_rate
    discount_factor = discount_factor

    # GLIE - Greedy in the Limit with Infinite Exploration for the e-greedy algorithm
    # This balances exploration and convergence to an optimal policy as the model continues to learn

    # e-greedy algorithm parameters
    epsilon = 1 # 100% of the time random actions are selected
    epsilon_decay_rate = 1 / episodes # a decay rate of 1/K where K is the number of episodes
    rng = np.random.default_rng() # random number generator

    # tracking the rewards
    rewards_per_episode = np.zeros(episodes)


    # start training for a number of episodes
    for i in range(episodes):
        state = env.reset()[0]
        terminated = False
        truncated = False

        while (not terminated and not truncated):
            # e-greedy algorithm for action selection
            if is_training and rng.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q[state,:])

            # get next state, reward, and termination states
            new_state, reward, terminated, truncated,_ = env.step(action)

            # update Q-table only when we are training
            if is_training:
                # perform a Q-Learning update after every step using the Q-Learning update formular
                q[state, action] = q[state, action] + learning_rate_a * (reward + discount_factor * np.max(q[new_state,:]) - q[state, action])

            state = new_state

        # decay epsilon after each episode
        epsilon = max(epsilon - epsilon_decay_rate, 0)

        # reduce the learning rate after every epsiode to stabilize the Q-values when there is no exploration
        if(epsilon == 0):
            learning_rate_a = 0.0001

        # accumulating positive rewards after every episode
        if reward == 1:
            rewards_per_episode[i] = 1

    # close environment after training
    env.close()

    # AFTER TRAINING
    # Graphing the rewards per episode after training
    sum_rewards = np.zeros(episodes)
    for t in range(episodes):
        sum_rewards[t] = np.sum(rewards_per_episode[max(0, t-100):(t+1)])
    plt.plot(sum_rewards)
    plt.savefig('frozen_lake8x8.png')

    if is_training:
        # Saving the trained Q-table to a file
        f = open("frozen_lake8x8.pkl", "wb")
        pickle.dump(q, f)
        f.close()

if __name__ == '__main__':
    episodes = 25
    run(episodes, is_training=False, render=True)
