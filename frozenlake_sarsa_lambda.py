import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle


def select_a_using_e_greedy(q, state, epsilon, is_training, env):
    if is_training and np.random.random() < epsilon:
        return env.action_space.sample()
    else:
        return np.argmax(q[state,:])
    

def run(episodes=15000, is_training=True, is_slippery=False, render=False, discount_rate=0.9, step_size=0.01, lamb=0.2 ):

    # initialize the FrozenLake environment
    env = gym.make("FrozenLake-v1",map_name="8x8", is_slippery=is_slippery, render_mode='human' if render else None)

    # model parameters
    discount_rate = discount_rate
    lamb = lamb
    step_size = step_size
    epsilon = 1.0
    epsilon_decay = 1 / episodes

    if is_training:
        # initialize Q-table
        q = np.zeros((env.observation_space.n, env.action_space.n))
    else:
        # load trained Q-table
        f = open("frozen_lake8x8_sarsa.pkl", "rb")
        q = pickle.load(f)
        f.close()

    for i in range(episodes):
        # initialize the eligibility trace table before every episode
        et = np.zeros((env.observation_space.n, env.action_space.n))

        state = env.reset()[0]
        terminated = False
        truncated = False

        while not terminated and not truncated:
            # select action under the current policy
            a = select_a_using_e_greedy(q, state, epsilon, is_training, env)

            # take the action
            new_state, reward, terminated, truncated,_ = env.step(a)

            # select new action in the new state under the current policy
            new_a = select_a_using_e_greedy(q, state, epsilon, is_training, env)

            # compute TD-error if in training mode
            if is_training:
                delta = reward + discount_rate * q[new_state, new_a] - q[state, a]

                # update eligibility trace of current state
                et[state, a] = et[state, a] + 1

                # for all (s, a) update Q and decay eligibility traces
                q += step_size * delta * et
                et *= discount_rate * lamb

            state = new_state
            a = new_a

        # decay epsilon after every episode
        epsilon = max(epsilon - epsilon_decay, 0)

    # close environment after training
    env.close()

    if is_training:
        # Saving the trained Q-table to a file
        f = open("frozen_lake8x8_sarsa.pkl", "wb")
        pickle.dump(q, f)
        f.close()

if __name__=='__main__':
    run(episodes=20, is_slippery=True, is_training=False, render=True)









        


