import gymnasium as gym
import numpy as np
from collections import deque
import random
import torch
from torch import nn
import torch.nn.functional as F

# defining a simple feed-forward model
class DQN(nn.Module):
    def __init__(self, in_states, h1_nodes, out_actions):
        super().__init__()

        # defining network layers
        self.fc1 = nn.Linear(in_states, h1_nodes) # this the first fully connected layer
        self.out = nn.Linear(h1_nodes, out_actions) # output layer

    def forward(self, x):
        x = F.relu(self.fc1(x)) # apply Relu activation function after fc1
        x = self.out(x) # compute output
        return x

# defining memory for Experience Replay
class ReplayMemory():
    def __init__(self, maxlen):
        self.memory = deque([], maxlen=maxlen)

    def append(self, transition):
        self.memory.append(transition)

    def sample(self, sample_size):
        return random.sample(self.memory, sample_size)

    def __len__(self):
        return len(self.memory)

# the frozenlake Deep Q-Learning
class FrozenLakeDQL():
    # adjustable hyperparameters
    learning_rate = 0.001
    discount_factor = 0.9
    network_syn_rate = 10 # number of steps the agent takes before syncing the policy network and the target network
    replay_memory_size = 1000 # size of replay memory
    mini_batch_size = 32 # size of the training data set sampled from the replay memory

    # loss function for neural network
    loss_fn = nn.MSELoss() # using the Mean Squared Error (MSE) here

    # optimizer for neural network
    optimizer = None # initialize later

    # Train the Frozen Lake environment
    def train(self, episodes, render=False, is_slippery=False):
        # create FrozenLake environment
        env = gym.make('FrozenLake-v1', map_name="4x4", is_slippery=is_slippery, render_mode='human' if render else None)

        # get size of state and action spaces
        num_states = env.observation_space.n
        num_actions = env.action_space.n

        # create policy and target networks.
        policy_dqn = DQN(in_states=num_states, h1_nodes=num_states, out_actions=num_actions)
        target_dqn = DQN(in_states=num_states, h1_nodes=num_states, out_actions=num_actions)

        # make the target and policy networks the same (copy weights/biases from one policy network into the target network)
        target_dqn.load_state_dict(policy_dqn.state_dict())

        # policy network optimizer
        self.optimizer = torch.optim.Adam(policy_dqn.parameters(), lr=self.learning_rate)

        # track number of steps taken. used to sync policy and target networks
        step_count = 0

        epsilon = 1 # 100% of the time random actions are selected

        # initialize experience replay
        memory = ReplayMemory(self.replay_memory_size)

        # tracks rewards collected per episode
        rewards_per_episode = np.zeros(episodes)


        for i in range(episodes):
            state = env.reset()[0] # initialize state to 0
            terminated = False # True when agent falls in hole or reaches the goal
            truncated = False # true when agen takes more than 200 actions - not relevant in this small state space

            # agent navigates map until it falls into hole/ reaches goal (terminated), or has taken 200 actions (truncated)
            while (not terminated and not truncated):
                # e-greedy algorithm for action selection
                if random.random() < epsilon:
                    # select random action
                    action = env.action_space.sample()
                else:
                    # select best action using the function approximator i.e. the policy network
                    with torch.no_grad():
                        action = policy_dqn(self.state_to_dqn_input(state, num_states)).argmax().item()

                # execute action
                new_state, reward, terminated, truncated,_ = env.step(action)

                # save experience into memory
                memory.append((state, action, new_state, reward, terminated))

                # move to the next state
                state = new_state

                # increment step counter
                step_count += 1

            # keep track of rewards collected per episode
            if reward == 1:
                rewards_per_episode[i] = 1

            # check if enough experience has been collected and if at least 1 reward has been collected
            if len(memory) > self.mini_batch_size and np.sum(rewards_per_episode)>0:
                mini_batch = memory.sample(self.mini_batch_size)

                # train policy network
                self.optimize(mini_batch, policy_dqn, target_dqn)

                # decay epsilon
                epsilon = max(epsilon - 1 / episodes, 0)

                # copy policy network to target network after a certain number of steps
                if step_count > self.network_syn_rate:
                    target_dqn.load_state_dict(policy_dqn.state_dict())
                    step_count = 0

        # close environment
        env.close()

        # save policy 
        torch.save(policy_dqn.state_dict(), "frozen_lake_dql.pt")
            

    '''
    converts a state to a one-hot encoded tensor representation

    Parameters: state=1, num_states=16
    Return: tensor([0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
    
    '''
    def state_to_dqn_input(self, state:int, num_states:int) -> torch.Tensor:
        input_tensor = torch.zeros(num_states)
        input_tensor[state] = 1
        return input_tensor

    def optimize(self, mini_batch, policy_dqn, target_dqn):
        # get number of input nodes
        num_states = policy_dqn.fc1.in_features

        current_q_list = []
        target_q_list = []

        for state, action, new_state, reward, terminated in mini_batch:
            if terminated:
                # agent has either reached goal (reward=1) or fell into a hole (reward=0)
                # when in terminal state, the target q value should be set to the reward.
                target = torch.FloatTensor([reward])
            else:
                # calculate target q-value
                with torch.no_grad():
                    target = torch.FloatTensor(
                        reward + self.discount_factor * target_dqn(self.state_to_dqn_input(new_state, num_states)).max()
                    )


            # get the current set of Q values
            current_q = policy_dqn(self.state_to_dqn_input(state, num_states))
            current_q_list.append(current_q)

            # get the target set of Q values
            target_q = target_dqn(self.state_to_dqn_input(state, num_states))

            # adjust the specific action to the target that was computed
            target_q[action] = target
            target_q_list.append(target_q)

        # compute loss for the whole minibatch
        loss = self.loss_fn(torch.stack(current_q_list), torch.stack(target_q_list))

        # optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    # run the FrozenLake environment with the learned policy
    def test(self, episodes, is_slippery=False):
        # create Frozenlake instace
        env = gym.make('FrozenLake-v1', map_name="4x4", is_slippery=is_slippery, render_mode='human')
        num_states = env.observation_space.n
        num_actions = env.action_space.n

        # load learned policy
        policy_dqn = DQN(in_states=num_states, h1_nodes=num_states, out_actions=num_actions)
        policy_dqn.load_state_dict(torch.load("frozen_lake_dql.pt"))
        policy_dqn.eval() # switch model to evaluation mode

        for i in range(episodes):
            state = env.reset()[0]
            terminated = False
            truncated = False

            while (not terminated and not truncated):
                # select best action
                with torch.no_grad():
                    action = policy_dqn(self.state_to_dqn_input(state, num_states)).argmax().item()

                # execute action
                state, reward, terminated, truncated,_ = env.step(action)

        env.close()

if __name__ == '__main__':
    frozen_lake = FrozenLakeDQL()
    is_slippery = True
    # frozen_lake.train(5000, is_slippery=is_slippery)
    frozen_lake.test(10, is_slippery=is_slippery)