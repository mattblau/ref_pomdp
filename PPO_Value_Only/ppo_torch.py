import os
import numpy as np
import torch as T
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical

class PPOMemory:
    def __init__(self, batch_size):
        self.states = []
        self.vals = []
        self.actions = []
        self.rewards = []
        self.dones = []

        self.batch_size = batch_size

    def generate_batches(self):
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i+self.batch_size] for i in batch_start]

        return np.array(self.states),\
                np.array(self.actions),\
                np.array(self.vals),\
                np.array(self.rewards),\
                np.array(self.dones),\
                batches

    def store_memory(self, state, action, vals, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.vals.append(vals)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear_memory(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.vals = []

class CriticNetwork(nn.Module):
    def __init__(self, input_dims, alpha, fc1_dims=256, fc2_dims=256,
            chkpt_dir='tmp/ppo'):
        super(CriticNetwork, self).__init__()

        self.checkpoint_file = os.path.join(chkpt_dir, 'critic_torch_ppo')
        self.critic = nn.Sequential(
                nn.Linear(*input_dims, fc1_dims),
                nn.ReLU(),
                nn.Linear(fc1_dims, fc2_dims), 
                nn.ReLU(),
                nn.Linear(fc1_dims, fc2_dims), # Added an additional layer
                nn.ReLU(),
                nn.Linear(fc2_dims, 1)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        value = self.critic(state)
        value = value

        return value

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))

class LearningAgent:
    def __init__(self, n_actions, input_dims, gamma=0.99, alpha=0.0003, gae_lambda=0.95,
            policy_clip=0.2, batch_size=64, n_epochs=10):
        self.gamma = gamma
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.gae_lambda = gae_lambda

        self.critic = CriticNetwork(input_dims, alpha)
        self.memory = PPOMemory(batch_size)
       
    def remember(self, state, action, vals, reward, done):
        self.memory.store_memory(state, action, vals, reward, done)

    def save_models(self):
        print('... saving models ...')
        self.critic.save_checkpoint()

    def load_models(self):
        print('... loading models ...')
        self.critic.load_checkpoint()

    def learn(self):
        with open('critic_losses.txt', 'a') as log_file:
            for _ in range(self.n_epochs):
                state_arr, action_arr, vals_arr,\
                reward_arr, dones_arr, batches = \
                        self.memory.generate_batches()

                values = vals_arr
                advantage = np.zeros(len(reward_arr), dtype=np.float32)

                for t in range(len(reward_arr)-1):
                    discount = 1
                    a_t = 0
                    for k in range(t, len(reward_arr)-1):
                        a_t += discount*(reward_arr[k] + self.gamma*values[k+1]*\
                                (1-int(dones_arr[k])) - values[k])
                        discount *= self.gamma*self.gae_lambda
                    advantage[t] = a_t
                advantage = T.tensor(advantage).to(self.critic.device)

                values = T.tensor(values).to(self.critic.device)
                for batch in batches:
                    states = T.tensor(state_arr[batch], dtype=T.float).to(self.critic.device)
                
                    critic_value = self.critic(states)

                    critic_value = T.squeeze(critic_value)

                    returns = advantage[batch] + values[batch]
                    critic_loss = (returns-critic_value)**2
                    critic_loss = critic_loss.mean()
                    self.critic.optimizer.zero_grad()
                    critic_loss.backward()
                    self.critic.optimizer.step()
                
            log_file.write(f'{critic_loss.item()}\n')

            self.memory.clear_memory()               

