import torch
import torch.nn as nn
import torch.optim as optim
import gym
import d4rl
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

writer = SummaryWriter()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Training with {device}')

class IQLAgent:
    def __init__(self, args):
        # meta data and env
        self.env = gym.make(args.env_name)
        self.action_dim = self.env.action_space.shape[0]
        self.obs_dim = self.env.observation_space.shape[0]

        # get dataset
        self.dataset = self.__init_datset()
      
        # network initialization 
        self.Q_net = MLP(input_dim=self.action_dim + self.obs_dim, output_dim=1).to(device)
        self.target_Q_net = MLP(input_dim=self.action_dim + self.obs_dim, output_dim=1).to(device)
        self.__hard_update(self.Q_net, self.target_Q_net)

        self.V_net = MLP(input_dim=self.obs_dim, output_dim=1).to(device)

        # hyperparameters
        self.batch_size = args.batch_size
        self.gradient_step = args.gradient_step
        self.learning_rate = args.learning_rate
        self.gamma = args.discount_factor
        self.tau = args.tau  # expectile
        self.alpha = args.alpha # soft update coef

        # loss function and optimizer
        self.Q_optim = optim.Adam(params=self.Q_net.parameters(), lr=self.learning_rate)
        self.V_optim = optim.Adam(params=self.V_net.parameters(), lr=self.learning_rate)
        self.expectile_loss = expectile_loss
        self.mse_loss = nn.MSELoss()

    
    def train(self):
        for i in tqdm(range(self.gradient_step)):
            indices = np.random.randint(len(self.dataset['observations']), size=self.batch_size)
            observations = self.dataset['observations'][indices]
            actions = self.dataset['actions'][indices]
            rewards = self.dataset['rewards'][indices]
            next_observations = self.dataset['next_observations'][indices]
            terminals = self.dataset['terminals'][indices]

            observations, actions, rewards, next_observations, terminals = map(lambda x: torch.from_numpy(x).to(device), (observations, actions, rewards, next_observations, terminals))

            # update value network
            target = self.target_Q_net(torch.cat((observations, actions), dim=1)).detach()
            state_value = self.V_net(observations)
            value_loss = self.expectile_loss(input=state_value, target=target, tau=self.tau)

            self.V_optim.zero_grad()
            value_loss.backward()
            self.V_optim.step()

            # update action value network
            target = rewards + self.gamma * (1 - terminals) * self.V_net(next_observations).squeeze()
            state_action_value = self.Q_net(torch.cat((observations, actions), dim=1)).flatten()
            q_loss = self.mse_loss(target.detach(), state_action_value)

            self.Q_optim.zero_grad()
            q_loss.backward()
            self.Q_optim.step()

            # update target network
            self.__soft_update(self.Q_net, self.target_Q_net, self.alpha)
            writer.add_scalar('ValueLoss', value_loss, i)
            writer.add_scalar('QLoss', q_loss, i)


    def __init_datset(self):  
        dataset = self.env.get_dataset()
        dataset = d4rl.qlearning_dataset(self.env)
        dataset['terminals'] = dataset['terminals'].astype(np.int64)

        return dataset
    

    def __hard_update(self, src, target):
        for src_param, target_param in zip(src.parameters(), target.parameters()):
            target_param.data.copy_(src_param)


    def __soft_update(self, src, target, alpha):
        for src_param, target_param in zip(src.parameters(), target.parameters()):
            target_param.data.copy_((1 - alpha) * target_param + alpha * src_param)
    

class MLP(nn.Module):
    def __init__(self,input_dim, output_dim, hidden_dim=256):
        super().__init__()
        # self.input_dim = 


        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.network(x)
    
def expectile_loss(input, target, tau):
    input = input.flatten()
    target = target.flatten()
    assert input.shape == target.shape, f'The shape of input and target is inconsisten. Input with shape {input.shape}, target with shape {target.shape}'  

    tau = torch.ones(size=input.shape).to(device) * tau 
    asymmtric_coef = torch.abs((tau - torch.gt(input, target).type(torch.LongTensor).to(device))).detach()

    return torch.mean(asymmtric_coef * ((input - target) ** 2))


