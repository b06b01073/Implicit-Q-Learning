import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
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
        self.dataset = self.init_datset()
      
        # network initialization 
        self.hidden_dim = args.hidden_dim
        self.Q_net = MLP(input_dim=self.action_dim + self.obs_dim, output_dim=1, hidden_dim=self.hidden_dim).to(device)
        self.target_Q_net = MLP(input_dim=self.action_dim + self.obs_dim, output_dim=1, hidden_dim=self.hidden_dim).to(device)
        self.hard_update(self.Q_net, self.target_Q_net)

        self.V_net = MLP(input_dim=self.obs_dim, output_dim=1, hidden_dim=self.hidden_dim).to(device)

        self.actor = Actor(obs_dim=self.obs_dim, action_dim=self.action_dim, hidden_dim=self.hidden_dim).to(device)

        # hyperparameters
        self.batch_size = args.batch_size
        self.gradient_step = args.gradient_step
        self.learning_rate = args.learning_rate
        self.gamma = args.discount_factor
        self.tau = args.tau  # expectile
        self.alpha = args.alpha # soft update coefs
        self.beta = args.beta # actor temp
        self.max_adv = args.max_adv

        # loss function and optimizer
        self.Q_optim = optim.Adam(params=self.Q_net.parameters(), lr=self.learning_rate)
        self.V_optim = optim.Adam(params=self.V_net.parameters(), lr=self.learning_rate)
        self.actor_optim = optim.Adam(params=self.actor.parameters(), lr=self.learning_rate)
        self.expectile_loss = expectile_loss
        self.mse_loss = nn.MSELoss()
        # self.awr_loss = awr_loss

    
    def train(self):
        self.TD_learning() # TD learning
        self.AWR() # advantage weighted regression


    def get_batch_data(self):
        indices = np.random.randint(len(self.dataset['observations']), size=self.batch_size)
        observations = self.dataset['observations'][indices]
        actions = self.dataset['actions'][indices]
        rewards = self.dataset['rewards'][indices]
        next_observations = self.dataset['next_observations'][indices]
        terminals = self.dataset['terminals'][indices]

        observations, actions, rewards, next_observations, terminals = map(lambda x: torch.from_numpy(x).to(device), (observations, actions, rewards, next_observations, terminals))

        return observations, actions, rewards, next_observations, terminals


    def TD_learning(self):
        for i in tqdm(range(self.gradient_step), desc='TD learning'):
            observations, actions, rewards, next_observations, terminals = self.get_batch_data()

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
            self.soft_update(self.Q_net, self.target_Q_net, self.alpha)
            writer.add_scalar('Value Loss', value_loss, i)
            writer.add_scalar('Q Loss', q_loss, i)

    
    def AWR(self):
        for i in tqdm(range(self.gradient_step), desc='extracing policy'):
            # caluate expoentiated advantage 
            observations, actions, _, _, _ = self.get_batch_data()
            adv = (self.target_Q_net(torch.cat((observations, actions), dim=1)) - self.V_net(observations))

            adv_weight = torch.clip(torch.exp(self.beta * (adv)), max=self.max_adv).detach()

            # get log prob from policy
            action_log_prob = self.actor.log_prob(actions, observations)
            
            # calculate loss and update actor
            actor_loss = (-adv_weight * action_log_prob).mean()

            self.actor_optim.zero_grad()
            actor_loss.backward()
            self.actor_optim.step()

            writer.add_scalar('Actor Loss', actor_loss, i)        


    def init_datset(self):  
        dataset = self.env.get_dataset()
        dataset = d4rl.qlearning_dataset(self.env)
        dataset['terminals'] = dataset['terminals'].astype(np.int64)

        return dataset
    

    def hard_update(self, src, target):
        for src_param, target_param in zip(src.parameters(), target.parameters()):
            target_param.data.copy_(src_param)


    def soft_update(self, src, target, alpha):
        for src_param, target_param in zip(src.parameters(), target.parameters()):
            target_param.data.copy_((1 - alpha) * target_param + alpha * src_param)
    

class MLP(nn.Module):
    def __init__(self,input_dim, output_dim, hidden_dim=256):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )


    def forward(self, x):
        return self.network(x)
    

class Actor(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.header = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)
    

    def forward(self, x):
        x = self.header(x)
        means = torch.exp(self.mean(x)) # this trick keeps the std always > 0
        stds = torch.exp(self.log_std(x)) # this trick keeps the std always > 0

        return means, stds
    

    def log_prob(self, actions, observations):
        means, stds = self.forward(observations)
        normal = Normal(means, stds)
        return normal.log_prob(actions)


def expectile_loss(input, target, tau):
    input = input.flatten()
    target = target.flatten()
    assert input.shape == target.shape, f'The shape of input and target is inconsisten. Input with shape {input.shape}, target with shape {target.shape}'  

    tau = torch.ones(size=input.shape).to(device) * tau 
    asymmtric_coef = torch.abs((tau - torch.gt(input, target).type(torch.LongTensor).to(device))).detach()

    return torch.mean(asymmtric_coef * ((input - target) ** 2))


# def awr_loss(q_value, baseline)