"""
DQN in PyTorch
"""
import os
import torch
import torch.nn as nn
import numpy as np
import random
from collections import namedtuple
from collections import deque
from typing import List, Tuple
import matplotlib.pyplot as plt

import gym
from gym import spaces
from gym.utils import seeding

from arguments import args
from autoencoders.autoencoder import AutoencoderConv
from mountaincar import MountainCarEnv

# from autoencoders.config import args


# CUDA compatability
use_cuda = torch.cuda.is_available()
device   = torch.device("cuda" if use_cuda else "cpu")

# seed
np_r, seed = seeding.np_random(None)

def preprocessImg(img):
    '''
    Convert to [1,c,h,w] from [h,w,c]
    '''
    # img = img.astype(np.float64)
    img = np.transpose(img, (2,0,1))
    img = np.expand_dims(img,0)
    return img

class DQN(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int) -> None:
        """DQN Network
        Args:
            input_dim (int): `state` dimension.
                `state` is 2-D tensor of shape (n, input_dim)
            output_dim (int): Number of actions.
                Q_value is 2-D tensor of shape (n, output_dim)
            hidden_dim (int): Hidden dimension in fc layer
        """
        super(DQN, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.PReLU()
        )

        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.PReLU()
        )

        self.final = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns a Q_value
        Args:
            x (torch.Tensor): `State` 2-D tensor of shape (n, input_dim)
        Returns:
            torch.Tensor: Q_value, 2-D tensor of shape (n, output_dim)
        """
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.final(x)

        return x


Transition = namedtuple("Transition",
                        field_names=["state", "action", "reward", "next_state", "done"])
States_Buffer = namedtuple("States",field_names=["orig_state","new_state"])


class ReplayMemory(object):

    def __init__(self, capacity: int) -> None:
        """Replay memory class
        Args:
            capacity (int): Max size of this memory
        """
        self.capacity = capacity
        self.cursor = 0
        self.memory = []

    def push(self,
             state: np.ndarray,
             action: int,
             reward: int,
             next_state: np.ndarray,
             done: bool) -> None:
        """Creates `Transition` and insert
        Args:
            state (np.ndarray): 1-D tensor of shape (input_dim,)
            action (int): action index (0 <= action < output_dim)
            reward (int): reward value
            next_state (np.ndarray): 1-D tensor of shape (input_dim,)
            done (bool): whether this state was last step
        """
        if len(self) < self.capacity:
            self.memory.append(None)

        self.memory[self.cursor] = Transition(state,
                                              action, reward, next_state, done)
        self.cursor = (self.cursor + 1) % self.capacity

    def pop(self, batch_size: int) -> List[Transition]:
        """Returns a minibatch of `Transition` randomly
        Args:
            batch_size (int): Size of mini-bach
        Returns:
            List[Transition]: Minibatch of `Transition`
        """
        return random.sample(self.memory, batch_size)

    def __len__(self) -> int:
        """Returns the length """
        return len(self.memory)


class stateBuffer(object):
    '''
    Buffer for storing states for autoencoders
    '''
    def __init__(self, capacity: int) -> None:
        """Replay memory class
        Args:
            capacity (int): Max size of this memory
        """
        self.capacity = capacity
        self.cursor = 0
        self.memory = []

    def push(self,
             orig_state: np.ndarray,
             new_state : np.ndarray,
             ) -> None:
        """Creates `Transition` and insert
        Args:
            orig_state (np.ndarray): 3-D tensor of shape (input_dim,)
            next_state (np.ndarray): 3-D tensor of shape (input_dim,)
        """
        if len(self) < self.capacity:
            self.memory.append(None)

        self.memory[self.cursor] = States_Buffer(orig_state, new_state)
        self.cursor = (self.cursor + 1) % self.capacity

    def pop(self, batch_size: int) -> List[States_Buffer]:
        """Returns a minibatch of `Transition` randomly
        Args:
            batch_size (int): Size of mini-bach
        Returns:
            List[Transition]: Minibatch of `Transition`
        """
        return random.sample(self.memory, batch_size)

    def __len__(self) -> int:
        """Returns the length """
        return len(self.memory)


class Agent(object):

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int) -> None:
        """Agent class that choose action and train
        Args:
            input_dim (int): input dimension
            output_dim (int): output dimension
            hidden_dim (int): hidden dimension
        """
        self.dqn = DQN(input_dim, output_dim, hidden_dim)
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.loss_fn = nn.MSELoss()
        self.optim = torch.optim.Adam(self.dqn.parameters())

    def _to_variable(self, x: np.ndarray) -> torch.Tensor:
        """torch.Variable syntax helper
        Args:
            x (np.ndarray): 2-D tensor of shape (n, input_dim)
        Returns:
            torch.Tensor: torch variable
        """
        return torch.autograd.Variable(torch.Tensor(x))

    def get_action(self, states: np.ndarray, eps: float) -> int:
        """Returns an action
        Args:
            states (np.ndarray): 2-D tensor of shape (n, input_dim)
            eps (float): 𝜺-greedy for exploration
        Returns:
            int: action index
        """
        if np.random.rand() < eps:
            return np.random.choice(self.output_dim)
        else:
            self.dqn.train(mode=False)
            scores = self.get_Q(states)
            _, argmax = torch.max(scores.data, 1)
            return int(argmax.numpy())

    def get_Q(self, states: np.ndarray) -> torch.FloatTensor:
        """Returns `Q-value`
        Args:
            states (np.ndarray): 2-D Tensor of shape (n, input_dim)
        Returns:
            torch.FloatTensor: 2-D Tensor of shape (n, output_dim)
        """
        states = self._to_variable(states.reshape(-1, self.input_dim))
        self.dqn.train(mode=False)
        return self.dqn(states)

    def train(self, Q_pred: torch.FloatTensor, Q_true: torch.FloatTensor) -> float:
        """Computes `loss` and backpropagation
        Args:
            Q_pred (torch.FloatTensor): Predicted value by the network,
                2-D Tensor of shape(n, output_dim)
            Q_true (torch.FloatTensor): Target value obtained from the game,
                2-D Tensor of shape(n, output_dim)
        Returns:
            float: loss value
        """
        self.dqn.train(mode=True)
        self.optim.zero_grad()
        loss = self.loss_fn(Q_pred, Q_true)
        loss.backward()
        self.optim.step()
        return loss

class Darling(object):
    # DisentAngled Representation LearnING
    # Parody of DARLA :P
    def __init__(self,tensorboard=0):
        self.autoencoder1  = AutoencoderConv()
        self.autoencoder2  = AutoencoderConv()
        self.criterion     = nn.MSELoss()
        self.optimizer1 = torch.optim.Adam(self.autoencoder1.parameters(), lr=1e-3, weight_decay=1e-5)
        self.optimizer2 = torch.optim.Adam(self.autoencoder2.parameters(), lr=1e-3, weight_decay=1e-5)
        self.losses = []
        self.tensorboard = tensorboard
        self.loss = 0

    def train(self,minibatch: List[Transition]):
        orig_states = np.vstack([x.orig_state for x in minibatch])
        new_states  = np.vstack([x.new_state for x in minibatch])

        
        orig_states = torch.FloatTensor(orig_states)
        new_states = torch.FloatTensor(new_states)
        
        s1,z1 = self.autoencoder1(orig_states)
        s2,z2 = self.autoencoder2(new_states)
        
        reconstruction_loss1 = self.criterion(orig_states,s1)
        reconstruction_loss2 = self.criterion(new_states,s2)
        latent_loss = self.criterion(z1,z2)

        loss = latent_loss + reconstruction_loss1 + reconstruction_loss2
        self.losses.append(loss.detach().numpy())
        self.loss = np.copy(loss.detach().numpy())

        self.optimizer1.zero_grad()
        loss.backward(retain_graph=True)
        self.optimizer1.step()

        self.optimizer2.zero_grad()
        loss.backward()
        self.optimizer2.step()
    
    def save(self,args):
        if not os.path.exists('./Weights'):
            os.makedirs('./Weights')
        torch.save({
                    'model_state_dict': self.autoencoder1.state_dict(),
                    'optimizer_state_dict': self.optimizer1.state_dict(),
                    }, args.weight_paths[0])
        torch.save({
                    'model_state_dict': self.autoencoder2.state_dict(),
                    'optimizer_state_dict': self.optimizer2.state_dict(),
                    }, args.weight_paths[1])


def train_helper(agent: Agent, minibatch: List[Transition], gamma: float) -> float:
    """Prepare minibatch and train them
    Args:
        agent (Agent): Agent has `train(Q_pred, Q_true)` method
        minibatch (List[Transition]): Minibatch of `Transition`
        gamma (float): Discount rate of Q_target
    Returns:
        float: Loss value
    """
    states = np.vstack([x.state for x in minibatch])
    actions = np.array([x.action for x in minibatch])
    rewards = np.array([x.reward for x in minibatch])
    next_states = np.vstack([x.next_state for x in minibatch])
    done = np.array([x.done for x in minibatch])

    Q_predict = agent.get_Q(states)
    Q_target = Q_predict.clone().data.numpy()
    Q_target[np.arange(len(Q_target)), actions] = rewards + gamma * np.max(agent.get_Q(next_states).data.numpy(), axis=1) * ~done
    Q_target = agent._to_variable(Q_target)

    return agent.train(Q_predict, Q_target)



def play_episode(orig_env: MountainCarEnv,
                 new_env: MountainCarEnv,
                 agent: Agent,
                 autoencoder_agent: Darling,
                 replay_memory: ReplayMemory,
                 state_memory:  stateBuffer,
                 eps: float,
                 batch_size: int) -> int:
    """Play an epsiode and train
    Args:
        env (gym.Env): gym environment (CartPole-v0)
        agent (Agent): agent will train and get action
        replay_memory (ReplayMemory): trajectory is saved here
        eps (float): 𝜺-greedy for exploration
        batch_size (int): batch size
    Returns:
        int: reward earned in this episode
    """
    init_state = np.array([np_r.uniform(low=-0.6, high=-0.4), 0])

    # initialise both envs to same state
    s = orig_env.reset(init_state)
    new_env.reset(init_state)

    done = False
    total_reward = 0

    while not done:

        a = agent.get_action(s, eps)
        s2, r, done, _ = orig_env.step(a)
        _,_,_,_ = new_env.step(a)

        # get frames for both environments
        orig_img = orig_env.render(mode='rgb_array')
        new_img = new_env.render(mode='rgb_array')

        orig_img = preprocessImg(orig_img)
        new_img = preprocessImg(new_img)

        total_reward += r

        if done:
            r = -1
        replay_memory.push(s, a, r, s2, done)

        # state_memory.push(s,np.flip(s))
        # push frames for both envs in buffer
        state_memory.push(orig_img,new_img)

        if len(replay_memory) > batch_size:

            minibatch = replay_memory.pop(batch_size)
            train_helper(agent, minibatch, args.gamma)

            minibatch_autoencoder = state_memory.pop(batch_size)
            autoencoder_agent.train(minibatch_autoencoder)

        s = s2

    return total_reward


def get_env_dim(env: gym.Env) -> Tuple[int, int]:
    """Returns input_dim & output_dim
    Args:
        env (gym.Env): gym Environment (CartPole-v0)
    Returns:
        int: input_dim
        int: output_dim
    """
    input_dim = env.observation_space.shape[0]
    output_dim = env.action_space.n

    return input_dim, output_dim


def epsilon_annealing(epsiode: int, max_episode: int, min_eps: float) -> float:
    """Returns 𝜺-greedy
    1.0---|\
          | \
          |  \
    min_e +---+------->
              |
              max_episode
    Args:
        epsiode (int): Current episode (0<= episode)
        max_episode (int): After max episode, 𝜺 will be `min_eps`
        min_eps (float): 𝜺 will never go below this value
    Returns:
        float: 𝜺 value
    """

    slope = (min_eps - 1.0) / max_episode
    return max(slope * epsiode + 1.0, min_eps)


def main():
    """Main
    """
    try:
        # env = gym.make(FLAGS.env)
        orig_env = MountainCarEnv()
        new_env = MountainCarEnv(color=[1,0,0,0.5])
        # env = gym.wrappers.Monitor(env, directory="monitors", force=True)
        rewards = deque(maxlen=100)
        input_dim, output_dim = get_env_dim(orig_env)
        agent = Agent(input_dim, output_dim, args.hidden_dim)
        replay_memory = ReplayMemory(args.capacity)
        state_memory = stateBuffer(args.capacity)
        autoencoder_agent = Darling()

        for i in range(args.n_episode):
            eps = epsilon_annealing(i, args.max_episode, args.min_eps)
            r = play_episode(orig_env, new_env, agent,autoencoder_agent, replay_memory,state_memory, eps, args.batch_size)
            autoencoder_loss = autoencoder_agent.loss
            print("[Episode: {:5}] Reward: {:5} 𝜺-greedy: {:5.2f} Autoencoder Loss: {:5}".format(i + 1, r, eps,autoencoder_loss))

            rewards.append(r)

            if len(rewards) == rewards.maxlen:

                if np.mean(rewards) >= 200:
                    print("Game cleared in {} games with {}".format(i + 1, np.mean(rewards)))
                    break
        autoencoder_agent.save(args)
        # plt.plot(autoencoder_agent.losses)
        # plt.grid()
        # plt.show()
    finally:
        orig_env.close()
        new_env.close()


if __name__ == '__main__':
    main()