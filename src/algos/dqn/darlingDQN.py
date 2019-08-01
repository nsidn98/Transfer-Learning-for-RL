"""
DQN in PyTorch
"""
import os
import torch
import torch.nn as nn
import numpy as np
import random
import gym
from collections import namedtuple
from collections import deque
from typing import List, Tuple
import matplotlib.pyplot as plt

from arguments import FLAGS
from autoencoders.autoencoder import AutoencoderLinear
# from autoencoders.config import args


# CUDA compatability
use_cuda = torch.cuda.is_available()
device   = torch.device("cuda" if use_cuda else "cpu")

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



class Agent(object):

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int, test: int, load_path: str) -> None:
        """Agent class that choose action and train
        Args:
            input_dim (int): input dimension
            output_dim (int): output dimension
            hidden_dim (int): hidden dimension
        """
        self.dqn = DQN(input_dim, output_dim, hidden_dim)
        self.input_dim = input_dim
        self.output_dim = output_dim
        if os.path.exists(load_path):
            print('LOADING DQN MODEL')
            checkpoint = torch.load(load_path)
            self.dqn.load_state_dict(checkpoint['model_state_dict'])
        if not test:
            self.loss_fn = nn.MSELoss()
            self.optim = torch.optim.Adam(self.dqn.parameters())
        if test:
            self.dqn.eval()

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

    def saveModel(self,save_path):
        print('SAVING DQN MODEL')
        torch.save({
                'model_state_dict': self.dqn.state_dict(),
                }, save_path)

class LatentTransform(object):
    # convert the original state to latent representation
    def __init__(self,input_shape,latent_shape,autoencoder_type='original'):
        self.autoencoder  = AutoencoderLinear(input_shape,latent_shape)
        if autoencoder_type == 'original':
            checkpoint = torch.load('Weights/autoencoder_1.pt')
        if autoencoder_type == 'target':
            checkpoint = torch.load('Weights/autoencoder_2.pt')

        self.autoencoder.load_state_dict(checkpoint['model_state_dict'])
        self.autoencoder.eval()
    
    def getLatent(self,state):
        state_c = np.copy(state) # so that no changes are made in the original state
        state_t = torch.FloatTensor(state_c)
        s,z = self.autoencoder(state_t)
        s = s.detach().numpy()
        z = z.detach().numpy()
        return s,z

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



def play_episode(env: gym.Env,
                 agent: Agent,
                 transformer: LatentTransform,
                 replay_memory: ReplayMemory,
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
    s = env.reset()
    done = False
    total_reward = 0

    while not done:
        # _l subscript for latent representation
        s_l,z_l = transformer.getLatent(s)
        a = agent.get_action(z_l, eps)
        s2, r, done, _ = env.step(a)

        s2_l,z2_l = transformer.getLatent(s2)
        total_reward += r

        if done:
            r = -1
        # replay_memory.push(s, a, r, s2, done)
        replay_memory.push(z_l, a, r, z2_l, done)


        if len(replay_memory) > batch_size:

            minibatch = replay_memory.pop(batch_size)
            train_helper(agent, minibatch, FLAGS.gamma)


        s = s2

    return total_reward

def test_episode(env: gym.Env,
                 agent: Agent,
                 transformer: LatentTransform,
                 state_type: str) -> int:
    """Play an epsiode and train
    Args:
        env (gym.Env): gym environment (CartPole-v0)
        agent (Agent): agent will train and get action
        replay_memory (ReplayMemory): trajectory is saved here
        eps (float): 𝜺-greedy for exploration
        state_type(str): either 'original' or 'target'(for flipped)
    Returns:
        int: reward earned in this episode
    """
    s = env.reset()
    done = False
    total_reward = 0

    while not done:
        # _l subscript for latent representation
        if state_type=='target':
            print('FLIPPING')
            s = np.flip(s)
        s_l,z_l = transformer.getLatent(s)
        a = agent.get_action(z_l, 0)
        s2, r, done, _ = env.step(a)

        s2_l,z2_l = transformer.getLatent(s2)
        total_reward += r

        if done:
            r = -1

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
    rewards_arr = []
    try:
        env = gym.make(FLAGS.env)
        # env = gym.wrappers.Monitor(env, directory="monitors", force=True)
        rewards = deque(maxlen=100)
        input_dim_orig, output_dim = get_env_dim(env)
        input_dim = FLAGS.latent_shape # 8
        agent = Agent(input_dim, output_dim, FLAGS.hidden_dim,FLAGS.test,FLAGS.weight_pathDQN)
        transformer = LatentTransform(input_dim_orig,input_dim,FLAGS.state_type)

        if FLAGS.test:
            rewards_arr = []
            for i in range(FLAGS.n_episode):
                r = test_episode(env,agent,transformer,FLAGS.state_type)
                rewards_arr.append(r)
                print("[Episode: {:5}] Reward: {:5}".format(i + 1, r))
            plt.plot(rewards_arr)
            plt.grid()
            plt.xlabel('Episodes')
            plt.ylabel('Rewards')
            plt.title('Rewards for flipped environment.\n Testing with DQN')
            plt.savefig('Outputs/DQN_flip_test.png')
            plt.show()
            np.save('Outputs/dqn_flip_test.npy',np.array(rewards_arr))

        if not FLAGS.test:
            replay_memory = ReplayMemory(FLAGS.capacity)
            for i in range(FLAGS.n_episode):
                eps = epsilon_annealing(i, FLAGS.max_episode, FLAGS.min_eps)
                r = play_episode(env, agent,transformer, replay_memory, eps, FLAGS.batch_size)
                print("[Episode: {:5}] Reward: {:5} 𝜺-greedy: {:5.2f}".format(i + 1, r, eps))

                rewards.append(r)
                rewards_arr.append(r)

                if len(rewards) == rewards.maxlen:

                    if np.mean(rewards) >= 200:
                        print("Game cleared in {} games with {}".format(i + 1, np.mean(rewards)))
                        break

            # save stuff
            agent.saveModel(FLAGS.weight_pathDQN)
            if not os.path.exists('./Outputs'):
                os.makedirs('./Outputs')
            plt.plot(rewards_arr)
            plt.grid()
            plt.xlabel('Episodes')
            plt.ylabel('Rewards')
            plt.title('Rewards for original environment.\n Training with DQN')
            plt.savefig('Outputs/DQN_orig.png')
            plt.show()
            np.save('Outputs/dqn_orig.npy',np.array(rewards_arr))
    finally:
        env.close()


if __name__ == '__main__':
    main()