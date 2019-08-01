'''
Test the autoencoder part of the states
'''
import numpy as np
import matplotlib.pyplot as plt
import gym
import torch
from typing import List, Tuple
from autoencoders.autoencoder import AutoencoderLinear


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

def main():
    env = gym.make('CartPole-v0')
    input_dim, output_dim = get_env_dim(env)
    autoencoder_1 = AutoencoderLinear(input_dim,int(2*input_dim))
    autoencoder_2 = AutoencoderLinear(input_dim,int(2*input_dim))
    checkpoint1 = torch.load('Weights/autoencoder_1.pt')
    checkpoint2 = torch.load('Weights/autoencoder_2.pt')
    autoencoder_1.load_state_dict(checkpoint1['model_state_dict'])
    autoencoder_2.load_state_dict(checkpoint2['model_state_dict'])
    autoencoder_1.eval(); autoencoder_2.eval()
    for i in range(2):
        state = env.reset()
        done = 0
        while not done:
            action = env.action_space.sample()
            next_state,r,done,_ = env.step(action)

            state_t = torch.from_numpy(state).float()
            flip_state_t = torch.from_numpy(np.flip(state).copy()).float()
            s1,z1 = autoencoder_1(state_t)
            s2,z2 = autoencoder_2(flip_state_t)
            # print(s2.detach().numpy(),np.flip(state))
            print(z1.detach().numpy())
            print(z2.detach().numpy())
            print()
            state = next_state


    # for i in range(FLAGS.n_episode):
    #     print("[Episode: {:5}] Reward: {:5} 𝜺-greedy: {:5.2f} Autoencoder Loss: {:5}".format(i + 1, r, eps,autoencoder_loss))

    #     rewards.append(r)

    #     if len(rewards) == rewards.maxlen:

    #         if np.mean(rewards) >= 200:
    #             print("Game cleared in {} games with {}".format(i + 1, np.mean(rewards)))
    #             break
    # autoencoder_agent.save(FLAGS)


if __name__ == '__main__':
    main()