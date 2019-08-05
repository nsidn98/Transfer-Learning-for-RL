import argparse

def str2list(v):
    if not v:
        return v
    else:
        return [v_ for v_ in v.split(',')]


parser = argparse.ArgumentParser(description='PyTorch Soft Actor Critic')

parser.add_argument('--download', type=bool, default=False,
                    help='whether to download CIFAR dataset in directory ./root')
parser.add_argument('--lr', type=float, default=1e-3,
                    help='Learning Rate for autoencoder')
parser.add_argument('--batch_size', type=int, default=20, 
                    help='batch size to sample')
parser.add_argument('--num_epochs', type=int, default=10,
                    help='number of episodes to train the agent')
parser.add_argument('--save_path', type=str,default='AE_weights.pt')
parser.add_argument('--orig_shape', type=int, default=64,
                    help='number of episodes to train the agent')
parser.add_argument('--target_shape', type=int, default=32,
                    help='number of episodes to train the agent')
parser.add_argument('--latent_shape', type=int, default=100,
                    help='number of episodes to train the agent')
parser.add_argument('--tensorboard', type=int, default=0,
                    help='number of episodes to train the agent')



args = parser.parse_args()