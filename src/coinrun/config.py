import argparse

def str2list(v):
    if not v:
        return v
    else:
        return [v_ for v_ in v.split(',')]


parser = argparse.ArgumentParser(description='Coinrun environment')

parser.add_argument('--num_epochs', type=int, default=10,
                    help='number of episodes to train the agent')
parser.add_argument('--lr', type=float, default=1e-3, metavar='G',
                    help='learning rate (default: 0.0003)')
parser.add_argument('--tensorboard', type=int, default=0, 
                    help='Whether we want tensorboardX logging')
parser.add_argument('--batch_size', type=int, default=64, 
                    help='batch size to sample')
parser.add_argument('--latent_dim', type=int, default=25, 
                    help='latent representation shape')
parser.add_argument('--buffer_capacity', type=int, default=200, 
                    help='latent representation shape')
parser.add_argument('--weight_path', type=str, default='orig_env_AE.pkl', 
                    help='weight to save AE models')
parser.add_argument('--train_epochs', type=int, default=2, 
                    help='number of epochs to train per update')
parser.add_argument('--max_steps', type=int, default=1000000, 
                    help='number of random steps in env')
parser.add_argument('--num_envs', type=int, default=1, 
                    help='number of envs')
parser.add_argument('--xavier',type=int, default=1,
                    help='do savier uniform initialisation on weights')







# parser.add_argument('--weight_paths',type=str2list,default='./Weights/autoencoder_1_shape.pt,./Weights/autoencoder_2_shape.pt')


args = parser.parse_args()