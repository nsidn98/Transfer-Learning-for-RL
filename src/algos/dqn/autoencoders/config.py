import argparse

def str2list(v):
    if not v:
        return v
    else:
        return [v_ for v_ in v.split(',')]


parser = argparse.ArgumentParser(description='PyTorch Soft Actor Critic')

parser.add_argument('--num_epochs', type=int, default=1000,
                    help='number of episodes to train the agent')
parser.add_argument('--lr', type=float, default=1e-2, metavar='G',
                    help='learning rate (default: 0.0003)')
parser.add_argument('--seed', type=int, default=123, metavar='N',
                    help='random seed (default: 123)')
parser.add_argument('--tensorboard', type=int, default=1, 
                    help='Whether we want tensorboardX logging')
parser.add_argument('--batch_size', type=int, default=5, 
                    help='batch size to sample')
parser.add_argument('--latent_shape', type=int, default=80, 
                    help='batch size to sample')
parser.add_argument('--weight_paths',type=str2list,default='./Weights/autoencoder_1_shape.pt,./Weights/autoencoder_2_shape.pt')

# autoencoder
parser.add_argument('--autoencoder_type',type=str,default='linear',
                    help='Type of autoencoder, conv_deconv or linear')

# dummyEnv
parser.add_argument('--env',type=str,default='flipped',
                    help='whether flipped states or different shape state')
parser.add_argument('--input_shape_1', type=int, default=20, 
                    help='shape of state of env1')
parser.add_argument('--input_shape_2', type=int, default=20, 
                    help='shape of state of env2')
parser.add_argument('--noise',type=int,default=0,
                    help='bool whether to add gaussian noise to the state or not')
parser.add_argument('--noise_mean',type=float,default=0,
                    help='gaussian noise mean')
parser.add_argument('--noise_std',type=float,default=0.01,
                    help='gaussian noise std')

# shape_env
parser.add_argument('--random',type=int,default=0,
                    help='append random values to states in shape_env instead of zeros') 
parser.add_argument('--flip',type=int,default=0,
                    help='flip the state for shape_env')

# loss function
# total_loss = alpha_latent*latent_loss + alpha_recon1*recon1_loss + alpha_recon2*recon2_loss
parser.add_argument('--alpha_latent',type=float,default=1)
parser.add_argument('--alpha_recon1',type=float,default=1)
parser.add_argument('--alpha_recon2',type=float,default=1)


args = parser.parse_args()