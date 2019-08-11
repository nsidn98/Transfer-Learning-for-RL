import argparse

def str2list(v):
    if not v:
        return v
    else:
        return [v_ for v_ in v.split(',')]


parser = argparse.ArgumentParser(description='PyTorch Soft Actor Critic')

# training params
parser.add_argument('--lr', type=float, default=1e-3,
                    help='Learning Rate for autoencoder')
parser.add_argument('--batch_size', type=int, default=20, 
                    help='batch size to sample')
parser.add_argument('--num_epochs', type=int, default=10,
                    help='number of epoochs to train the the autoencoder')
parser.add_argument('--save_path', type=str,default='AE_weights.pt',
                    help='path to save the autoencoder weights')
parser.add_argument('--container',type=int,default=0,
                    help='if trainining in container(path for data folder is different)')

# shape params
parser.add_argument('--orig_shape', type=int, default=64,
                    help='original image shape(env1)')
parser.add_argument('--target_shape', type=int, default=32,
                    help='target image shape(env2)')
parser.add_argument('--latent_shape', type=int, default=100,
                    help='latent representation shape')

# logs
parser.add_argument('--tensorboard', type=int, default=0,
                    help='whether to log in tensorboard')
parser.add_argument('--rm_runs', type=int, default=0,
                    help='whether to clean the previous logs(do it only if you feel previous logs are useless)')

parser.add_argument('--train', type=int, default=1,
                    help='whether to train or not')

# loss functions
parser.add_argument('--scale_loss', type=int, default=0,
                    help='whether to normalise each component of the loss term')
parser.add_argument('--latent_loss_type',type=str,default='L1',
                    help='type of loss for Latent representation. Choices=[L1,MSE]')
parser.add_argument('--recon_loss_type',type=str,default='MSE',
                    help='type of loss for reconstruction. Choices=[L1,MSE]')
parser.add_argument('--weight_paths',type=str2list,default='./Weights/autoencoder_1_shape.pt,./Weights/autoencoder_2_shape.pt')





args = parser.parse_args()