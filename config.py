import argparse


parser = argparse.ArgumentParser(description='PyTorch Soft Actor Critic')

parser.add_argument('--num_epochs', type=int, default=1000,
                    help='number of episodes to train the agent')
parser.add_argument('--lr', type=float, default=1e-2, metavar='G',
                    help='learning rate (default: 0.0003)')
parser.add_argument('--seed', type=int, default=123, metavar='N',
                    help='random seed (default: 123)')
# parser.add_argument('--save_path', type=str, default='./Models/policy.pt', 
#                     help='file path to save the weights')
# parser.add_argument('--load_model', type=int, default=0, 
#                     help='bool to load model from pre-trained weights')
parser.add_argument('--tensorboard', type=int, default=1, 
                    help='Whether we want tensorboardX logging')
parser.add_argument('--batch_size', type=int, default=5, 
                    help='batch size to sample')
parser.add_argument('--latent_shape', type=int, default=80, 
                    help='batch size to sample')
parser.add_argument('--input_shape', type=int, default=20, 
                    help='batch size to sample')
parser.add_argument('--noise',type=int,default=0,
                    help='bool whether to add gaussian noise to the state or not')
parser.add_argument('--noise_mean',type=float,default=0,
                    help='gaussian noise mean')
parser.add_argument('--noise_std',type=float,default=0.01,
                    help='gaussian noise std')


                




args = parser.parse_args()