import os
import random
import datetime
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from tensorboardX import SummaryWriter

from config import args
from dummyStates import flippedStates,shapeStates

class Autoencoder(nn.Module):
    def __init__(self,input_shape,latent_shape):
        '''
        input_shape: the state representation shape in the original environments
        latent_shape: the state shape we want to train the RL agent on
        Will return decode(encode(input)) and latent representation
        '''
        super(Autoencoder, self).__init__()
        self.encoder = nn.Linear(input_shape,latent_shape)
        self.decoder = nn.Linear(latent_shape,input_shape)

    def forward(self, x):
        z =   F.relu(self.encoder(x))
        out = self.decoder(z)
        return out, z


if __name__ == "__main__":
    # args stuff
    if not os.path.exists('./Weights'):
        os.makedirs('./Weights')
    LEARNING_RATE = args.lr
    INPUT_SHAPE_1 = args.input_shape_1
    INPUT_SHAPE_2 = args.input_shape_2
    LATENT_SHAPE = args.latent_shape
    NUM_EPOCHS = args.num_epochs
    BATCH_SIZE = args.batch_size

    # tensorboardx
    if args.tensorboard:
            print('Init tensorboardX')
            writer = SummaryWriter(log_dir='runs/{}'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")))

    # make states
    if args.env == 'flipped':
        assert INPUT_SHAPE_1 == INPUT_SHAPE_2
        env = flippedStates(args.input_shape_1)
        original_states, new_states = env.getStates()

    if args.env == 'shape':
        assert INPUT_SHAPE_1 + 2 == INPUT_SHAPE_2
        env = shapeStates(args.input_shape_1)
        original_states, new_states = env.getStates()
        # print(original_states.shape,new_states.shape)


    # original_states = np.array(original_states)
    # new_states  = np.array(new_states)

    # CUDA compatability
    use_cuda = torch.cuda.is_available()
    device   = torch.device("cuda" if use_cuda else "cpu")

    # instantiate autoencoders
    autoencoder_1 = Autoencoder(INPUT_SHAPE_1,LATENT_SHAPE).to(device)
    autoencoder_2 = Autoencoder(INPUT_SHAPE_2,LATENT_SHAPE).to(device)
    criterion = nn.MSELoss()
    optimizer_1 = torch.optim.Adam(autoencoder_1.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    optimizer_2 = torch.optim.Adam(autoencoder_2.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)

    print('STARTING TRAINING...')
    print('#'*50)
    losses = []
    for epoch in tqdm(range(NUM_EPOCHS)):
        idx = np.random.randint(INPUT_SHAPE_1, size=BATCH_SIZE)
        batch_original_input = original_states[idx,:]
        batch_new_input  = new_states[idx,:]

        if args.noise:
            noise = np.random.normal(args.noise_mean,args.noise_std,INPUT_SHAPE_1)
            batch_original_input += noise
            batch_new_input  += noise

        
        orig_state = torch.FloatTensor(batch_original_input).to(device)
        new_state  = torch.FloatTensor(batch_new_input).to(device)

        s1,z1 = autoencoder_1(orig_state)
        s2,z2 = autoencoder_2(new_state)
        
        reconstruction_loss_1 = criterion(orig_state,s1)
        reconstruction_loss_2 = criterion(new_state.float(),s2)
        latent_loss = criterion(z1,z2)

        loss = latent_loss + reconstruction_loss_1 + reconstruction_loss_2
        losses.append(loss.detach().cpu().numpy())
        if args.tensorboard:
            writer.add_scalar('Autoencoder_1_Loss',reconstruction_loss_1.item(),epoch)
            writer.add_scalar('Autoencoder_2_Loss',reconstruction_loss_2.item(),epoch)
            writer.add_scalar('Latent_Loss',latent_loss.item(),epoch)
            writer.add_scalar('Total_Loss',loss.item(),epoch)
            
            

        optimizer_1.zero_grad()
        loss.backward(retain_graph=True)
        optimizer_1.step()

        optimizer_2.zero_grad()
        loss.backward()
        optimizer_2.step()
        

        if epoch % 100 == 0 and epoch !=0:
                    # print('Saving model...')
                    torch.save({
                                'episode': epoch,
                                'model_state_dict': autoencoder_1.state_dict(),
                                'optimizer_state_dict': optimizer_1.state_dict(),
                                'loss': loss,
                                }, args.weight_paths[0])
                    torch.save({
                                'episode': epoch,
                                'model_state_dict': autoencoder_2.state_dict(),
                                'optimizer_state_dict': optimizer_2.state_dict(),
                                'loss': loss,
                                }, args.weight_paths[1])
    # print('LOSS:',losses)
    plt.plot(losses)
    plt.grid()
    plt.title('Total Loss')
    plt.xlabel('epochs')
    plt.ylabel('Loss')
    plt.show()
                    
        

