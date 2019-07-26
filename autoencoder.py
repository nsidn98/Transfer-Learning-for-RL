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
    INPUT_SHAPE = args.input_shape  
    LATENT_SHAPE = args.latent_shape
    NUM_EPOCHS = args.num_epochs
    BATCH_SIZE = args.batch_size

    # tensorboardx
    if args.tensorboard:
            print('Init tensorboardX')
            writer = SummaryWriter(log_dir='runs/{}'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")))

    # make states
    identity = np.identity(INPUT_SHAPE, dtype = np.float32)
    original_states = []
    flipped_states = []
    for i in range(INPUT_SHAPE):
        original_states.append(identity[i])
        flipped_states.append(identity[-(i+1)])

    original_states = np.array(original_states)
    flipped_states  = np.array(flipped_states)

    # CUDA compatability
    use_cuda = torch.cuda.is_available()
    device   = torch.device("cuda" if use_cuda else "cpu")

    # instantiate autoencoders
    autoencoder_1 = Autoencoder(INPUT_SHAPE,LATENT_SHAPE).to(device)
    autoencoder_2 = Autoencoder(INPUT_SHAPE,LATENT_SHAPE).to(device)
    criterion = nn.MSELoss()
    optimizer_1 = torch.optim.Adam(autoencoder_1.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    optimizer_2 = torch.optim.Adam(autoencoder_2.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)

    print('STARTING TRAINING...')
    print('#'*50)
    losses = []
    for epoch in tqdm(range(NUM_EPOCHS)):
        idx = np.random.randint(INPUT_SHAPE, size=BATCH_SIZE)
        batch_original_input = original_states[idx,:]
        batch_flipped_input  = flipped_states[idx,:]

        if args.noise:
            noise = np.random.normal(args.noise_mean,args.noise_std,INPUT_SHAPE)
            batch_original_input += noise
            batch_flipped_input  += noise

        
        orig_state = torch.FloatTensor(batch_original_input).to(device)
        flip_state = torch.FloatTensor(batch_flipped_input).to(device)

        s1,z1 = autoencoder_1(orig_state)
        s2,z2 = autoencoder_2(flip_state)
        
        reconstruction_loss_1 = criterion(orig_state,s1)
        reconstruction_loss_2 = criterion(flip_state.float(),s2)
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
                                }, 'Weights/autoencoder_1.pt')
                    torch.save({
                                'episode': epoch,
                                'model_state_dict': autoencoder_2.state_dict(),
                                'optimizer_state_dict': optimizer_2.state_dict(),
                                'loss': loss,
                                }, 'Weights/autoencoder_2.pt')
    # print('LOSS:',losses)
    # plt.plot(losses)
    # plt.grid()
    # plt.show()
                    
        

