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

class AutoEncoder(nn.Module):
    def __init__(self,args,image_dim=64,latent_dim=100):
        super(AutoEncoder, self).__init__()
        self.args = args
        self.image_dim = image_dim # a 28x28 image corresponds to 4 on the FC layer, a 64x64 image corresponds to 13
        self.latent_dim = latent_dim
        self.latentLayerShape = self.getShape() # shape after all convs
        # self.batch_size = 50
        # shape before linear(latent)
        # wxw --> (w-3)x(w-3) --> (w-6)x(w-6) --> (w/2-4)x(w/2-4) --> (w/4-3)x(w/4-3)
        # so 32x32 becomes (32/4-3)x(32/4-3) = 5x5
        # 28x28 becomes 4x4
        # 64x64 becomes 13x13
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=4, stride=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=4, stride=2),
            nn.ReLU()
            )
        self.fc1 = nn.Linear(32*self.latentLayerShape*self.latentLayerShape, self.latent_dim)
        self.fc2 = nn.Linear(self.latent_dim, 32*self.latentLayerShape*self.latentLayerShape)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 32, kernel_size=4, stride=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=1),
	        )
        if self.args.xavier:
            print('Initialising Xavier Weights\n')
            self.apply(self.weights_init_)
        print("Latent Layer shape after conv(NOT latent representation dimension):",self.latentLayerShape)

    def forward(self, x):
        # uncomment to add noise to the original image
        # n = x.size()[0]
        # noise = Variable(self.noise_scale * torch.randn(n, 1, self.image_dim, self.image_dim))
        # x = torch.add(x, noise)

        z = self.encoder(x)
        z = z.view(-1, 32*self.latentLayerShape*self.latentLayerShape)
        z = self.fc1(z)
        x_hat = self.fc2(z)
        x_hat = x_hat.view(-1, 32, self.latentLayerShape, self.latentLayerShape)
        x_hat = self.decoder(x_hat)
        x_hat = torch.sigmoid(x_hat)
        # x_hat = F.tanh(x_hat)

        return z, x_hat

    def encode(self, x):
        #x = x.unsqueeze(0)
        z, x_hat = self.forward(x)

        return z

    def getShape(self):
        # w --> w-3 --> w-6 --> w/2-4 --> w/4-3
        # gives the shape of the linear(latent) layer  
        # use w_out = (w-k_size+2*padding)/stride + 1  
        return int(self.image_dim/4-3) # if four conv layers
        # return int(self.image_dim/2-4) # if three conv layers

    def weights_init_(self,m):
        if isinstance(m, nn.Linear):
            # print('Linear Xavier')
            nn.init.xavier_uniform_(m.weight, gain=1)
            nn.init.constant_(m.bias, 0)
        if isinstance(m,nn.Conv2d):
            # print('Conv Xavier')
            nn.init.xavier_uniform_(m.weight)
        if isinstance(m, nn.ConvTranspose2d):
            # print('Convtranspose Xavier')
            nn.init.xavier_uniform_(m.weight)



        

if __name__ == "__main__":
    img = torch.rand((1,3,64,64))
    a = AutoEncoder()
    print(a.latentLayerShape)
    z,x =a(img)
    # from config import args
    # from dummyStates import flippedStates,shapeStates

    # # args stuff
    # if not os.path.exists('./Weights'):
    #     os.makedirs('./Weights')
    # LEARNING_RATE = args.lr
    # INPUT_SHAPE_1 = args.input_shape_1
    # INPUT_SHAPE_2 = args.input_shape_2
    # LATENT_SHAPE = args.latent_shape
    # NUM_EPOCHS = args.num_epochs
    # BATCH_SIZE = args.batch_size

    # # tensorboardx
    # if args.tensorboard:
    #         print('Init tensorboardX')
    #         writer = SummaryWriter(log_dir='runs/{}'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")))

    # # make states
    # if args.env == 'flipped':
    #     assert INPUT_SHAPE_1 == INPUT_SHAPE_2
    #     env = flippedStates(args.input_shape_1)
    #     original_states, new_states = env.getStates()

    # if args.env == 'shape':
    #     env = shapeStates(args.input_shape_1,args.input_shape_2)
    #     original_states, new_states = env.getStates(flip=args.flip,random=args.random)


    # # CUDA compatability
    # use_cuda = torch.cuda.is_available()
    # device   = torch.device("cuda" if use_cuda else "cpu")

    # # instantiate autoencoders
    # if args.autoencoder_type == 'linear':
    #     autoencoder_1 = AutoencoderLinear(INPUT_SHAPE_1,LATENT_SHAPE).to(device)
    #     autoencoder_2 = AutoencoderLinear(INPUT_SHAPE_2,LATENT_SHAPE).to(device)
    # elif args.autoencoder_type == 'conv':
    #     autoencoder_1 = AutoencoderConv(INPUT_SHAPE_1,LATENT_SHAPE).to(device)
    #     autoencoder_2 = AutoencoderConv(INPUT_SHAPE_2,LATENT_SHAPE).to(device)
    # criterion = nn.MSELoss()
    # optimizer_1 = torch.optim.Adam(autoencoder_1.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    # optimizer_2 = torch.optim.Adam(autoencoder_2.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)

    # print('STARTING TRAINING...')
    # print('#'*50)
    # # print('\n\n\n')
    # losses = []
    # for epoch in tqdm(range(NUM_EPOCHS)):
    #     idx = np.random.randint(INPUT_SHAPE_1, size=BATCH_SIZE)
    #     batch_original_input = original_states[idx,:]
    #     batch_new_input  = new_states[idx,:]

    #     if args.noise:
    #         noise = np.random.normal(args.noise_mean,args.noise_std,INPUT_SHAPE_1)
    #         batch_original_input += noise
    #         batch_new_input  += noise

        
    #     orig_state = torch.FloatTensor(batch_original_input).to(device)
    #     new_state  = torch.FloatTensor(batch_new_input).to(device)

    #     s1,z1 = autoencoder_1(orig_state)
    #     s2,z2 = autoencoder_2(new_state)
        
    #     reconstruction_loss_1 = criterion(orig_state,s1)
    #     reconstruction_loss_2 = criterion(new_state.float(),s2)
    #     latent_loss = criterion(z1,z2)

    #     # add losses
    #     loss = args.alpha_latent*latent_loss + args.alpha_recon1*reconstruction_loss_1 + args.alpha_recon2*reconstruction_loss_2
    #     losses.append(loss.detach().cpu().numpy())
    #     if args.tensorboard:
    #         writer.add_scalar('Autoencoder_1_Loss',reconstruction_loss_1.item(),epoch)
    #         writer.add_scalar('Autoencoder_2_Loss',reconstruction_loss_2.item(),epoch)
    #         writer.add_scalar('Latent_Loss',latent_loss.item(),epoch)
    #         writer.add_scalar('Total_Loss',loss.item(),epoch)
            
            

    #     optimizer_1.zero_grad()
    #     loss.backward(retain_graph=True)
    #     optimizer_1.step()

    #     optimizer_2.zero_grad()
    #     loss.backward()
    #     optimizer_2.step()
        

    #     if epoch % 100 == 0 and epoch !=0:
    #                 # print('Saving model...')
    #                 torch.save({
    #                             'episode': epoch,
    #                             'model_state_dict': autoencoder_1.state_dict(),
    #                             'optimizer_state_dict': optimizer_1.state_dict(),
    #                             'loss': loss,
    #                             }, args.weight_paths[0])
    #                 torch.save({
    #                             'episode': epoch,
    #                             'model_state_dict': autoencoder_2.state_dict(),
    #                             'optimizer_state_dict': optimizer_2.state_dict(),
    #                             'loss': loss,
    #                             }, args.weight_paths[1])
    # # print('LOSS:',losses)
    # plt.plot(losses)
    # plt.grid()
    # plt.title('Total Loss')
    # plt.xlabel('epochs')
    # plt.ylabel('Loss')
    # plt.show()
                    