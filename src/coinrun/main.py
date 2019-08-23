import numpy as np
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
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from tensorboardX import SummaryWriter

from config import args
from coinrun import setup_utils, make
from autoencoders.autoencoder import AutoEncoder

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

transition = np.dtype([('s', np.float64, (3, 64, 64))])

if args.tensorboard:
    print('Init tensorboard')
    writer = SummaryWriter(log_dir='runs/{}'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")))



class Model():
    def __init__(self):
        self.AE = AutoEncoder(latent_dim=args.latent_dim).double().to(device)
        self.counter = 0
        self.buffer = np.empty(args.buffer_capacity, dtype=transition)
        setup_utils.setup_and_load(use_cmd_line_args=False)
        self.env = make('standard',num_envs=args.num_envs)
        self.optimizer = optim.Adam(self.AE.parameters(),lr = args.lr)
        self.criterion = nn.MSELoss()
        self.step=0

    def store(self,x):
        self.buffer['s'][self.counter] = x
        self.counter += 1 
        if self.counter == args.buffer_capacity:
            self.counter = 0
            return True
        else:
            return False
    
    def save_param(self):
        if not os.path.exists('./Weights'):
            os.makedirs('./Weights')
        torch.save(self.AE.state_dict(),'./Weights/' + args.weight_path)
    
    def load_param(self):
        self.AE.load_state_dict(torch.load('./Weights'+args.weight_path))

    def update(self):
        s = torch.tensor(self.buffer['s'],dtype=torch.double).to(device)
        for _ in range(args.train_epochs):
            for index in BatchSampler(SubsetRandomSampler(range(args.buffer_capacity)),args.batch_size, False):
                s_in = s[index]
                z,s_hat = self.AE(s_in)
                loss = self.criterion(s_hat,s_in)
                if args.tensorboard:
                    writer.add_scalar('Loss',loss.item(),self.step)
                self.optimizer.zero_grad()
                self.step+=1
                
    def run(self):
        for step in range(args.max_steps):
            act = np.array([self.env.action_space.sample() for _ in range(args.num_envs)])
            # act = self.env.action_space.sample()
            # print(np.int32(act),type(np.int32(act)))
            obs,_,_,_ = self.env.step(act)
            obs = np.transpose(np.squeeze(obs),(2,0,1))
            if self.store((obs/256)):
                print('Updating')
                self.update()
            if step % 10000 == 0 :
                print('Saving Model')
                self.save_param()
        self.env.close()
            

def main():
    model = Model()
    model.run()


if __name__ == '__main__':
    main()