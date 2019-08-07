import os
import datetime
import numpy as np
import shutil
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter


from config import args
from dataset import PascalVOCDataset


class AutoEncoder(nn.Module):
    def __init__(self,image_dim=28,latent_dim=100,noise_scale=0):
        super(AutoEncoder, self).__init__()

        self.image_dim = image_dim # a 28x28 image corresponds to 4 on the FC layer, a 64x64 image corresponds to 13
        self.latent_dim = latent_dim
        self.noise_scale = noise_scale
        self.latentLayerShape = self.getShape()
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
            nn.ReLU())
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
	    nn.Sigmoid())

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

        return z, x_hat

    def encode(self, x):
        #x = x.unsqueeze(0)
        z, x_hat = self.forward(x)

        return z

    def getShape(self):
        # gives the shape of the linear(latent) layer  
        # use w_out = (w-k_size+2*padding)/stride + 1  
        return int(self.image_dim/4-3)

def main():
    transform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


    trainset = torchvision.datasets.CIFAR10(root='~/data', train=True,
                                            download=False, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                            shuffle=True, num_workers=2)


    AE = AutoEncoder(32,100)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(AE.parameters(), lr=args.lr, weight_decay=1e-5)
    print('Training...')
    losses = []
    for epoch in range(args.num_epochs):
        running_loss = 0
        for i, data in enumerate(trainloader,0):
            inputs, _ = data
            # zero the parameter gradients
            optimizer.zero_grad()

            _, x_hat = AE(inputs)
            loss = criterion(x_hat,inputs)
            loss.backward()
            optimizer.step()
            # print statistics
            running_loss += loss.item()
            losses.append(loss)
            k=1
            if i % k == 0:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / k))
                running_loss = 0.0
    np.save('losses.npy',np.array(losses))

    torch.save({
                'model_state_dict': AE.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                }, args.save_path)

def train():
    # NOTE: Change the file paths here appropriately
    homedir = os.path.expanduser("~")
    data_root = homedir + "/data/VOCdevkit/VOC2007"
    list_file_path = os.path.join(data_root,"ImageSets","Main","train.txt")
    img_dir = os.path.join(data_root,"JPEGImages")

    dataset = PascalVOCDataset(list_file_path,img_dir,args.orig_shape,args.target_shape)
    trainloader = DataLoader(dataset,batch_size=args.batch_size,shuffle=True)

    if args.rm_runs:
        shutil.rmtree('runs')
    if args.tensorboard:
        print('Init tensorboardX')
        writer = SummaryWriter(log_dir='runs/{}'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")))

    # CUDA compatability
    use_cuda = torch.cuda.is_available()
    device   = torch.device("cuda" if use_cuda else "cpu")
    print('#'*50)
    print(device)


    AE1 = AutoEncoder(args.orig_shape,args.latent_shape).to(device)
    AE2 = AutoEncoder(args.target_shape,args.latent_shape).to(device)
    criterion = nn.MSELoss()
    optimizer_1 = torch.optim.Adam(AE1.parameters(), lr=args.lr, weight_decay=1e-5)
    optimizer_2 = torch.optim.Adam(AE2.parameters(), lr=args.lr, weight_decay=1e-5)

    i = 0
    for epoch in range(args.num_epochs):
        for batch in trainloader:
            orig_env_image = torch.autograd.Variable(batch['orig_env_image']).to(device)
            target_env_image = torch.autograd.Variable(batch['target_env_image']).to(device)

            z1,s1 = AE1(orig_env_image)
            z2,s2 = AE2(target_env_image)

            if args.scale_loss:
                reconstruction_loss1 = criterion(orig_env_image,s1)/(args.orig_shape**2)
                reconstruction_loss2 = criterion(target_env_image,s2)/(args.target_shape**2)
                latent_loss = criterion(z1,z2)*100
            else:
                reconstruction_loss1 = criterion(orig_env_image,s1)
                reconstruction_loss2 = criterion(target_env_image,s2)
                latent_loss = criterion(z1,z2)

            loss = reconstruction_loss1 + reconstruction_loss2 + latent_loss
            if args.tensorboard:
                writer.add_scalar('Autoencoder_1_Loss',reconstruction_loss1.item(),i)
                writer.add_scalar('Autoencoder_2_Loss',reconstruction_loss2.item(),i)
                writer.add_scalar('Latent_Loss',latent_loss.item(),i)
                writer.add_scalar('Total_Loss',loss.item(),i)

            optimizer_1.zero_grad()
            loss.backward(retain_graph=True)
            optimizer_1.step()

            optimizer_2.zero_grad()
            loss.backward()
            optimizer_2.step()
            i+=1

            print(loss.item(),reconstruction_loss1.item(),reconstruction_loss2.item(),latent_loss.item())
        torch.save({
                    'model_state_dict': AE1.state_dict(),
                    'optimizer_state_dict': optimizer_1.state_dict(),
                    }, args.weight_paths[0])
        torch.save({
                    'model_state_dict': AE2.state_dict(),
                    'optimizer_state_dict': optimizer_2.state_dict(), 
                    }, args.weight_paths[1])
            
            
            


    
    

if __name__ == "__main__":
    train()