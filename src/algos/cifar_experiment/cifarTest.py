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

def getCriterion(type):
    if type == 'MSE':
        criterion = nn.MSELoss()
    elif type == 'L1':
        criterion = nn.L1Loss()
    elif type == 'SmoothL1':
        criterion = nn.SmoothL1Loss()
    return criterion

def train():
    # NOTE: Change the file paths here appropriately
    if args.container:
        homedir = '/scratch/scratch1/sidnayak/'
    else:
        homedir = os.path.expanduser("~")
    data_root = homedir + "/data/VOCdevkit/VOC2007"
    list_file_path = os.path.join(data_root,"ImageSets","Main","train.txt")
    img_dir = os.path.join(data_root,"JPEGImages")
    if not os.path.exists('./Weights'):
        os.makedirs('./Weights')

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

    recon_criterion = getCriterion(args.recon_loss_type)
    latent_criterion = getCriterion(args.latent_loss_type)

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
                reconstruction_loss1 = recon_criterion(orig_env_image,s1)/(args.orig_shape**2)
                reconstruction_loss2 = recon_criterion(target_env_image,s2)/(args.target_shape**2)
                latent_loss = latent_criterion(z1,z2)/args.latent_shape
                loss = reconstruction_loss1 + reconstruction_loss2 + latent_loss
            else:
                reconstruction_loss1 = recon_criterion(orig_env_image,s1)
                reconstruction_loss2 = recon_criterion(target_env_image,s2)
                latent_loss = latent_criterion(z1,z2)
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
        print('Saving Weights')
        torch.save({
                    'model_state_dict': AE1.state_dict(),
                    'optimizer_state_dict': optimizer_1.state_dict(),
                    }, args.weight_paths[0])
        torch.save({
                    'model_state_dict': AE2.state_dict(),
                    'optimizer_state_dict': optimizer_2.state_dict(), 
                    }, args.weight_paths[1])

def tensor2img(img):
    img = img[0] # remove the 'batch' dimension
    img = img.numpy()
    img = np.transpose(img,(2,1,0))
    return img
            
def test():
     # NOTE: Change the file paths here appropriately
    if args.container:
        homedir = '/scratch/scratch1/sidnayak/'
    else:
        homedir = os.path.expanduser("~")
    data_root = homedir + "/data/VOCdevkit/VOC2007"
    list_file_path = os.path.join(data_root,"ImageSets","Main","train.txt")
    img_dir = os.path.join(data_root,"JPEGImages")
    if not os.path.exists('./Weights'):
        os.makedirs('./Weights')

    dataset = PascalVOCDataset(list_file_path,img_dir,args.orig_shape,args.target_shape)
    trainloader = DataLoader(dataset,batch_size=1,shuffle=True)



    AE1 = AutoEncoder(args.orig_shape,args.latent_shape)
    AE2 = AutoEncoder(args.target_shape,args.latent_shape)
    criterion = nn.MSELoss()
    if os.path.exists(args.weight_paths[0]):
        print('Loading weights')
        checkpoint1 = torch.load(args.weight_paths[0],map_location='cpu')
        checkpoint2 = torch.load(args.weight_paths[1],map_location='cpu')

        AE1.load_state_dict(checkpoint1['model_state_dict'])
        AE2.load_state_dict(checkpoint2['model_state_dict'])

    AE1.eval()
    AE2.eval()

    i = 0
    for batch in trainloader:
        orig_env_image = torch.autograd.Variable(batch['orig_env_image'])
        target_env_image = torch.autograd.Variable(batch['target_env_image'])
        z1,s1 = AE1(orig_env_image)
        z2,s2 = AE2(target_env_image)

        img1 = tensor2img(orig_env_image)
        img2 = tensor2img(target_env_image)
        s1_img = tensor2img(s1.detach())
        s2_img = tensor2img(s2.detach())
        
        plt.figure()

        ax= plt.subplot(2,2,1)
        im=ax.imshow(img1)
        plt.title('64x64')
        ax= plt.subplot(2,2,2)
        im=ax.imshow(s1_img)
        plt.title('reconstruct(64x64)')
        
        ax= plt.subplot(2,2,3)
        im=ax.imshow(img1)
        plt.title('32x32')
        ax= plt.subplot(2,2,4)
        im=ax.imshow(s2_img)
        plt.title('reconstruct(32x32)')
        plt.tight_layout()
        print(list(z1.detach().numpy()))
        print(list(z2.detach().numpy()))
        # print(criterion(z1,z2))
        plt.show()
        # plot z1 and z2
        plt.plot(np.array(list(z1.detach().numpy()))[0],label='z1')
        plt.plot(np.array(list(z2.detach().numpy()))[0],label='z2')
        plt.grid()
        plt.legend()
        plt.show()
        

        i+=1
        if i == 1:
            break
           

if __name__ == "__main__":
    if args.train:
        print('Training')
        train()
    else:
        print('Testing')
        test()