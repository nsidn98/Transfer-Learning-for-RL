import matplotlib.pyplot as plt
import numpy as np
import os
import torch
from torch.utils.data import Dataset
from PIL import Image


class PascalVOCDataset(Dataset):
    """Pascal VOC 2007 Dataset"""
    def __init__(self, list_file, img_dir, orig_env_shape, target_env_shape):
        '''
        list_file:  path to a .txt file which contains the names of all the .jpg/.png
                    files to load the images from
        img_dir:    path to a directory which contains all the images
        '''
        self.images = open(list_file, "rt").read().split("\n")[:-1]

        self.img_extension = ".jpg"

        self.image_root_dir = img_dir
        self.orig_env_shape = orig_env_shape
        self.target_env_shape = target_env_shape


    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        name = self.images[index]
        image_path = os.path.join(self.image_root_dir, name + self.img_extension)

        orig_image,target_image = self.load_image(path=image_path)
        data = {
                'orig_env_image':   torch.FloatTensor(orig_image),
                'target_env_image': torch.FloatTensor(target_image)
        }
        return data

    def load_image(self, path=None):
        raw_image = Image.open(path)

        raw_image_orig = np.transpose(raw_image.resize((self.orig_env_shape, self.orig_env_shape)), (2,1,0))
        orig_imx_t = np.array(raw_image_orig, dtype=np.float32)/255.0

        raw_image_target = np.transpose(raw_image.resize((self.target_env_shape, self.target_env_shape)), (2,1,0))
        target_imx_t = np.array(raw_image_target, dtype=np.float32)/255.0
        return orig_imx_t, target_imx_t

    def load_sample_image(self):
        index = np.random.randint(0,len(self.images),1)[0]
        sample = self.__getitem__(index)
        image,target = sample['orig_env_image'],sample['target_env_image']
        image.transpose_(0,2)
        target.transpose_(0,2)
        # plt.imshow(image)
        # plt.title('Random number is '+str(index))
        # plt.show()

        fig = plt.figure()

        a = fig.add_subplot(1,2,1)
        plt.imshow(image)

        a = fig.add_subplot(1,2,2)
        plt.imshow(target)

        plt.show()



if __name__ == "__main__":
    homedir = os.path.expanduser("~")
    data_root = homedir + "/data/VOCdevkit/VOC2007"
    list_file_path = os.path.join(data_root,"ImageSets","Main","train.txt")
    img_dir = os.path.join(data_root,"JPEGImages")

    dataset = PascalVOCDataset(list_file_path,img_dir,64,32)
    dataset.load_sample_image()