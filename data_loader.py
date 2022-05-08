import torch.utils.data as data
from PIL import Image
import numpy as np
import math
import os
import os.path
import random
from preprocessing import*


def dataloader(filepath):
    #return array of addresses of all images


    left_fold  = 'image_2/'
    right_fold = 'image_3/'
    disp_L = 'disp_occ_0/'

    image = [img for img in os.listdir(filepath+left_fold) if img.find('_10') > -1]

    #split training and validation into 80%:20% (160:40 in case of 200 images in total)
    train = image[:160]
    val   = image[160:]

    left_train  = [filepath+left_fold+img for img in train]
    right_train = [filepath+right_fold+img for img in train]
    disp_train_L = [filepath+disp_L+img for img in train]

    left_val  = [filepath+left_fold+img for img in val]
    right_val = [filepath+right_fold+img for img in val]
    disp_val_L = [filepath+disp_L+img for img in val]

    return left_train, right_train, disp_train_L, left_val, right_val, disp_val_L



#RGB image loader fn
def default_loader(path):
    img=Image.open(path).convert('RGB')
    return img

#disparity iamge loader fn
def disparity_loader(file):
    img = Image.open(file)
    return img
     


#create Map-style dataset class (contains __getitem__() and __len__ methods) to get fed into TORCH.UTILS later on
class myImageFloder(data.Dataset):
    def __init__(self, left, right, left_disparity, training, loader=default_loader, dploader= disparity_loader):
 
        self.left = left
        self.right = right
        self.disp_L = left_disparity
        self.loader = loader
        self.dploader = dploader
        self.training = training

    def __getitem__(self, index):
        left  = self.left[index]
        right = self.right[index]
        disp_L= self.disp_L[index]

        left_img = self.loader(left)
        right_img = self.loader(right)
        dataL = self.dploader(disp_L)


        if self.training:
            #randomly crop 256x512 areas in training images for augmentation
           w, h = left_img.size
           th, tw = 256, 512
 
           x1 = random.randint(0, w - tw)
           y1 = random.randint(0, h - th)

           left_img = left_img.crop((x1, y1, x1 + tw, y1 + th))
           right_img = right_img.crop((x1, y1, x1 + tw, y1 + th))

           dataL = np.ascontiguousarray(dataL,dtype=np.float32)/256
           dataL = dataL[y1:y1 + th, x1:x1 + tw]

           processed = get_transform(augment=False)  
           left_img   = processed(left_img)
           right_img  = processed(right_img)

           return left_img, right_img, dataL
        else:
           w, h = left_img.size

           left_img = left_img.crop((w-1232, h-368, w, h))
           right_img = right_img.crop((w-1232, h-368, w, h))
           w1, h1 = left_img.size

           dataL = dataL.crop((w-1232, h-368, w, h))
           dataL = np.ascontiguousarray(dataL,dtype=np.float32)/256

           #obtain transformer operator for image augmentation (if needed)
           processed = get_transform(augment=False)  
           left_img       = processed(left_img)
           right_img      = processed(right_img)

           return left_img, right_img, dataL

    def __len__(self):
        return len(self.left)