
import torchvision.transforms as transforms
import argparse
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

import torch.nn.functional as F

import time
from data_loader import *
from model import *

#model needed parameters
parser = argparse.ArgumentParser(description='PSMNet')
parser.add_argument('--maxdisp', type=int ,default=80,
                    help='maxium disparity')
parser.add_argument('--model', default='stackhourglass',
                    help='select model')
parser.add_argument('--datapath', default="D:/prj_1_dl/training/",
                    help='datapath')
parser.add_argument('--epochs', type=int, default=10,
                    help='number of epochs to train')
parser.add_argument('--loadmodel', default='D:/prj_1_dl/pretrained_sceneflow_new.tar',
                    help='load model')
parser.add_argument('--savemodel', default='./',
                    help='save model')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

all_left_img, all_right_img, all_left_disp, test_left_img, test_right_img, test_left_disp = dataloader("D:/prj_1_dl/training/")


#data_loaders
#Using map-style datasets and shuffled sampler
TrainImgLoader = torch.utils.data.DataLoader(
         myImageFloder(all_left_img,all_right_img,all_left_disp, True), 
         batch_size= 2, shuffle= True, num_workers= 2, drop_last=False)#12 , 8

TestImgLoader = torch.utils.data.DataLoader(
         myImageFloder(test_left_img,test_right_img,test_left_disp, False), 
         batch_size= 2, shuffle= False, num_workers= 2, drop_last=False)#8 , 4


#Define model as stackhourglass based architecture or basic block based one 
if args.model == 'stackhourglass':
    model = PSMNet_hourglass(args.maxdisp)
elif args.model == 'basic':
    model = PSMNet_basic(args.maxdisp)
else:
    print('no model')

if args.cuda:
    model = nn.DataParallel(model)
    model.cuda()

#load the model parameters from parameters pretrained on scene flow data
if args.loadmodel is not None:
    print('Load pretrained model')
    pretrain_dict = torch.load(args.loadmodel)
    model.load_state_dict(pretrain_dict['state_dict'])



optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))

def train(imgL,imgR, disp_L):
        model.train()

        if args.cuda:
           
            imgL, imgR, disp_true = imgL.cuda(), imgR.cuda(), disp_L.cuda()
            #trans=transforms.ToPILImage()
            #ground_truth=trans(disp_true[0])
       #---------
        mask = disp_true < args.maxdisp
        mask.detach_()
        #----
        optimizer.zero_grad()
        
        if args.model == 'stackhourglass':
            output1, output2, output3 = model(imgL,imgR)
            output1 = torch.squeeze(output1,1)
            output2 = torch.squeeze(output2,1)
            output3 = torch.squeeze(output3,1)
            #model_output=trans(output3[0])
            t=int(time.time())
          
          #save model's output and ground truth to observe model's improvement
            #model_output.save("D:/prj_1_dl/evl/" +str(t) + '.png' )
            #ground_truth.save("D:/prj_1_dl/evl/" +str(t) +'_0.png' )

            loss = 0.5*F.smooth_l1_loss(output1[mask], disp_true[mask], size_average=True) + 0.7*F.smooth_l1_loss(output2[mask], disp_true[mask], size_average=True) + F.smooth_l1_loss(output3[mask], disp_true[mask], size_average=True) 
        elif args.model == 'basic':
            output = model(imgL,imgR)
           
            loss = F.smooth_l1_loss(output[mask], disp_true[mask], size_average=True)

        loss.backward()
        optimizer.step()

        return loss.data,F.smooth_l1_loss(output3[mask], disp_true[mask], size_average=True)

def test(imgL,imgR,disp_true):

        model.eval()
  
        if args.cuda:
            imgL, imgR, disp_true = imgL.cuda(), imgR.cuda(), disp_true.cuda()
        #---------
        mask = disp_true < 80
        #----

        if imgL.shape[2] % 16 != 0:
            times = imgL.shape[2]//16       
            top_pad = (times+1)*16 -imgL.shape[2]
        else:
            top_pad = 0

        if imgL.shape[3] % 16 != 0:
            times = imgL.shape[3]//16                       
            right_pad = (times+1)*16-imgL.shape[3]
        else:
            right_pad = 0  

        imgL = F.pad(imgL,(0,right_pad, top_pad,0))
        imgR = F.pad(imgR,(0,right_pad, top_pad,0))

        with torch.no_grad():
            output3 = model(imgL,imgR)
            output3 = torch.squeeze(output3)
       

        if top_pad !=0:
            img = output3[:,top_pad:,top_pad:]
        else:
            img = output3
    
        if len(disp_true[mask])==0:
           loss = 0
        else:
           loss = F.l1_loss(img[mask],disp_true[mask]) #torch.mean(torch.abs(img[mask]-disp_true[mask]))  # end-point-error

        return loss.data.cpu()

#to adjust learning rate eveny n epoches(not used)
def adjust_learning_rate(optimizer, epoch):
    lr = 0.001
    print(lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def main():
    losses=[]
    avg_losses=[]
    avg_val_losses=[]
    for epoch in range(0,14):
        print('This is %d-th epoch' %(epoch))
        total_train_loss = 0
        adjust_learning_rate(optimizer,epoch)

        ## training ##
        for batch_idx, (imgL_crop, imgR_crop, disp_crop_L) in enumerate(TrainImgLoader):
        
            start_time = time.time()

            loss,x= train(imgL_crop,imgR_crop, disp_crop_L)
            losses.append(x.data.cpu().numpy())
            print('Iter %d training loss = %.3f , time = %.2f' %(batch_idx, loss, time.time() - start_time))
            total_train_loss += loss
        avg_loss=sum(losses)/len(losses)
        losses=[]
        avg_losses.append(avg_loss)
        val_loss=[]
        print('validation')
        for batch_idx, (imgL, imgR, disp_L) in enumerate(TestImgLoader):
            val_loss_new = test(imgL,imgR, disp_L)
            val_loss.append(val_loss_new)
    
        avg_val_loss=sum(val_loss)/len(val_loss)
        avg_val_losses.append(avg_val_loss.cpu().numpy())

        print('avg loss ',avg_loss)
        print('avg val loss ',avg_val_loss.cpu().numpy())

    print('showing',avg_losses,avg_val_losses)
    plt.plot(avg_losses,label="Training Error")
    plt.plot(avg_val_losses,label="Validation Error")
    plt.xlabel('Number of epoches')
    plt.ylabel('Average Loss per epoch')
    plt.legend()
    plt.show()
 


if __name__ == '__main__':
   main()
    
