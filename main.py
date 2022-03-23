from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import time
import torchvision
import math
from net import *
import torch.utils.data
import dataloader as dl
from PIL import Image
import SecenFlowLoader as SF
import torchvision.transforms as transforms
from PIL import Image
parser=argparse.ArgumentParser(description='PSMNet')
parser.add_argument('--maxdisp',type=int,default=192,help='maxium disparity')
parser.add_argument('--datapath',help='datapath')
parser.add_argument('--epoches',type=int,default=10,help='number of epoches')
parser.add_argument('--savemodel',default='./',help='save model')
parser.add_argument('--loadmodel',default=None,help='load model')
args=parser.parse_args()

torch.cuda.manual_seed(1)
train_left_img, train_right_img, train_left_disp, test_left_img, test_right_img, test_left_disp = dl.dataloader(args.datapath)

Train_data_loader=torch.utils.data.DataLoader(SF.myImageFloder(train_left_img,train_right_img,train_left_disp,training=True),batch_size=2,shuffle=True,num_workers=1,drop_last=False)
Test_data_loader=torch.utils.data.DataLoader(SF.myImageFloder(test_left_img,test_right_img,test_left_disp,training=False),batch_size=1,shuffle=True,num_workers=1,drop_last=False)
model=PSMNet(args.maxdisp)
model=nn.DataParallel(model)
model.cuda()
optimizer = optim.Adam(model.parameters(), lr=0.001)
pretrain = torch.load(args.loadmodel)
model.module.load_state_dict(pretrain['state_dict'])
def train(imgL,imgR,disp_L):
    model.train()
    imgL,imgR,disp_L=imgL.cuda(),imgR.cuda(),disp_L.cuda()
    mask=disp_L<args.maxdisp
    mask.detach_()
    optimizer.zero_grad()
    output1,output2,output3=model(imgL,imgR)
    output1 = torch.squeeze(output1, 1)
    output2 = torch.squeeze(output2, 1)
    output3 = torch.squeeze(output3, 1)
    loss = 0.5 * F.smooth_l1_loss(output1[mask], disp_L[mask], size_average=True) + 0.7 * F.smooth_l1_loss(output2[mask], disp_L[mask], size_average=True) + F.smooth_l1_loss(output3[mask], disp_L[mask],size_average=True)
    loss.backward()
    optimizer.step()
    return loss.data

def test(imgL,imgR,disp):
    model.eval()
    imgL,imgR,disp=imgL.cuda(),imgR.cuda(),disp.cuda()
    mask = disp < args.maxdisp
    if imgL.shape[2] % 16 != 0:
        times = imgL.shape[2] // 16
        top_pad = (times + 1) * 16 - imgL.shape[2]
    else:
        top_pad = 0

    if imgL.shape[3] % 16 != 0:
        times = imgL.shape[3] // 16
        right_pad = (times + 1) * 16 - imgL.shape[3]
    else:
        right_pad = 0

    imgL = F.pad(imgL, (0, right_pad, top_pad, 0))
    imgR = F.pad(imgR, (0, right_pad, top_pad, 0))
    with torch.no_grad():
        output3 = model(imgL, imgR)[0]
        output3 = torch.squeeze(output3,1)

    if top_pad != 0:
        img = output3[:, top_pad:, :]
    else:
        img = output3
    disp_=    transforms.ToPILImage()(disp)
    disp_.show()
    img_ = transforms.ToPILImage()(img)
    img_.show()

    if len(disp[mask]) == 0:
        loss = 0
    else:
        loss = F.smooth_l1_loss(img[mask], disp[mask])

    return loss.data.cpu()



def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}

def main():
    #Train
     start_time=time.time()
     # for i in range(args.epoches):
     #     total_loss=0
     #     for batch_idx,(imgL,imgR,disp) in enumerate(Train_data_loader):
     #         print("%d/%d"%(batch_idx,len(Train_data_loader)))
     #         total_loss+=train(imgL,imgR,disp)
     #     print('%d-th epoch,loss:%f'%(i,total_loss/len(Train_data_loader)))
     # model_save_path=args.savemodel+'/model.tar'
     # torch.save(
     #     {     'state_dict': model.state_dict(),'train_loss': total_loss/len(Train_data_loader)},model_save_path
     # )

     for batch_idx, (imgL, imgR, disp_L) in enumerate(Test_data_loader):
         if batch_idx>1 :
             break
         test_loss = test(imgL, imgR, disp_L)
         print(test_loss)


if __name__=='__main__':
    main()