from re import S
from torch.autograd import Variable
from turtle import forward
from matplotlib.cbook import contiguous_regions
import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
import torch.nn.functional as feature_extract
import os
import torch.optim as optim
import math
import torchvision
import random
import numpy as np
def convbn(in_num,out_num,kernel_size,stride,pad,dilation):
    return nn.Sequential(nn.Conv2d(in_num,out_num,kernel_size=kernel_size,stride=stride,padding=dilation if dilation>1 else pad,dilation=dilation,bias=False),
    nn.BatchNorm2d(out_num)
    )

def convbn_3D(in_num,out_num,kernel_size,stride,pad):
    return nn.Sequential(nn.Conv3d(in_num,out_num,kernel_size=kernel_size,stride=stride,padding= pad,bias=False),
    nn.BatchNorm3d(out_num)
    )

class BasicBlock(nn.Module):
    expansion=1
    def __init__(self,in_num,out_num,stride,downsample,pad,dilation):
        super(BasicBlock,self).__init__()
        self.conv1=nn.Sequential(convbn(in_num,out_num,3,stride,pad,dilation),
        nn.ReLU(inplace=True))
        self.conv2=convbn(out_num,out_num,3,1,pad,dilation)
        self.downsample=downsample
        self.stride=stride

    def forward(self,x):
        out=self.conv1(x)
        out=self.conv2(out)

        if self.downsample is not None:
            x=self.downsample(x)
        
        out+=x
        return out

    
class FeatureExtract(nn.Module):
    def __init__(self):
        super(FeatureExtract,self).__init__()
        #CNN
        self.in_num=32
        self.firstconv=nn.Sequential(
            convbn(3,32,3,2,1,1),
            nn.ReLU(inplace=True),
            convbn(32,32,3,1,1,1),
            nn.ReLU(inplace=True),
            convbn(32,32,3,1,1,1),
            nn.ReLU(inplace=True),
        )
        self.layer1=self.__make_block(BasicBlock,32,3,1,1,1)
        self.layer2=self.__make_block(BasicBlock,64,16,2,1,1,)
        self.layer3=self.__make_block(BasicBlock,128,3,1,1,1)
        self.layer4=self.__make_block(BasicBlock,128,3,1,1,2)
        #SPP module
        self.branch1=nn.Sequential(
            nn.AvgPool2d((32,32),stride=(32,32)),
            convbn(128,32,1,1,0,1),
            nn.ReLU(inplace=True)
        )
        self.branch2=nn.Sequential(
            nn.AvgPool2d((16,16),stride=(16,16)),
            convbn(128,32,1,1,0,1),
            nn.ReLU(inplace=True)
        )
        self.branch3=nn.Sequential(
            nn.AvgPool2d((8,8),stride=(8,8)),
            convbn(128,32,1,1,0,1),
            nn.ReLU(inplace=True)
        )
        self.branch4=nn.Sequential(
            nn.AvgPool2d((4,4),stride=(4,4)),
            convbn(128,32,1,1,0,1),
            nn.ReLU(inplace=True)
        )
        self.lastconv=nn.Sequential(
            convbn(320,128,3,1,1,1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 32, kernel_size=1, padding=0, stride = 1, bias=False)
        )


    def __make_block(self,block,out_num,block_num,stride,pad,dilation):
        downsample=None
        if stride!=1 or self.in_num!=out_num*block.expansion:
            downsample=nn.Sequential(
                nn.Conv2d(self.in_num,out_num*block.expansion,kernel_size=1,stride=stride,bias=False),
                nn.BatchNorm2d(out_num*block.expansion)
            )

        layers=[]
        layers.append(block(self.in_num,out_num,stride,downsample,pad,dilation))
        self.in_num=out_num*block.expansion
        for i in range(1,block_num):
            layers.append(block(self.in_num,out_num,1,None,pad,dilation))

        return nn.Sequential(*layers)

    def forward(self,x):
        output=self.firstconv(x)
        output=self.layer1(output)
        output2_16=self.layer2(output)
        output=self.layer3(output2_16)
        output4_3=self.layer4(output)

        out_branch1=self.branch1(output4_3)
        out_branch1=F.interpolate(out_branch1,(output4_3.size()[2],output4_3.size()[3]),mode='bilinear',align_corners=True)
        out_branch2=self.branch2(output4_3)
        out_branch2=F.interpolate(out_branch2,(output4_3.size()[2],output4_3.size()[3]),mode='bilinear',align_corners=True)
        out_branch3=self.branch3(output4_3)
        out_branch3=F.interpolate(out_branch3,(output4_3.size()[2],output4_3.size()[3]),mode='bilinear',align_corners=True)
        out_branch4=self.branch4(output4_3)
        out_branch4=F.interpolate(out_branch4,(output4_3.size()[2],output4_3.size()[3]),mode='bilinear',align_corners=True)
        out_feature=torch.cat((output2_16,output4_3,out_branch1,out_branch2,out_branch3,out_branch4),1)
        out_feature=self.lastconv(out_feature)

        return out_feature
    

class HourGlass(nn.Module):
    def __init__(self,in_num):
        super(HourGlass,self).__init__()

        self.conv1=nn.Sequential(
            convbn_3D(in_num,in_num*2,3,2,1),
             nn.ReLU(inplace=True),
        )
        self.conv2=convbn_3D(in_num*2,in_num*2,kernel_size=3,stride=1,pad=1)
        self.conv3=nn.Sequential(
            convbn_3D(in_num*2,in_num*2,3,2,1),
            nn.ReLU(inplace=True),
        )
        self.conv4=nn.Sequential(
            convbn_3D(in_num*2,in_num*2,3,1,1),
            nn.ReLU(inplace=True),
        )
        self.conv5=nn.Sequential(
            nn.ConvTranspose3d(in_num*2,in_num*2,kernel_size=3,padding=1,output_padding=1,stride=2,bias=False),
            nn.BatchNorm3d(in_num*2)        
        )
        self.conv6=nn.Sequential(
            nn.ConvTranspose3d(in_num*2,in_num,kernel_size=3,padding=1,output_padding=1,stride=2,bias=False),
            nn.BatchNorm3d(in_num)
        )

    def forward(self,x,presqu,postsqu):
        out=self.conv1(x)
        pre=self.conv2(out)
        if postsqu is not None:
            pre=F.relu(pre+postsqu,inplace=True)
        else:
            pre=F.relu(pre,inplace=True)

        out=self.conv3(pre)
        out=self.conv4(out)

        if presqu is not None:
            post=F.relu(self.conv5(out)+presqu,inplace=True)
        else:
            post=F.relu(self.conv5(out)+pre,inplace=True)
        out=self.conv6(post)
        return out,pre,post

class disparityregression(nn.Module):
    def __init__(self, maxdisp):
        super(disparityregression, self).__init__()
        self.disp = torch.Tensor(np.reshape(np.array(range(maxdisp)),[1, maxdisp,1,1])).cuda()

    def forward(self, x):
        out = torch.sum(x*self.disp.data,1, keepdim=True)
        return out       

class PSMNet(nn.Module):
    def __init__(self,maxdisp):
        super(PSMNet,self).__init__()
        self.maxdisp=maxdisp
        self.featureextract=FeatureExtract()
        self.dres0=nn.Sequential(
            convbn_3D(64,32,3,1,1),
            nn.ReLU(inplace=True),
            convbn_3D(32, 32, 3, 1, 1),
            nn.ReLU(inplace=True),
        )
        self.dres1=nn.Sequential(
            convbn_3D(32,32,3,1,1),
            nn.ReLU(inplace=True),
            convbn_3D(32, 32, 3, 1, 1),
        )
        self.dres2=HourGlass(32)
        self.dres3=HourGlass(32)
        self.dres4=HourGlass(32)
        self.classify1=nn.Sequential(
            convbn_3D(32,32,3,1,1),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1,bias=False)
        )
        self.classify2=nn.Sequential(
            convbn_3D(32,32,3,1,1),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1,bias=False)
        )
        self.classify3=nn.Sequential(
            convbn_3D(32,32,3,1,1),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1,bias=False)
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1]*m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self,left,right):
        ref_img=self.featureextract(left)
        tar_img=self.featureextract(right)
        cost=Variable(torch.FloatTensor(ref_img.size()[0],ref_img.size()[1]*2,self.maxdisp // 4,ref_img.size()[2],ref_img.size()[3]).zero_()).cuda()

        for i in range (self.maxdisp//4):
            if i>0:
                cost[:,:ref_img.size()[1],i,:,i:]=ref_img[:,:,:,i:]
                cost[:,:ref_img.size()[1]:,i,:,i:]=tar_img[:,:,:,:-i]
            else:
                cost[:,:ref_img.size()[1],i,:,:]=ref_img
                cost[:,ref_img.size()[1]:,i,:,:]=tar_img
        cost=cost.contiguous()
        cost0=self.dres0(cost)
        cost0=self.dres1(cost0)+cost0
        out1,pre1,post1=self.dres2(cost0,None,None)
        out1=out1+cost0
        out2,pre2,post2=self.dres3(out1,pre1,post1)
        out2=out2+cost0
        out3,pre3,post3=self.dres4(out2,pre2,post2)
        out3=out3+cost0       
        cost1=self.classify1(out1) 
        cost2=self.classify1(out2) +cost1
        cost3=self.classify1(out3) +cost2

        if self.train:
            cost1 = F.interpolate(cost1, [self.maxdisp,left.size()[2],left.size()[3]], mode='trilinear')
            cost2 = F.interpolate(cost2, [self.maxdisp,left.size()[2],left.size()[3]], mode='trilinear')
            cost1 = torch.squeeze(cost1,1)
            pred1 = F.softmax(cost1,dim=1)
            pred1 = disparityregression(self.maxdisp)(pred1)
            cost2 = torch.squeeze(cost2,1)
            pred2 = F.softmax(cost2,dim=1)
            pred2 = disparityregression(self.maxdisp)(pred2)
        cost3 = F.interpolate(cost3, [self.maxdisp,left.size()[2],left.size()[3]], mode='trilinear')
        cost3 = torch.squeeze(cost3,1)
        pred3 = F.softmax(cost3,dim=2)
        pred3 = disparityregression(self.maxdisp)(pred3)


        if self.train:
            return pred1,pred2,pred3
        else:
            return pred3



