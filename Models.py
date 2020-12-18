import os
import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as tfunc

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

import torchvision

class DenseNet121(nn.Module):

    def __init__(self, classCount, isTrained):
	
        super(DenseNet121, self).__init__()
		
        self.densenet121 = torchvision.models.densenet121(pretrained=isTrained)

        kernelCount = self.densenet121.classifier.in_features
		
        self.densenet121.classifier = nn.Sequential(nn.Linear(kernelCount, classCount), nn.Sigmoid())

    def forward(self, x):
        x = self.densenet121(x)
        return x

class DenseNet169(nn.Module):
    
    def __init__(self, classCount, isTrained):
        
        super(DenseNet169, self).__init__()
        
        self.densenet169 = torchvision.models.densenet169(pretrained=isTrained)
 
        kernelCount = self.densenet169.classifier.in_features
        
        self.densenet169.classifier = nn.Sequential(nn.Linear(kernelCount, classCount), nn.Sigmoid())
        
    def forward (self, x):
        x = self.densenet169(x)
        return x
    
class DenseNet201(nn.Module):
    
    def __init__ (self, classCount, isTrained):
        
        super(DenseNet201, self).__init__()
        
        self.densenet201 = torchvision.models.densenet201(pretrained=isTrained)
        
        kernelCount = self.densenet201.classifier.in_features
        
        self.densenet201.classifier = nn.Sequential(nn.Linear(kernelCount, classCount), nn.Sigmoid())
        
    def forward (self, x):
        x = self.densenet201(x)
        return x


class DenseNet161(nn.Module):

    def __init__(self, classCount, isTrained):

        super(DenseNet161, self).__init__()

        self.densenet161 = torchvision.models.densenet161(pretrained=isTrained)

        kernelCount = self.densenet161.classifier.in_features

        self.densenet161.classifier = nn.Sequential(nn.Linear(kernelCount, classCount), nn.Sigmoid())

    def forward(self, x):
        x = self.densenet161(x)
        return x

class ResNet50(nn.Module):

    def __init__(self, classCount, isTrained):

        super(ResNet50, self).__init__()

        self.resnet50 = torchvision.models.resnet50(pretrained=isTrained)

        kernelCount = self.resnet50.fc.in_features

        self.resnet50.fc = nn.Sequential(nn.Linear(kernelCount, classCount), nn.Sigmoid())

    def forward(self, x):
        x = self.resnet50(x)
        return x
    
class ResNet18(nn.Module):

    def __init__(self, classCount, isTrained):

        super(ResNet18, self).__init__()

        self.resnet18 = torchvision.models.resnet18(pretrained=isTrained)

        #for param in self.resnet18.parameters():
        #    param.requires_grad = False

        kernelCount = self.resnet18.fc.in_features

        self.resnet18.fc = nn.Sequential(nn.Linear(kernelCount, classCount), nn.Sigmoid())
        #self.resnet18.fc = nn.Sequential(nn.Linear(kernelCount ,512), nn.ReLU(), nn.Dropout(0.2), nn.Linear(512,classCount), nn.Sigmoid())
    
    def forward(self, x):
        x = self.resnet18(x)
        return x

class ResNet14(nn.Module):

    def __init__(self, classCount, isTrained):

        super(ResNet14, self).__init__()

        self.resnet14 = torchvision.models.resnet14(pretrained=isTrained)

        #for param in self.resnet18.parameters():
        #    param.requires_grad = False

        kernelCount = self.resnet14.fc.in_features

        self.resnet14.fc = nn.Sequential(nn.Linear(kernelCount, classCount), nn.Sigmoid())
        #self.resnet18.fc = nn.Sequential(nn.Linear(kernelCount ,512), nn.ReLU(), nn.Dropout(0.2), nn.Linear(512,classCount), nn.Sigmoid())
    
    def forward(self, x):
        x = self.resnet14(x)
        return x

class Vgg11_BN(nn.Module):

    def __init__(self, classCount, isTrained):

        super(Vgg11_BN, self).__init__()

        self.vgg11_bn = torchvision.models.vgg11_bn(pretrained=isTrained)        

        self.vgg11_bn.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, classCount),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.vgg11_bn(x)
        return x

class Vgg16_BN(nn.Module):

    def __init__(self, classCount, isTrained):

        super(Vgg16_BN, self).__init__()

        self.vgg16_bn = torchvision.models.vgg16_bn(pretrained=isTrained)

        self.vgg16_bn.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, classCount),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.vgg16_bn(x)
        return x


class ReisenNet6(nn.Module):

    def __init__(self, classCount, isTrained):

        super(ReisenNet6, self).__init__()
        
        self.fc1 = nn.Linear(112 * 112 * 3, 16384)
        self.fc2 = nn.Linear(16384, 4096)
        self.fc3 = nn.Linear(4096, 1024)
        self.fc4 = nn.Linear(1024, 256)
        self.fc5 = nn.Linear(256, 64)
        self.fc6 = nn.Linear(64, classCount)
        	
        self.dropout = nn.Dropout(p = 0.2)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        	
        x = self.dropout(tfunc.relu(self.fc1(x)))
        x = self.dropout(tfunc.relu(self.fc2(x)))
        x = self.dropout(tfunc.relu(self.fc3(x)))		
        x = self.dropout(tfunc.relu(self.fc4(x)))
        x = self.dropout(tfunc.relu(self.fc5(x)))
        	
        x = tfunc.log_softmax(self.fc6(x), dim = 1)
        
        return x
