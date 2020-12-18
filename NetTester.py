import os
import numpy as np
import time
import sys

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn.functional as tfunc
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as func

from sklearn.metrics.ranking import roc_auc_score

from Models import DenseNet121
from Models import DenseNet169
from Models import DenseNet201
from Models import DenseNet161
from Models import ResNet50
from Models import ResNet18
from Models import ResNet14
from Models import Vgg11_BN
from Models import Vgg16_BN

from DatasetGenerator import DatasetGenerator

#-------------------------------------------------------------------------------- 

class NetTester ():

    def epochTrain (model, dataLoader, optimizer, scheduler, epochMax, classCount, loss):
        
        model.train()

        lossTrain = 0
        lossTrainNorm = 0
        losstensorMean = 0
        
        for batchID, (data_input, target) in enumerate (dataLoader):
                        
            target = target.cuda()
                 
            varInput = data_input.cuda()
            varTarget = target.cuda()     
            varOutput = model(varInput)
          #  try:
          #      varOutput = model(varInput)
          #  except RuntimeError as exception:
          #      if "out of memory" in str(exception):
          #          print("WARNING: out of memory")
          #          if hasattr(torch.cuda, 'empty_cache'):
          #              torch.cuda.empty_cache()
          #      else:
          #          raise exception

            losstensor = loss(varOutput, varTarget)
            losstensorMean += losstensor
                  

            lossTrain += losstensor.item()
            lossTrainNorm += 1

            optimizer.zero_grad()
            losstensor.backward()
            optimizer.step()

        acc = 0
        outLoss = lossTrain / lossTrainNorm
        losstensorMean = losstensorMean / lossTrainNorm

        return outLoss, losstensorMean, acc

    #-------------------------------------------------------------------------------- 
        
    def epochVal (model, dataLoader, optimizer, scheduler, epochMax, classCount, loss):
        
        model.eval()
		
        lossVal = 0
        lossValNorm = 0
        losstensorMean = 0
        
        #correct = torch.zeros(1).squeeze().cuda()
        #total = torch.zeros(1).squeeze().cuda()
        
        for i, (data_input, target) in enumerate (dataLoader):
            
            target = target.cuda()
            
            with torch.no_grad():
                varInput = data_input.cuda()
                varTarget = target.cuda() 		 
                varOutput = model(varInput)
           # try:
           #     varOutput = model(varInput)
           # except RuntimeError as exception:
           #     if "out of memory" in str(exception):
           #         print("WARNING: out of memory")
           #         if hasattr(torch.cuda, 'empty_cache'):
           #             torch.cuda.empty_cache()
           #     else:
           #         raise exception
  
            #prediction = torch.argmax(varOutput, 1)
            #correct += (prediction == target).sum().float()
            #total += len(target)
           
            #_, predicted = torch.max(varOutput.data, 1)
            #print(predicted)
            losstensor = loss(varOutput, varTarget)
            losstensorMean += losstensor
                  

            lossVal += losstensor.item()
            lossValNorm += 1
        
        #pre, recall, f1score = ChexnetTrainer.computeAcc_Recall_F1score(varTarget, varOutput, classCount)
		
        #print(varOutput)
	
        #acc = (correct/total).cpu().detach().data.numpy()
        acc = 0
        outLoss = lossVal / lossValNorm
        losstensorMean = losstensorMean / lossValNorm
       
        scheduler.step(outLoss)
		
        return outLoss, losstensorMean, acc
               
    #--------------------------------------------------------------------------------     
     
    #---- Computes area under ROC curve 
    #---- dataGT - ground truth data
    #---- dataPRED - predicted data
    #---- classCount - number of classes
    
    def computeAUROC (dataGT, dataPRED, classCount):
        
        outAUROC = []
        
        datanpGT = dataGT.cpu().numpy()
        datanpPRED = dataPRED.cpu().numpy()
        
        for i in range(classCount):
            outAUROC.append(roc_auc_score(datanpGT[:, i], datanpPRED[:, i]))
            
        return outAUROC
      
        
        
    def computeAcc_Recall_F1score_No_Print (dataGT, dataPRED, classCount):
        
#        outPre = []
#        outRecall = []
#        outF1score = []
        
        datanpGT = dataGT.cpu().numpy()
        datanpPRED = dataPRED.cpu().numpy()
        
        for i in range(classCount):
            TP = 0
            TN = 0
            FN = 0
            FP = 0
            
            for j in range(np.size(datanpGT,i)):
                datanpPRED[j, i] = 1 if datanpPRED[j, i] >= 0.5 else 0
                TP += ((datanpPRED[j, i] == 1) & (datanpGT[j, i] == 1))
                # TN predict 和 label 同时为0
                TN += ((datanpPRED[j, i] == 0) & (datanpGT[j, i] == 0))
                # FN predict 0 label 1
                FN += ((datanpPRED[j, i] == 0) & (datanpGT[j, i] == 1))
                # FP predict 1 label 0
                FP += ((datanpPRED[j, i] == 1) & (datanpGT[j, i] == 0))
                
            p = TP / (TP + FP)
            r = TP / (TP + FN)
            f1 = 2 * r * p / (r + p)
            
#            outPre.append(p)  
#            outRecall.append(r)
#            outF1score.append(F1)
            
        return p, r, f1      
    
    
    
    def computeAcc_Recall_F1score (dataGT, dataPRED, classCount):
        
#        outPre = []
#        outRecall = []
#        outF1score = []
        
        datanpGT = dataGT.cpu().numpy()
        datanpPRED = dataPRED.cpu().numpy()
        
        acc = []
        p = []
        r = []
        f1 = []

        print('GT')        
        for j in range(np.size(datanpGT, 0)):
   
            for i in range(classCount):
                if i == (classCount - 1):
                    print(datanpGT[j, i])
                else: 
                    print(datanpGT[j, i], end = ' ')

        print('\n')
        print('PRED')

        for j in range(np.size(datanpGT, 0)):
        
            for i in range(classCount):
                if i == (classCount - 1):
                    print(datanpPRED[j, i])
                else:
                    print(datanpPRED[j, i], end = ' ')
            
        print('\n')

        for i in range(classCount):
            TP = 0
            TN = 0
            FN = 0
            FP = 0

#            print('class%d', i);
            for j in range(np.size(datanpGT, 0)):
#                print(datanpGT[j, i], end = ' ')
#                print(datanpPRED[j, i])


                datanpPRED[j, i] = 1 if datanpPRED[j, i] >= 0.5 else 0
                TP += ((datanpPRED[j, i] == 1) & (datanpGT[j, i] == 1))
                # TN predict 和 label 同时为0
                TN += ((datanpPRED[j, i] == 0) & (datanpGT[j, i] == 0))
                # FN predict 0 label 1
                FN += ((datanpPRED[j, i] == 0) & (datanpGT[j, i] == 1))
                # FP predict 1 label 0
                FP += ((datanpPRED[j, i] == 1) & (datanpGT[j, i] == 0))
             
               # print('#%d' % j, end = '     ')
               # print('GT:  ', end = '')
               # print(datanpGT[j, i], end = '     ')
               # print('PRED:  ', end = '')
               # print(datanpPRED[j, i])

            acc.append((TP + TN) / (TP + TN + FN + FP))    
            p.append(TP / (TP + FP))
            r.append(TP / (TP + FN))
            f1.append(2 * r[i] * p[i] / (r[i] + p[i]))

            
#            outPre.append(p)  
#            outRecall.append(r)
#            outF1score.append(F1)
            
        return acc, p, r, f1

    def computeAcc_Recall_F1score2 (dataGT, dataPRED, classCount):
        
#        outPre = []
#        outRecall = []
#        outF1score = []
        
        datanpGT = dataGT.cpu().numpy()
        datanpPRED = dataPRED.cpu().numpy()
        result = []
        ground = []
          
        for i in range(classCount):
            TP = 0
            TN = 0
            FN = 0
            FP = 0
            
            for j in range(np.size(datanpGT,i)):
                #print('#%d' % j, end = '     ')
                #print('GT:  ', end = '')
                #print(datanpGT[j, i], end = '     ')
                #print('PRED:  ', end = '')
                #print(datanpPRED[j, i])


                datanpPRED[j, i] = 1 if datanpPRED[j, i] >= 0.5 else 0
                TP += ((datanpPRED[j, i] == 1) & (datanpGT[j, i] == 1))
                # TN predict 和 label 同时为0
                TN += ((datanpPRED[j, i] == 0) & (datanpGT[j, i] == 0))
                # FN predict 0 label 1
                FN += ((datanpPRED[j, i] == 0) & (datanpGT[j, i] == 1))
                # FP predict 1 label 0
                FP += ((datanpPRED[j, i] == 1) & (datanpGT[j, i] == 0))
             
                result.append(datanpPRED[j, i])
                ground.append(datanpGT[j, i])
               # print('#%d' % j, end = '     ')
               # print('GT:  ', end = '')
               # print(datanpGT[j, i], end = '     ')
               # print('PRED:  ', end = '')
               # print(datanpPRED[j, i])

            acc = (TP + TN) / (TP + TN + FN + FP)    
            p = TP / (TP + FP)
            r = TP / (TP + FN)
            f1 = 2 * r * p / (r + p)
            
#            outPre.append(p)  
#            outRecall.append(r)
#            outF1score.append(F1)
            
        return acc, p, r, f1, result, ground 
    #--------------------------------------------------------------------------------  
    
    #---- Test the trained network 
    #---- pathDirData - path to the directory that contains images
    #---- pathFileTrain - path to the file that contains image paths and label pairs (training set)
    #---- pathFileVal - path to the file that contains image path and label pairs (validation set)
    #---- nnArchitecture - model architecture 'DENSE-NET-121', 'DENSE-NET-169' or 'DENSE-NET-201'
    #---- nnIsTrained - if True, uses pre-trained version of the network (pre-trained on imagenet)
    #---- nnClassCount - number of output classes 
    #---- trBatchSize - batch size
    #---- trMaxEpoch - number of epochs
    #---- transResize - size of the image to scale down to (not used in current implementation)
    #---- transCrop - size of the cropped image 
    #---- launchTimestamp - date/time, used to assign unique name for the checkpoint file
    #---- checkpoint - if not None loads the model and continues training
    
    def test (pathDirData, pathFileTest, pathModel, nnArchitecture, nnClassCount, nnIsTrained, trBatchSize, transResize, transCrop, launchTimeStamp):   
        
        
        CLASS_NAMES = [ '0', '1', '2', '3' ]
        
        cudnn.benchmark = True
        
        #-------------------- SETTINGS: NETWORK ARCHITECTURE, MODEL LOAD
        if nnArchitecture == 'DENSE-NET-121': model = DenseNet121(nnClassCount, nnIsTrained).cuda()
        elif nnArchitecture == 'DENSE-NET-169': model = DenseNet169(nnClassCount, nnIsTrained).cuda()
        elif nnArchitecture == 'DENSE-NET-201': model = DenseNet201(nnClassCount, nnIsTrained).cuda()
        elif nnArchitecture == 'DENSE-NET-161': model = DenseNet161(nnClassCount, nnIsTrained).cuda()
        elif nnArchitecture == 'RES-NET-50': model = ResNet50(nnClassCount, nnIsTrained).cuda()
        elif nnArchitecture == 'RES-NET-18': model = ResNet18(nnClassCount, nnIsTrained).cuda()
        elif nnArchitecture == 'VGG-11-BN': model = Vgg11_BN(nnClassCount, nnIsTrained).cuda()
        elif nnArchitecture == 'VGG-16-BN': model = Vgg16_BN(nnClassCount, nnIsTrained).cuda()
        elif nnArchitecture == 'RES-NET-14': model = ResNet14(nnClassCount, nnIsTrained).cuda()
       
        model = torch.nn.DataParallel(model).cuda() 
        
        modelCheckpoint = torch.load(pathModel)
        model.load_state_dict(modelCheckpoint['state_dict'], strict = False)

        #-------------------- SETTINGS: DATA TRANSFORMS, TEN CROPS
        #normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        normalize = transforms.Normalize([0.0807, 0.0577, 0.0899], [0.0642, 0.0501, 0.0540])
        
        #-------------------- SETTINGS: DATASET BUILDERS
        transformList = []
        transformList.append(transforms.Resize(transResize))
        transformList.append(transforms.TenCrop(transCrop))
        transformList.append(transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])))
        transformList.append(transforms.Lambda(lambda crops: torch.stack([normalize(crop) for crop in crops])))
        transformSequence=transforms.Compose(transformList)
        
        datasetTest = DatasetGenerator(pathImageDirectory=pathDirData, pathDatasetFile=pathFileTest, transform=transformSequence)
        dataLoaderTest = DataLoader(dataset=datasetTest, batch_size=trBatchSize, num_workers=8, shuffle=False, pin_memory=True)
        
        outGT = torch.FloatTensor().cuda()
        outPRED = torch.FloatTensor().cuda()
       
        model.eval()
        
        for i, (data_input, target) in enumerate(dataLoaderTest):
            with torch.no_grad():    
                target = target.cuda()
                outGT = torch.cat((outGT, target), 0)
            
                bs, n_crops, c, h, w = data_input.size()
            
            
                varInput = torch.autograd.Variable(data_input.view(-1, c, h, w).cuda())
                out = model(varInput)
                outMean = out.view(bs, n_crops, -1).mean(1)
            
                outPRED = torch.cat((outPRED, outMean.data), 0)
            

        aurocIndividual = ChexnetTrainer.computeAUROC(outGT, outPRED, nnClassCount)
        aurocMean = np.array(aurocIndividual).mean()
        


       
        acc, pre, recall, f1score = ChexnetTrainer.computeAcc_Recall_F1score(outGT, outPRED, nnClassCount)


       
        print ('AUROC mean ', aurocMean)
        
        for i in range (0, len(aurocIndividual)):
            print (CLASS_NAMES[i], ' ', aurocIndividual[i])
       
        print ('\nAcc ')

        for i in range (0, len(aurocIndividual)):
            print (CLASS_NAMES[i], ' ', acc[i])
		
        print ('\nPre ')
        
        for i in range (0, len(aurocIndividual)):
            print (CLASS_NAMES[i], ' ', pre[i])
             
        print ('\nRecall ')
        
        for i in range (0, len(aurocIndividual)):
            print (CLASS_NAMES[i], ' ', recall[i])
             
        print ('\nF1score ')
        
        for i in range (0, len(aurocIndividual)):
            print (CLASS_NAMES[i], ' ', f1score[i])
     
        return
#-------------------------------------------------------------------------------- 

    def PredictWitnModel (pathDirData, pathFileTest, pathModel, nnArchitecture, nnClassCount, nnIsTrained, trBatchSize, transResize, transCrop, launchTimeStamp):   
        
        
        CLASS_NAMES = [ '0', '1', '2', '3', '4' ]
        
        cudnn.benchmark = True
        
        #-------------------- SETTINGS: NETWORK ARCHITECTURE, MODEL LOAD
        if nnArchitecture == 'DENSE-NET-121': model = DenseNet121(nnClassCount, nnIsTrained).cuda()
        elif nnArchitecture == 'DENSE-NET-169': model = DenseNet169(nnClassCount, nnIsTrained).cuda()
        elif nnArchitecture == 'DENSE-NET-201': model = DenseNet201(nnClassCount, nnIsTrained).cuda()
        elif nnArchitecture == 'DENSE-NET-161': model = DenseNet161(nnClassCount, nnIsTrained).cuda()
        elif nnArchitecture == 'RES-NET-50': model = ResNet50(nnClassCount, nnIsTrained).cuda()
        elif nnArchitecture == 'RES-NET-18': model = ResNet18(nnClassCount, nnIsTrained).cuda()
        elif nnArchitecture == 'VGG-11-BN': model = Vgg11_BN(nnClassCount, nnIsTrained).cuda()
        elif nnArchitecture == 'VGG-16-BN': model = Vgg16_BN(nnClassCount, nnIsTrained).cuda()
        
        model = torch.nn.DataParallel(model).cuda() 
        
        modelCheckpoint = torch.load(pathModel)
        model.load_state_dict(modelCheckpoint['state_dict'], strict = False)

        #-------------------- SETTINGS: DATA TRANSFORMS, TEN CROPS
        normalize = transforms.Normalize([0.1772, 0.1917, 0.2516], [0.0154, 0.0282, 0.0111])

        
        #-------------------- SETTINGS: DATASET BUILDERS
        transformList = []
        transformList.append(transforms.Resize(transResize))
        transformList.append(transforms.TenCrop(transCrop))
        transformList.append(transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])))
        transformList.append(transforms.Lambda(lambda crops: torch.stack([normalize(crop) for crop in crops])))
        transformSequence=transforms.Compose(transformList)
        
        datasetTest = DatasetGenerator(pathImageDirectory=pathDirData, pathDatasetFile=pathFileTest, transform=transformSequence)
        dataLoaderTest = DataLoader(dataset=datasetTest, batch_size=trBatchSize, num_workers=8, shuffle=False, pin_memory=True)
        
        outGT = torch.FloatTensor().cuda()
        outPRED = torch.FloatTensor().cuda()
       
        model.eval()
        
        for i, (data_input, target) in enumerate(dataLoaderTest):
            with torch.no_grad():    
                target = target.cuda()
                outGT = torch.cat((outGT, target), 0)
            
                bs, n_crops, c, h, w = data_input.size()
            
            
                varInput = torch.autograd.Variable(data_input.view(-1, c, h, w).cuda())
                out = model(varInput)
                outMean = out.view(bs, n_crops, -1).mean(1)
            
                outPRED = torch.cat((outPRED, outMean.data), 0)
                

        aurocIndividual = ChexnetTrainer.computeAUROC(outGT, outPRED, nnClassCount)
        aurocMean = np.array(aurocIndividual).mean()
        
        acc, pre, recall, f1score, result, ground = ChexnetTrainer.computeAcc_Recall_F1score2(outGT, outPRED, nnClassCount)

		
        return result, ground



