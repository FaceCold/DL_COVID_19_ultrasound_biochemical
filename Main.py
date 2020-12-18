import os
import numpy as np
import sys
import time

from NetTester import NetTester

os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3"

#-------------------------------------------------------------------------------- 

def main ():
    

    runTest()



def runTest():
    
    pathDirData = './score_database'
    pathFileTest = './dataset/score_test_evaluation_included2.txt'
    nnArchitecture = 'RES-NET-18'
    nnIsTrained = True
    nnClassCount = 4
    trBatchSize = 64
    imgtransResize = 512
    imgtransCrop = 448
    
    pathModel = './20201020.tar'
    timestampLaunch = ''
    
    NetTester.test(pathDirData, pathFileTest, pathModel, nnArchitecture, nnClassCount, nnIsTrained, trBatchSize, imgtransResize, imgtransCrop, timestampLaunch)

#-------------------------------------------------------------------------------- 

if __name__ == '__main__':
    main()





