#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Jan 20 2019

@author: zhaoyu
"""
from config import config
import glob
import os
import string
import numpy as np
import pandas as pd
import SimpleITK as sitk
from train import value_predict

#%%
CurrentTest = 'TestDataFolder'
IfNorm = True
kfold=6

dimz = config['dimz']
dimx = config['dimx']
dimy = config['dimy']
channelNum = config['channelNum']
#%%
def reWriteFiles(hdrFileList, imgFileList, IfNorm):
    arrayDict = dict()
    for i in range(len(hdrFileList)):
        # chage file name of .img
        currentfile = hdrFileList[i]    
        name, ext = os.path.splitext(currentfile)
        Bname = os.path.basename(name)
        Dname = os.path.dirname(currentfile)
        if Bname.startswith('swDP_'):
            item = "_".join(Bname.split("_")[0:2])
        else:
            if len(Bname.split("_"))>1 and (Bname.split("_")[1]).isdigit():
                item = "_".join(Bname.split("_")[0:2])
            else:
                item = "_".join(Bname.split("_")[0:1])        
        # if item.startswith('sw'):
        #     item = item[2:]
        newname = os.path.join(Dname,item+ext)
        os.rename(currentfile, newname)

        # chage file name of .hdr
        currentfile = imgFileList[i]    
        name, ext = os.path.splitext(currentfile)
        Bname = os.path.basename(name)
        Dname = os.path.dirname(currentfile)
        if Bname.startswith('swDP_'):
            item = "_".join(Bname.split("_")[0:2])
        else:
            if len(Bname.split("_"))>1 and (Bname.split("_")[1]).isdigit():
                item = "_".join(Bname.split("_")[0:2])
            else:
                item = "_".join(Bname.split("_")[0:1])       
        # if item.startswith('sw'):
        #     item = item[2:]
        newname = os.path.join(Dname,item+ext)
        os.rename(currentfile, newname)

        currentImg = sitk.ReadImage(newname)
        currentArray = sitk.GetArrayFromImage(currentImg)
        if IfNorm == True:
            mean = np.mean(currentArray)  # mean for data centering
            std = np.std(currentArray)  # std for data normalization
            currentArray -= mean
            currentArray /= std
        arrayDict[item] = currentArray
        if i == 0:
            shapeZ = currentArray.shape[0]
            shapeX = currentArray.shape[1]
            shapeY = currentArray.shape[2]
    return arrayDict, shapeZ, shapeX, shapeY

def WritePredictResultToXLSX(list_ID, Y_predict, Y_predict_name, Y_possibility, outXLSX):
    dataRestore = {'patientID':list_ID, 'patientValue':Y_predict_name, 'Prediction':Y_predict, 'PosMSA':Y_possibility[:,0], 'PosPID':Y_possibility[:,1], 'PosPSP':Y_possibility[:,2]}
    df = pd.DataFrame(data=dataRestore)    
    df.to_excel(outXLSX, index=None,encoding='utf_8_sig', columns=['patientID', 'patientValue', 'Prediction', 'PosMSA', 'PosPID', 'PosPSP'])    

def finalTestDataPrepare(hdrFileList, 
                            imgFileList,
                            tempDataStore, 
                            IfNorm, 
                            IfSave = True):
    arrayDict, shapeZ, shapeX, shapeY = reWriteFiles(hdrFileList, imgFileList, IfNorm)

    # prepare trainging data and label
    sorted_keys = sorted(arrayDict.keys(), reverse=False)
    NN = len(sorted_keys)
    ImgArray = np.zeros((NN,shapeZ,shapeX,shapeY))

    ii=0
    for currentKey in sorted_keys:
        ImgArray[ii,:,:,:] =  arrayDict[currentKey]
        ii += 1

    if IfSave == True:
        if not os.path.exists(tempDataStore):
            os.mkdir(tempDataStore)
        np.save(os.path.join(tempDataStore, 'x_train' + '.npy'), ImgArray)
        np.save(os.path.join(tempDataStore, 'ID_train' + '.npy'), sorted_keys)
    
    return ImgArray, sorted_keys

def oneModelEvaluate(ImgArray, list_ID, BMtype, load_weight_dir, disease_names, outXLSX):    
    y_predict = value_predict(ImgArray, BMtype, load_weight_dir, outputDir=None)
    Y_predict = np.argmax(y_predict, axis=1)
    Y_predict_name = list()
    for item in Y_predict:
        Y_predict_name.append(disease_names[item])
    WritePredictResultToXLSX(list_ID, Y_predict_name, outXLSX)

def MultiModelEvaluate(ImgArray, list_ID, BM_list, weight_dir_list, disease_names, outXLSX):
    y_predict_list = list()
    for i in range(len(weight_dir_list)):
        y_predict = value_predict(ImgArray, BM_list[i], weight_dir_list[i], None)
        y_predict_list.append(y_predict)
        print(i)
    y_predict_mean = np.mean(np.array(y_predict_list), axis=0)
    Y_possibility = y_predict_mean
    Y_predict = np.argmax(Y_possibility, axis=1)
    Y_predict_name = list()
    for item in Y_predict:
        Y_predict_name.append(disease_names[item])
    WritePredictResultToXLSX(list_ID, Y_predict, Y_predict_name, Y_possibility, outXLSX)

if __name__ == "__main__":
    disease_names = ['MSA', 'PD', 'PSP']
      
    #%%
    # original data
    dataDir = os.path.join('../DATASET', CurrentTest)    

    projectStoreRoot = '../ProjectFolder'
    tempDataStore = os.path.join(projectStoreRoot, CurrentTest)
    if not os.path.exists(tempDataStore):
        os.makedirs(tempDataStore)    
    
    Format1 = '/*.hdr'
    hdrFileList = glob.glob((dataDir+Format1))
    hdrFileList.sort()
    Format2 = '/*.img'
    imgFileList = glob.glob((dataDir+Format2))
    imgFileList.sort()

    ImgArray, list_ID = finalTestDataPrepare(hdrFileList, 
                                             imgFileList,
                                             tempDataStore, 
                                             IfNorm, 
                                             IfSave = True)
    # resnew
    GM_singal_resnew = list()
    resnew_list = ['resnew']*6
    rootdir = '../ProjectFolder'
    for currentFold in range(1,kfold+1):
        GM_singal_resnew.append(os.path.join(rootdir,'data_{0}_{1}'.format(kfold,currentFold),'Weights.h5'))

    outXLSX = os.path.join(rootdir, CurrentTest + 'Test_Result.xlsx')
    MultiModelEvaluate(ImgArray, list_ID, resnew_list, GM_singal_resnew, disease_names, outXLSX)