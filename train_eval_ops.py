import numpy as np
import random
import cv2
import os

def GetTrainData(train_data_dir,labels,k=50000):
    imgs=[]
    gt_label_idxs=[]
    img_names=os.listdir(train_data_dir)
    random.shuffle(img_names)
    img_names=img_names[:k]
    for i,img_name in enumerate(img_names):
        gt_label=img_name.split("_")[0]
        gt_label_idx=labels.index(gt_label)
        img=cv2.imread(train_data_dir+"/"+img_name)/255
        gt_label_idxs.append(gt_label_idx)
        imgs.append(img)
    imgs=np.array(imgs)
    gt_label_idxs=np.array(gt_label_idxs)
    return imgs,gt_label_idxs

def GetTestImgs(test_data_dir,labels):
    imgs=[]
    gt_label_idxs=[]
    img_names=os.listdir(test_data_dir)
    img_names=img_names
    for i,img_name in enumerate(img_names):
        gt_label=img_name.split("_")[0]
        gt_label_idx=labels.index(gt_label)
        img=cv2.imread(test_data_dir+"/"+img_name)/255
        gt_label_idxs.append(gt_label_idx)
        imgs.append(img)
    imgs=np.array(imgs)
    gt_label_idxs=np.array(gt_label_idxs)
    return imgs,gt_label_idxs

def TestAccuracy(model,test_imgs,test_label_idxs):
    pred_label_idxs=model.predict(test_imgs)
    pred_label_idxs=np.argmax(pred_label_idxs,axis=-1)

    correct_count=0
    for i,pred_label_idx in enumerate(pred_label_idxs):
        test_label_idx=test_label_idxs[i]
        if(test_label_idx==pred_label_idx):
            correct_count+=1
    acc=correct_count/len(test_label_idxs)  
    return acc

def TestLoss(model,test_imgs,test_label_idxs):
    loss=model.evaluate(test_imgs,test_label_idxs)
    return loss