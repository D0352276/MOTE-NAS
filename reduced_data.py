import tensorflow as tf
import numpy as np
import cv2
import os
from sklearn.cluster import KMeans
from json_io import Dict2JSON,JSON2Dict

def VGG16(input_shape=[224,224,3]):
    input_ts=tf.keras.Input(shape=input_shape)
    model=tf.keras.applications.VGG16(input_tensor=input_ts,include_top=True,weights="imagenet")
    x=model.layers[-5].output
    x=tf.reduce_mean(x,axis=-1)
    x=tf.reshape(x,[-1,49])
    model=tf.keras.Model(inputs=input_ts,outputs=x)
    return model

def Imgs2MeanEbdCode(model,imgs):
    _imgs=[]
    for img in imgs:
        _imgs.append(cv2.resize(img,(224,224)))
    codes=model.predict(np.array(_imgs))
    m_code=np.mean(codes,axis=0)
    return m_code.tolist()

def LoadImgs(cifar100_dir):
    imgs_dict={}
    imgs_name=os.listdir(cifar100_dir)
    for img_name in imgs_name:
        img=cv2.imread(cifar100_dir+"/"+img_name)
        label=img_name.split("_")[0]
        if(label not in imgs_dict):
            imgs_dict[label]=[]
        imgs_dict[label].append(img/255)
    return imgs_dict

def Labels2MeanEbdCode(imgs_dict):
    model=VGG16()
    labels_code_dict={}
    for label in list(imgs_dict.keys()):
        imgs=imgs_dict[label]
        labels_code_dict[label]=Imgs2MeanEbdCode(model,imgs)
    return labels_code_dict

def KLabels(labels_code_dict_path,k):
    labels_code_dict=JSON2Dict(labels_code_dict_path)
    codes=list(labels_code_dict.values())
    kmeans=KMeans(n_clusters=k)
    kmeans.fit(codes)
    centers=kmeans.cluster_centers_
    choose_labels={}
    for i in range(len(centers)):
        choose_labels[i]=[None,0]
    for label in labels_code_dict:
        code=labels_code_dict[label]

        clus_idx=kmeans.predict(np.array([code]))
        clus_idx=clus_idx[0]
        distance=0
        for i,center in enumerate(centers):
            if(i==clus_idx):continue
            distance+=np.sqrt(np.sum((np.array(code)-np.array(center))**2))

        if(distance>choose_labels[clus_idx][1]):
            choose_labels[clus_idx]=[label,distance]

    choosen_labels=[]
    for i in range(len(centers)):
        choosen_labels.append(choose_labels[i][0])
    return choosen_labels

def CIFAR100toRD(cifar100_dir,out_dir,labels_code_dict_path,k):
    choosen_labels=KLabels(labels_code_dict_path,k)
    cifar100_train_dir=cifar100_dir+"/train"
    cifar100_test_dir=cifar100_dir+"/test"
    out_train_dir=out_dir+"/train"
    out_test_dir=out_dir+"/test"
    all_imgs_name=os.listdir(cifar100_train_dir)
    for img_name in all_imgs_name:
        label=img_name.split("_")[0]
        if(label not in choosen_labels):continue
        in_path=cifar100_train_dir+"/"+img_name
        out_path=out_train_dir+"/"+img_name
        img=cv2.imread(in_path)
        cv2.imwrite(out_path,img)
    all_imgs_name=os.listdir(cifar100_test_dir)
    for img_name in all_imgs_name:
        label=img_name.split("_")[0]
        if(label not in choosen_labels):continue
        in_path=cifar100_test_dir+"/"+img_name
        out_path=out_test_dir+"/"+img_name
        img=cv2.imread(in_path)
        cv2.imwrite(out_path,img)
    return 