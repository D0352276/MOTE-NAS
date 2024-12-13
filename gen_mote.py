import os
import time
import random 
import cv2
import numpy as np
import tensorflow as tf
from json_io import Dict2JSON,JSON2Dict
from nas_prcss import CellPth2Cell,CellPths2Cells
from nas101_cell import GetNas101Cell
from nas201_cell import GetNas201Cell
from darts_cell import GetDARTSCell
from meta_models import CreateTinyMetaModel,CreateTinyDARTSMetaModel,CreateMobileMetaModel,CompileModel
from model_operation import Training
from callbacks import TimeClock,LossRecorder,NanChecker
from scipy.stats import boxcox
from flops import FLOPs

def GetTrainData(train_data_dir,labels,k=-1):
    imgs=[]
    gt_label_idxs=[]
    img_names=os.listdir(train_data_dir)
    random.shuffle(img_names)
    if(k>=0):
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

def GetTestData(train_data_dir,labels,k=-1):
    return GetTrainData(train_data_dir,labels,k)

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

def GetProxyC100TrainData(labels_len=10):
    if(labels_len==2):
        labels=['snail','man']
    elif(labels_len==5):
        labels=['clock','poppy','cockroach','pickuptruck','cloud']
    elif(labels_len==10): 
        labels=['sea','seal','willowtree','tractor','orange','caterpillar','pear','mapletree','beaver','poppy']
    elif(labels_len==20):
        labels=['aquariumfish','motorcycle','hamster','sea','cockroach','bottle','whale','telephone','dinosaur','tank','wardrobe','rocket','cloud','oaktree','lawnmower','willowtree','clock','poppy','bowl','plate']
    elif(labels_len==30):
        labels=['plate', 'beetle', 'cloud', 'mountain', 'willowtree', 'wardrobe', 'dolphin', 'apple', 'poppy', 'bottle', 'clock', 'lawnmower', 'rocket', 'worm', 'oaktree', 'motorcycle', 'whale', 'tank', 'telephone', 'pickuptruck', 'bicycle', 'cockroach', 'skyscraper', 'sea', 'bowl', 'television', 'trout', 'house', 'can', 'cup']
    elif(labels_len==40):
        labels=['porcupine', 'plate', 'beetle', 'cloud', 'mountain', 'willowtree', 'bus', 'wardrobe', 'chair', 'snail', 'leopard', 'apple', 'poppy', 'bottle', 'clock', 'lawnmower', 'rocket', 'oaktree', 'forest', 'motorcycle', 'whale', 'tank', 'castle', 'telephone', 'pickuptruck', 'bicycle', 'cockroach', 'plain', 'sea', 'bowl', 'television', 'girl', 'woman', 'trout', 'aquariumfish', 'bridge', 'house', 'can', 'dinosaur', 'cup']
    elif(labels_len==50):
        labels=['porcupine', 'plate', 'beetle', 'orange', 'cloud', 'willowtree', 'bus', 'lizard', 'wardrobe', 'lamp', 'chair', 'snail', 'dolphin', 'apple', 'shark', 'poppy', 'bottle', 'clock', 'lawnmower', 'rocket', 'worm', 'oaktree', 'forest', 'crocodile', 'motorcycle', 'whale', 'chimpanzee', 'tank', 'road', 'castle', 'telephone', 'hamster', 'pickuptruck', 'lobster', 'crab', 'lion', 'bicycle', 'cockroach', 'plain', 'keyboard', 'sea', 'bowl', 'television', 'trout', 'aquariumfish', 'cattle', 'house', 'can', 'dinosaur', 'cup']
    elif(labels_len==60):
        labels=['porcupine', 'plate', 'beetle', 'orange', 'cloud', 'willowtree', 'bus', 'lizard', 'wardrobe', 'lamp', 'chair', 'snail', 'dolphin', 'leopard', 'apple', 'man', 'shark', 'poppy', 'bottle', 'mapletree', 'clock', 'lawnmower', 'rocket', 'worm', 'oaktree', 'table', 'forest', 'crocodile', 'motorcycle', 'whale', 'chimpanzee', 'tank', 'road', 'castle', 'telephone', 'hamster', 'pickuptruck', 'crab', 'lion', 'bicycle', 'cockroach', 'plain', 'keyboard', 'skyscraper', 'sea', 'train', 'bee', 'bowl', 'television', 'woman', 'trout', 'aquariumfish', 'cattle', 'tiger', 'boy', 'house', 'can', 'dinosaur', 'cup', 'skunk']
    elif(labels_len==80):
        labels=['porcupine', 'palmtree', 'bed', 'plate', 'tractor', 'beetle', 'orange', 'shrew', 'cloud', 'mountain', 'willowtree', 'bus', 'lizard', 'wardrobe', 'lamp', 'chair', 'snail', 'orchid', 'dolphin', 'leopard', 'apple', 'man', 'streetcar', 'squirrel', 'sweetpepper', 'butterfly', 'shark', 'poppy', 'bottle', 'mapletree', 'clock', 'lawnmower', 'rocket', 'worm', 'oaktree', 'table', 'forest', 'crocodile', 'motorcycle', 'whale', 'chimpanzee', 'raccoon', 'tank', 'road', 'castle', 'telephone', 'hamster', 'pickuptruck', 'lobster', 'crab', 'lion', 'bicycle', 'mushroom', 'cockroach', 'plain', 'snake', 'skyscraper', 'sea', 'train', 'couch', 'bee', 'bowl', 'television', 'flatfish', 'spider', 'woman', 'beaver', 'baby', 'sunflower', 'trout', 'aquariumfish', 'cattle', 'tiger', 'bridge', 'boy', 'house', 'can', 'dinosaur', 'cup', 'skunk']
    elif(labels_len==100):
        labels=list(JSON2Dict("labels_code.json").keys())
    return GetTrainData("reduced_data/proxy_cifar100-"+str(labels_len)+"/train",labels)

def GetProxyC100TestData(labels_len=10):
    if(labels_len==2):
        labels=['snail','man']
    elif(labels_len==5):
        labels=['clock','poppy','cockroach','pickuptruck','cloud']
    elif(labels_len==10): 
        labels=['sea','seal','willowtree','tractor','orange','caterpillar','pear','mapletree','beaver','poppy']
    elif(labels_len==20):
        labels=['aquariumfish','motorcycle','hamster','sea','cockroach','bottle','whale','telephone','dinosaur','tank','wardrobe','rocket','cloud','oaktree','lawnmower','willowtree','clock','poppy','bowl','plate']
    elif(labels_len==30):
        labels=['plate', 'beetle', 'cloud', 'mountain', 'willowtree', 'wardrobe', 'dolphin', 'apple', 'poppy', 'bottle', 'clock', 'lawnmower', 'rocket', 'worm', 'oaktree', 'motorcycle', 'whale', 'tank', 'telephone', 'pickuptruck', 'bicycle', 'cockroach', 'skyscraper', 'sea', 'bowl', 'television', 'trout', 'house', 'can', 'cup']
    elif(labels_len==40):
        labels=['porcupine', 'plate', 'beetle', 'cloud', 'mountain', 'willowtree', 'bus', 'wardrobe', 'chair', 'snail', 'leopard', 'apple', 'poppy', 'bottle', 'clock', 'lawnmower', 'rocket', 'oaktree', 'forest', 'motorcycle', 'whale', 'tank', 'castle', 'telephone', 'pickuptruck', 'bicycle', 'cockroach', 'plain', 'sea', 'bowl', 'television', 'girl', 'woman', 'trout', 'aquariumfish', 'bridge', 'house', 'can', 'dinosaur', 'cup']
    elif(labels_len==50):
        labels=['porcupine', 'plate', 'beetle', 'orange', 'cloud', 'willowtree', 'bus', 'lizard', 'wardrobe', 'lamp', 'chair', 'snail', 'dolphin', 'apple', 'shark', 'poppy', 'bottle', 'clock', 'lawnmower', 'rocket', 'worm', 'oaktree', 'forest', 'crocodile', 'motorcycle', 'whale', 'chimpanzee', 'tank', 'road', 'castle', 'telephone', 'hamster', 'pickuptruck', 'lobster', 'crab', 'lion', 'bicycle', 'cockroach', 'plain', 'keyboard', 'sea', 'bowl', 'television', 'trout', 'aquariumfish', 'cattle', 'house', 'can', 'dinosaur', 'cup']
    elif(labels_len==60):
        labels=['porcupine', 'plate', 'beetle', 'orange', 'cloud', 'willowtree', 'bus', 'lizard', 'wardrobe', 'lamp', 'chair', 'snail', 'dolphin', 'leopard', 'apple', 'man', 'shark', 'poppy', 'bottle', 'mapletree', 'clock', 'lawnmower', 'rocket', 'worm', 'oaktree', 'table', 'forest', 'crocodile', 'motorcycle', 'whale', 'chimpanzee', 'tank', 'road', 'castle', 'telephone', 'hamster', 'pickuptruck', 'crab', 'lion', 'bicycle', 'cockroach', 'plain', 'keyboard', 'skyscraper', 'sea', 'train', 'bee', 'bowl', 'television', 'woman', 'trout', 'aquariumfish', 'cattle', 'tiger', 'boy', 'house', 'can', 'dinosaur', 'cup', 'skunk']
    elif(labels_len==80):
        labels=['porcupine', 'palmtree', 'bed', 'plate', 'tractor', 'beetle', 'orange', 'shrew', 'cloud', 'mountain', 'willowtree', 'bus', 'lizard', 'wardrobe', 'lamp', 'chair', 'snail', 'orchid', 'dolphin', 'leopard', 'apple', 'man', 'streetcar', 'squirrel', 'sweetpepper', 'butterfly', 'shark', 'poppy', 'bottle', 'mapletree', 'clock', 'lawnmower', 'rocket', 'worm', 'oaktree', 'table', 'forest', 'crocodile', 'motorcycle', 'whale', 'chimpanzee', 'raccoon', 'tank', 'road', 'castle', 'telephone', 'hamster', 'pickuptruck', 'lobster', 'crab', 'lion', 'bicycle', 'mushroom', 'cockroach', 'plain', 'snake', 'skyscraper', 'sea', 'train', 'couch', 'bee', 'bowl', 'television', 'flatfish', 'spider', 'woman', 'beaver', 'baby', 'sunflower', 'trout', 'aquariumfish', 'cattle', 'tiger', 'bridge', 'boy', 'house', 'can', 'dinosaur', 'cup', 'skunk']
    elif(labels_len==100):
        labels=list(JSON2Dict("labels_code.json").keys())
    return GetTrainData("reduced_data/proxy_cifar100-"+str(labels_len)+"/test",labels)

def Cell2Function(cell,cell_pth_type="nas201"):
    if(cell_pth_type=="nas101"):
        get_cell_function=GetNas101Cell(cell["operations"],cell["adj_matrix"])
    elif(cell_pth_type=="nas201"):
        get_cell_function=GetNas201Cell(cell["operations"],cell["adj_matrix"])
    elif(cell_pth_type=="darts"):
        norm_cell_function=GetDARTSCell(cell["norm_operations"],cell["norm_adj_matrix"])
        rdce_cell_function=GetDARTSCell(cell["rdce_operations"],cell["rdce_adj_matrix"])
        get_cell_function=(norm_cell_function,rdce_cell_function)
    return get_cell_function

def Cell2TinyMetaModel(cell,cell_pth_type="nas201",labels_len=10):
    get_cell_function=Cell2Function(cell,cell_pth_type)
    if(cell_pth_type=="darts"):
        norm_cell_function,rdce_cell_function=get_cell_function
        tiny_meta_model=CreateTinyDARTSMetaModel(norm_cell_function,rdce_cell_function,labels_len)
    else:
        tiny_meta_model=CreateTinyMetaModel(get_cell_function,labels_len)
    tiny_meta_model=CompileModel(tiny_meta_model,lr=0.001)
    return tiny_meta_model

def Cell2MobileMetaModel(cell,labels_len=10):
    meta_model=CreateMobileMetaModel(cell["operations"],1.0,labels_len)
    flops=FLOPs(meta_model)/1024/1024
    alpha=2-flops/2
    meta_model=CreateMobileMetaModel(cell["operations"],alpha,labels_len)
    meta_model=CompileModel(meta_model,lr=0.001)
    return meta_model

def LandscapeLosses(train_x,train_y,model,init_whts,cnvg_whts,grain_size=0.1):
    landscape_losses=[]
    for alpha in range(0,int(1/grain_size)+1):
        alpha*=grain_size
        intrplt_whts=(1-alpha)*init_whts+alpha*cnvg_whts
        model.set_weights(intrplt_whts)
        hist=model.fit(train_x,train_y,batch_size=1024,epochs=1,verbose=0)
        loss=hist.history["loss"][0]
        landscape_losses.append(loss)
    return landscape_losses

def Cell2TraingLosses(train_x,train_y,cell,cell_pth_type="nas201",proxy_labels_len=10):
    if(cell["dirty_bit"]==1):return cell
    batch_size=1024
    epochs=50

    if(cell_pth_type=="nasmob"):tiny_meta_model=Cell2MobileMetaModel(cell,proxy_labels_len) #Reduced Arch
    else:tiny_meta_model=Cell2TinyMetaModel(cell,cell_pth_type,proxy_labels_len) #Reduced Arch
    init_whts=tiny_meta_model.get_weights()
    init_whts=np.array(init_whts)

    while(1):
        nan_checker=NanChecker()
        loss_recd=LossRecorder()
        time_clock=TimeClock()
        Training(tiny_meta_model,(train_x,train_y),batch_size=batch_size,epochs=epochs,verbose=0,callbacks=[time_clock,loss_recd,nan_checker])
        if(nan_checker.Check()==False):
            break

    cnvg_whts=tiny_meta_model.get_weights()
    cnvg_whts=np.array(cnvg_whts)
    cell["landscape_losses"]=LandscapeLosses(train_x,train_y,tiny_meta_model,init_whts,cnvg_whts)
    losses=loss_recd.GetLosses()
    cell["proxy_losses"]=losses
    cost_time=time_clock.TimeConsume()
    cell["proxy_train_time"]=cost_time
    cell["proxy_train_epochs"]=epochs
    cell["dirty_bit"]=1
    return cell

def Cell2Terms(cell):
    losses=cell["proxy_losses"]
    train_time=cell["proxy_train_time"]
    landscape_losses=cell["landscape_losses"]
    landscape_term=0

    for i,landscape_loss in enumerate(landscape_losses):
        landscape_term+=1/(landscape_loss)

    speed_term=train_time/(sum(losses)/len(losses))
    # speed_term=train_time/(sum(landscape_losses)/len(landscape_losses))

    cell["landscape_term"]=landscape_term
    cell["speed_term"]=speed_term
    cell["mote"]=landscape_term+speed_term
    return cell

def CellsBoxCoxParams(cells):
    landscape_terms=[]
    sp_terms=[]
    for cell in cells:
        landscape_terms.append(cell["landscape_term"])
        sp_terms.append(cell["speed_term"])
    landscape_terms,l_lam=boxcox(landscape_terms)
    sp_terms,s_lam=boxcox(sp_terms)
    landscape_terms=np.array(landscape_terms)
    sp_terms=np.array(sp_terms)

    l_std=np.std(landscape_terms)
    s_std=np.std(sp_terms)
    return l_lam,l_std,s_lam,s_std

def CellPthssBoxCoxParams(cell_pths):
    cells=CellPths2Cells(cell_pths)
    return CellsBoxCoxParams(cells)

#nas101
def Cell2MOTE101(cell,l_lam=0.397,l_std=0.463,s_lam=0.714,s_std=1.538):
    landscape_term=(boxcox(cell["landscape_term"],l_lam))
    speed_term=(boxcox(cell["speed_term"],s_lam))
    cell["mote"]=(l_lam**2*l_std)*landscape_term+(s_lam**2*s_std)*speed_term
    return cell

#nas201    
def Cell2MOTE201(cell,l_lam=0.86,l_std=1.216,s_lam=0.617,s_std=0.961):
    landscape_term=(boxcox(cell["landscape_term"],l_lam))
    speed_term=(boxcox(cell["speed_term"],s_lam))
    cell["mote"]=(l_lam**2*l_std)*landscape_term+(s_lam**2*s_std)*speed_term
    return cell

#nas301
def Cell2MOTE301(cell,l_lam=0.752,l_std=1.173,s_lam=0.258,s_std=0.854):
    landscape_term=(boxcox(cell["landscape_term"],l_lam))
    speed_term=(boxcox(cell["speed_term"],s_lam))
    cell["mote"]=(l_lam**2*l_std)*landscape_term+(s_lam**2*s_std)*speed_term
    return cell

def CellPth2MOTE(train_x,train_y,cell_pth,cell_pth_type="nas201",proxy_labels_len=10):
    cell=CellPth2Cell(cell_pth)
    cell=Cell2TraingLosses(train_x,train_y,cell,cell_pth_type,proxy_labels_len)
    cell=Cell2Terms(cell)
    if(cell_pth_type=="nas101"):
        cell=Cell2MOTE101(cell)
    elif(cell_pth_type=="nas201"):
        cell=Cell2MOTE201(cell)
    elif(cell_pth_type=="nas301"):
        cell=Cell2MOTE301(cell)
    Dict2JSON(cell,cell_pth)
    return cell_pth

def CellPths2MOTE(train_x,train_y,cell_pths,cell_pth_type="nas201",proxy_labels_len=10):
    for cell_pth in cell_pths:
        CellPth2MOTE(train_x,train_y,cell_pth,cell_pth_type,proxy_labels_len)
    return cell_pths
