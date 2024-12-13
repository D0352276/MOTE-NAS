import tensorflow as tf
from json_io import Dict2JSON,JSON2Dict
from modules import ConvBN,DepthConvBN,gelu
from mobile_cell import MobileBlock
def GenAllOperations(save_dir):
    exps=["exp_conv","exp_dwconv"]
    extracts=["dwconv_3x3","dwconv_5x5"]
    attens=["se_atten","da_atten"]

    ops_list=[]
    for exp in exps:
        for extract in extracts:
            for atten in attens:
                ops_list.append(["input",exp,extract,atten,"output"])
    
    all_ops_list=[]
    for ops1 in ops_list:
        for ops2 in ops_list:
            for ops3 in ops_list:
                for ops4 in ops_list:
                    for ops5 in ops_list:
                        all_ops_list.append([ops1,ops2,ops3,ops4,ops5])
    for i,ops_list in enumerate(all_ops_list):
        cell={}
        cell["id"]=i
        cell["operations"]=ops_list
        Dict2JSON(cell,save_dir+"/"+str(i)+".json")
    return

class MobileNetVX(tf.Module):
    def __init__(self,ops_name_list,alpha=1.0,name="mobilenetvx"):
        super(MobileNetVX,self).__init__(name=name)
        self._ops_name_list=ops_name_list
        self._alpha=alpha
        self._name=name
        self._Build()
    @tf.Module.with_name_scope
    def _Build(self):
        self._conv=ConvBN(int(32*self._alpha),(3,3),(2,2),activation=gelu,name=self._name+"_conv")
        self._dwconv=DepthConvBN((3,3),activation=gelu,name=self._name+"_dwconv")
        self._conv2=ConvBN(int(32*self._alpha),(1,1),(1,1),activation=None,name=self._name+"_conv2")
        self._irb1=MobileBlock(self._ops_name_list[0],int(64*self._alpha),t=1,strides=(2,2),blck_len=2,activation=gelu,name=self._name+"_irb1")
        self._irb2=MobileBlock(self._ops_name_list[1],int(96*self._alpha),t=2,strides=(2,2),blck_len=3,activation=gelu,name=self._name+"_irb2")
        self._irb3=MobileBlock(self._ops_name_list[2],int(128*self._alpha),t=3,strides=(2,2),blck_len=4,activation=gelu,name=self._name+"_irb3")
        self._irb4=MobileBlock(self._ops_name_list[3],int(196*self._alpha),t=3,strides=(1,1),blck_len=2,activation=gelu,name=self._name+"_irb4")
        self._irb5=MobileBlock(self._ops_name_list[4],int(256*self._alpha),t=3,strides=(2,2),blck_len=3,activation=gelu,name=self._name+"_irb5")
        self._convout=ConvBN(1024,(1,1),(1,1),activation=gelu,name=self._name+"_convout")
        self._gap=tf.keras.layers.GlobalAveragePooling2D(keepdims=True,name=self._name+"_gap")
        self._convout2=ConvBN(2048,(1,1),(1,1),activation=gelu,use_bn=True,bias=True,name=self._name+"_convout2")
        self._flatten=tf.keras.layers.Flatten(name=self._name+"_flatten")
        self._dropout=tf.keras.layers.Dropout(0.2,name=self._name+"_dropout")
    @tf.Module.with_name_scope
    def __call__(self,input_ts):
        _x=self._conv(input_ts)
        x=self._dwconv(_x)
        x=self._conv2(x)+_x
        x=self._irb1(x)
        x=self._irb2(x)
        x=self._irb3(x)
        x=self._irb4(x)
        x=self._irb5(x)
        x=self._convout(x)
        x=self._gap(x)
        x=self._convout2(x)
        x=self._flatten(x)
        x=self._dropout(x)
        output_ts=x
        return output_ts
        
# GenAllOperations("data/nasbenchmob")