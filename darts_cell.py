import tensorflow as tf
import numpy as np
import random
from modules import SepConvBN,DilConvBN,Identity,Zeros,ConvBN

class DARTSCell(tf.keras.layers.Layer):
    def __init__(self,filters,ops_name,adj,name="dartscell"):
        super(DARTSCell,self).__init__(name=name)
        self._filters=filters
        self._ops_name=ops_name
        self._adj=adj
        self._name=name
    def build(self,input_shape):
        self._ops=[]
        strides=(1,1)
        for i,op_name in enumerate(self._ops_name):
                
            if(op_name=="none" or op_name=="zeros"):
                op=Zeros(name=self._name+"_op_"+str(i))
            elif(op_name=="skip_connect"):
                op=Identity(name=self._name+"_op_"+str(i))
            elif(op_name=="sep_conv_3x3"):
                op=SepConvBN(self._filters,(3,3),strides=strides,name=self._name+"_op_"+str(i))
            elif(op_name=="sep_conv_5x5"):
                op=SepConvBN(self._filters,(5,5),strides=strides,name=self._name+"_op_"+str(i))
            elif(op_name=="dil_conv_3x3"):
                op=DilConvBN(self._filters,(3,3),strides=strides,name=self._name+"_op_"+str(i))
            elif(op_name=="dil_conv_5x5"):
                op=DilConvBN(self._filters,(5,5),strides=strides,name=self._name+"_op_"+str(i))
            elif(op_name=="avg_pool_3x3"):
                op=tf.keras.layers.AveragePooling2D((3,3),strides=strides,padding="same",name=self._name+"_op_"+str(i))
            elif(op_name=="max_pool_3x3"):
                op=tf.keras.layers.MaxPooling2D((3,3),strides=strides,padding="same",name=self._name+"_op_"+str(i))
            else:
                raise Exception("Nas201Cell Build Error: no operation '"+op_name+"' in candidate ops.")
            self._ops.append(op)
        return
    def call(self,input_ts):
        x0,x1=input_ts
        x2=self._ops[0](x0)+self._ops[1](x1)
        x3=self._ops[2](x0)+self._ops[3](x1)+self._ops[4](x2)
        x4=self._ops[5](x0)+self._ops[6](x1)+self._ops[7](x2)+self._ops[8](x3)
        x5=self._ops[9](x0)+self._ops[10](x1)+self._ops[11](x2)+self._ops[12](x3)+self._ops[13](x4)
        out_ts=x2+x3+x4+x5
        return out_ts

def GetDARTSCell(ops,adj):
    def _GetDARTSCell(filters,name):
        return DARTSCell(filters,ops,adj,name)
    return _GetDARTSCell

def RandomDARTSCell():
    candidate_ops=["none","skip_connect","sep_conv_3x3","sep_conv_5x5","dil_conv_3x3","dil_conv_5x5","avg_pool_3x3","max_pool_3x3"]
    ops=random.choices(candidate_ops,k=6)
    return GetDARTSCell(ops)



