import tensorflow as tf
import numpy as np
from modules import ReLUConvBN,Identity,Zeros

class Nas201Cell(tf.keras.layers.Layer):
    def __init__(self,filters,ops_name,adj,name="nas201cell"):
        super(Nas201Cell,self).__init__(name=name)
        self._filters=filters
        self._adj=np.array(adj)
        self._max_nodes=np.shape(adj)[0]
        self._ops_name=ops_name
        self._name=name
    def build(self,input_shape):
        self._ops=[]
        for i,op_name in enumerate(self._ops_name):
            if(op_name=="none" or op_name=="zeros"):
                op=Zeros(name=self._name+"_op_"+str(i))
            elif(op_name=="skip_connect"):
                op=Identity(name=self._name+"_op_"+str(i))
            elif(op_name=="nor_conv_1x1"):
                op=ReLUConvBN(self._filters,(1,1),name=self._name+"_op_"+str(i))
            elif(op_name=="nor_conv_3x3"):
                op=ReLUConvBN(self._filters,(3,3),name=self._name+"_op_"+str(i))
            elif(op_name=="avg_pool_3x3"):
                op=tf.keras.layers.AveragePooling2D((3,3),strides=(1,1),padding="same",name=self._name+"_op_"+str(i))
            else:
                raise Exception("Nas201Cell Build Error: no operation '"+op_name+"' in candidate ops.")
            self._ops.append(op)
        return
    def call(self,input_ts):
        x0=input_ts
        x1=self._ops[0](x0)
        x2=self._ops[1](x0)+self._ops[2](x1)
        x3=self._ops[3](x0)+self._ops[4](x1)+self._ops[5](x2)
        return x3

def GetNas201Cell(ops,adj):
    def _GetNas201Cell(filters,name):
        return Nas201Cell(filters,ops,adj,name)
    return _GetNas201Cell