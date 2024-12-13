import tensorflow as tf
import numpy as np
from modules import ConvBN,Identity

class Nas101Cell(tf.Module):
    def __init__(self,filters,ops,adj,name="nas101cell"):
        super(Nas101Cell,self).__init__(name=name)
        self._filters=filters
        self._adj=np.array(adj)
        self._max_nodes=np.shape(adj)[0]
        self._ops=ops
        self._name=name
    @tf.Module.with_name_scope
    def _Build(self,ops,in_ch):
        self._filters=in_ch//2
        self._ts_ops=[]
        op=None
        for i,chosen_op in enumerate(ops):
            if(chosen_op=="input"):
                op=Identity(name=self._name+"_op_"+str(i))
            elif(chosen_op=="conv3x3-bn-relu"):
                op=ConvBN(self._filters,(3,3),name=self._name+"_op_"+str(i))
            elif(chosen_op=="conv1x1-bn-relu"):
                op=ConvBN(self._filters,(1,1),name=self._name+"_op_"+str(i))
            elif(chosen_op=="maxpool3x3"):
                op=tf.keras.layers.MaxPool2D(pool_size=(3,3),strides=(1,1),padding="same",name=self._name+"_op"+str(i))
            elif(chosen_op=="output"):
                op=Identity(name=self._name+"_op_"+str(i))
            else:
                raise Exception("Nas101Cell Error:"+chosen_op+" does not exist.")
            self._ts_ops.append(op)
        return
    @tf.Module.with_name_scope
    def __call__(self,input_ts):
        in_ch=input_ts.get_shape().as_list()[3]
        self._Build(self._ops,in_ch)

        cur_ts_list=[[] for i in range(self._max_nodes)]
        cur_ts_list[0].append(input_ts)
        for i,chosen_op in enumerate(self._ops):
            if(chosen_op=="input"):
                x=cur_ts_list[i][0]
                target_idxes=np.where(self._adj[i]==1)[0]
                for j,target_idx in enumerate(target_idxes):
                    target_ts=ConvBN(in_ch//2,(1,1),name=self._name+"_in_proj_"+str(i)+str(j))(x)
                    cur_ts_list[target_idx].append(target_ts)
            elif(chosen_op!="output"):
                if(len(cur_ts_list[i])>1):
                    x=tf.keras.layers.Add()(cur_ts_list[i])
                else:
                    x=cur_ts_list[i][0]
                target_ts=self._ts_ops[i](x)
                target_idxes=np.where(self._adj[i]==1)[0]
                for target_idx in target_idxes:
                    cur_ts_list[target_idx].append(target_ts)
            else:
                if(self._adj[0][i]==1):
                    x=tf.concat(cur_ts_list[i][1:],axis=-1)
                    out_ch=x.get_shape().as_list()[3]
                    output_ts=x+ConvBN(out_ch,(1,1),name=self._name+"_in_proj")(cur_ts_list[i][0])
                else:
                    output_ts=tf.concat(cur_ts_list[i],axis=-1)
        return output_ts

def GetNas101Cell(ops,adj):
    def _GetNas101Cell(filters,name):
        return Nas101Cell(filters,ops,adj,name)
    return _GetNas101Cell



