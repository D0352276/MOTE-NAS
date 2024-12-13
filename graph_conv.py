import tensorflow as tf
from tensorflow.keras import initializers
mish=tf.keras.layers.Lambda(lambda x:x*tf.math.tanh(tf.math.softplus(x)))

class DenseBN(tf.Module):
    def __init__(self,filters,use_bn=True,activation=mish,name="densebn"):
        super(DenseBN,self).__init__(name=name)
        self._filters=filters
        self._use_bn=use_bn
        self._activation=activation
        self._name=name
        self._Build()
    @tf.Module.with_name_scope
    def _Build(self):
        self._dense=tf.keras.layers.Dense(self._filters,activation=None,name=self._name+"_dense")
        if(self._use_bn==True):self._bn=tf.keras.layers.BatchNormalization(momentum=0.997,epsilon=1e-4,name=self._name+"_bn")
        self._act=tf.keras.layers.Activation(self._activation,name=self._name+"_act")
    @tf.Module.with_name_scope
    def __call__(self,input_ts):
        x=self._dense(input_ts)
        if(self._use_bn==True):x=self._bn(x)
        x=self._act(x)
        output_ts=x
        return output_ts  

class NodesTransform(tf.keras.layers.Layer):
    def __init__(self,embd_len,activation=mish,use_res=True,name="nodestransform"):
        super(NodesTransform,self).__init__()
        self._embd_len=embd_len
        self._activation=activation
        self._use_res=use_res
        self._name=name
    def build(self,input_shape):
        self._feats_len=input_shape[2]
        self._weight=self.add_weight(name="weight",
                                    shape=(self._feats_len,self._embd_len),
                                    initializer=initializers.get("glorot_uniform"),
                                    constraint=None,
                                    trainable=True)

        self._act=tf.keras.layers.Activation(self._activation,name="act")
        super(NodesTransform,self).build(input_shape)
    def call(self,input_ts):
        in_mat=input_ts
        out_mat=tf.matmul(in_mat,self._weight)
        out_mat=self._act(out_mat)
        if(self._feats_len==self._embd_len and self._use_res==True):
            out_mat=out_mat+in_mat
        output=out_mat
        return output

class GraphConv(tf.keras.layers.Layer):
    def __init__(self,filters,activation=mish,use_res=True,name="graphconv"):
        super(GraphConv,self).__init__()
        self._filters=filters
        self._activation=activation
        self._use_res=use_res
        self._name=name
    def build(self,input_shape):
        self._feats_len=input_shape[1][2]
        self._weight=self.add_weight(name="weight",
                                    shape=(self._feats_len,self._filters),
                                    initializer=initializers.get("glorot_uniform"),
                                    constraint=None,
                                    trainable=True)
        self._act=tf.keras.layers.Activation(self._activation,name="act")
        super(GraphConv,self).build(input_shape)
    def call(self,input_ts):
        adj_mat,op_mat=input_ts
        hid_mat=tf.matmul(op_mat,self._weight)
        out_mat=tf.matmul(adj_mat,hid_mat)
        out_mat=self._act(out_mat)
        output=out_mat
        return output

class BIGraphConv(tf.keras.layers.Layer):
    def __init__(self,filters,activation=mish,name="bigconv"):
        super(BIGraphConv,self).__init__()
        self._filters=filters
        self._activation=activation
        self._name=name
    def build(self,input_shape):
        self._gcnconv_1=GraphConv(self._filters,activation=self._activation,name="_gcnconv_1")
        self._gcnconv_2=GraphConv(self._filters,activation=self._activation,name="_gcnconv_2")
        super(BIGraphConv,self).build(input_shape)
    def call(self,input_ts):
        adj,x=input_ts
        x1=self._gcnconv_1([adj,x])
        x2=self._gcnconv_2([tf.transpose(adj,[0,2,1]),x])
        out_ts=(x1+x2)/2
        return out_ts

