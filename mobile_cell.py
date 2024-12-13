import tensorflow as tf
import numpy as np
from modules import ConvBN,DepthConvBN,ExpDWConvs,SEModule,DepthAttention,gelu


class MobileCell(tf.Module):
    def __init__(self,ops_name,filters,t,strides=(1,1),activation=gelu,name="mobilecell"):
        super(MobileCell,self).__init__(name=name)
        self._ops_name=ops_name
        self._filters=filters
        self._t=t
        self._strides=strides
        self._activation=activation
        self._name=name
    def _Build(self,input_ch):
        if(self._ops_name[1]=="exp_conv"):
            self._exp=ConvBN(int(input_ch*self._t),(1,1),activation=self._activation,name=self._name+"_exp")
        elif(self._ops_name[1]=="exp_dwconv"):
            self._exp=ExpDWConvs(self._t,activation=self._activation,name=self._name+"_exp")

        if(self._ops_name[2]=="dwconv_3x3"):
            self._extract=DepthConvBN((3,3),self._strides,activation=self._activation,name=self._name+"_extract")
        elif(self._ops_name[2]=="dwconv_5x5"):
            self._extract=DepthConvBN((5,5),self._strides,activation=self._activation,name=self._name+"_extract")

        if(self._ops_name[3]=="se_atten"):
            self._atten=SEModule(name=self._name+"_atten")
        elif(self._ops_name[3]=="da_atten"):
            self._atten=DepthAttention(name=self._name+"_atten")

        self._depress=ConvBN(self._filters,(1,1),activation=None,name=self._name+"_depress")
        self._proj=ConvBN(self._filters,(1,1),strides=self._strides,activation=None,name=self._name+"_proj")
        return
    def __call__(self,input_ts):
        input_ch=input_ts.get_shape().as_list()[3]
        self._Build(input_ch)
        x=input_ts
        x=self._exp(x)
        x=self._extract(x)
        x=self._atten(x)
        x=self._depress(x)
        if(input_ch==self._filters and self._strides==(1,1)):
            output_ts=x+input_ts
        else:
            output_ts=x+self._proj(input_ts)
        return output_ts
    
class MobileBlock(tf.Module):
    def __init__(self,ops_name,filters,t,strides=(1,1),blck_len=1,activation=gelu,name="mobileblck"):
        super(MobileBlock,self).__init__(name=name)
        self._ops_name=ops_name
        self._filters=filters
        self._t=t
        self._strides=strides
        self._activation=activation
        self._blck_len=blck_len
        self._activation=activation
        self._name=name
        self._Build()
    @tf.Module.with_name_scope
    def _Build(self):
        self._ithcells=[]
        self._se_layers=[]
        self._firstcell=MobileCell(self._ops_name,
                                   self._filters,
                                   self._t,
                                   self._strides,
                                   self._activation,
                                   name=self._name+"_firstcell")
        for i in range(self._blck_len-1):
            self._ithcells.append(MobileCell(self._ops_name,
                                             self._filters,
                                             self._t,
                                             (1,1),
                                             self._activation,
                                             name=self._name+"_cell"+str(i+1)))
    @tf.Module.with_name_scope
    def __call__(self,input_ts):
        x=self._firstcell(input_ts)
        for i in range(self._blck_len-1):
            x=self._ithcells[i](x)
        output_ts=x
        return output_ts

