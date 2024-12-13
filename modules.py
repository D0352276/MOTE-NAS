import tensorflow as tf
import numpy as np
gelu=tf.keras.layers.Lambda(lambda x:0.5*x*(1+tf.math.tanh(tf.math.sqrt(np.math.pi/2)*(x+0.044715*x**3))))
hard_sigmoid=tf.keras.layers.Lambda(lambda x:tf.nn.relu6(x+3.0)/6.0)
hard_swish=tf.keras.layers.Lambda(lambda x:x*(tf.nn.relu6(x+3.0)/6.0))

class ConvBN(tf.Module):
    def __init__(self,
                 filters,
                 kernel_size,
                 strides=(1,1),
                 padding="same",
                 bias=False,
                 use_bn=True,
                 activation=gelu,
                 name="convbn"):
        super(ConvBN,self).__init__()
        self._filters=filters
        self._kernel_size=kernel_size
        self._strides=strides
        self._padding=padding
        self._bias=bias
        self._use_bn=use_bn
        self._activation=activation
        self._name=name
        self._Build()
    @tf.Module.with_name_scope
    def _Build(self):
        self._conv=tf.keras.layers.Conv2D(filters=self._filters,
                                          kernel_size=self._kernel_size,
                                          strides=self._strides,
                                          padding=self._padding,
                                          use_bias=self._bias,
                                          name=self._name+"_conv")
        if(self._use_bn==True):self._bn=tf.keras.layers.BatchNormalization(momentum=0.997,epsilon=1e-4,name=self._name+"_bn")
        self._act=tf.keras.layers.Activation(self._activation,name=self._name+"_act")
    @tf.Module.with_name_scope
    def __call__(self,input_ts):
        x=self._conv(input_ts)
        if(self._use_bn==True):x=self._bn(x)
        x=self._act(x)
        output_ts=x
        return output_ts

class DenseBN(tf.Module):
    def __init__(self,filters,use_bn=True,activation=gelu,name="convbn"):
        self._filters=filters
        self._use_bn=use_bn
        self._activation=activation
        self._name=name
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

class DepthConvBN(tf.Module):
    def __init__(self,
                 kernel_size,
                 strides=(1,1),
                 padding="same",
                 bias=False,
                 use_bn=True,
                 activation=gelu,
                 name="depthconvbn"):
        super(DepthConvBN,self).__init__(name=name)
        self._kernel_size=kernel_size
        self._strides=strides
        self._padding=padding
        self._bias=bias
        self._use_bn=use_bn
        self._activation=activation
        self._name=name
        self._Build()
    @tf.Module.with_name_scope
    def _Build(self):
        self._depthconv=tf.keras.layers.DepthwiseConv2D(self._kernel_size,
                                                        self._strides,
                                                        depth_multiplier=1,
                                                        padding=self._padding,
                                                        use_bias=self._bias,
                                                        name=self._name+"_depthconv")
        if(self._use_bn==True):self._bn=tf.keras.layers.BatchNormalization(momentum=0.997,epsilon=1e-4,name=self._name+"_bn")
        self._act=tf.keras.layers.Activation(self._activation,name=self._name+"_act")
    @tf.Module.with_name_scope
    def __call__(self,input_ts):
        x=self._depthconv(input_ts)
        if(self._use_bn==True):x=self._bn(x)
        x=self._act(x)
        output_ts=x
        return output_ts
    
class SepConvBN(tf.Module):
    def __init__(self,
                 filters,
                 kernel_size,
                 strides=(1,1),
                 padding="same",
                 bias=False,
                 use_bn=True,
                 activation=None,
                 name="spbconvbn"):
        super(SepConvBN,self).__init__(name=name)
        self._filters=filters
        self._kernel_size=kernel_size
        self._strides=strides
        self._padding=padding
        self._bias=bias
        self._use_bn=use_bn
        self._activation=activation
        self._name=name
        self._Build()
    @tf.Module.with_name_scope
    def _Build(self):
        self._spbconv=tf.keras.layers.SeparableConv2D(self._filters,
                                                      self._kernel_size,
                                                      self._strides,
                                                      depth_multiplier=1,
                                                      padding=self._padding,
                                                      use_bias=self._bias,
                                                      name=self._name+"_spbconv")
        if(self._use_bn==True):self._bn=tf.keras.layers.BatchNormalization(momentum=0.997,epsilon=1e-4,name=self._name+"_bn")
        self._act=tf.keras.layers.Activation(self._activation,name=self._name+"_act")
    @tf.Module.with_name_scope
    def __call__(self,input_ts):
        x=self._spbconv(input_ts)
        if(self._use_bn==True):x=self._bn(x)
        x=self._act(x)
        output_ts=x
        return output_ts
    
class DilConvBN(tf.Module):
    def __init__(self,
                 filters,
                 kernel_size,
                 dilation_rate=(2,2),
                 strides=(1,1),
                 padding="same",
                 bias=False,
                 use_bn=True,
                 activation=gelu,
                 name="dilconvbn"):
        super(DilConvBN,self).__init__(name=name)
        self._filters=filters
        self._kernel_size=kernel_size
        self._dilation_rate=dilation_rate
        self._strides=strides
        self._padding=padding
        self._bias=bias
        self._use_bn=use_bn
        self._activation=activation
        self._name=name
        self._Build()
    @tf.Module.with_name_scope
    def _Build(self):
        self._spbconv=tf.keras.layers.SeparableConv2D(self._filters,
                                                      self._kernel_size,
                                                      self._strides,
                                                      dilation_rate=self._dilation_rate,
                                                      depth_multiplier=1,
                                                      padding=self._padding,
                                                      use_bias=self._bias,
                                                      name=self._name+"_spbconv")
        if(self._use_bn==True):self._bn=tf.keras.layers.BatchNormalization(momentum=0.997,epsilon=1e-4,name=self._name+"_bn")
        self._act=tf.keras.layers.Activation(self._activation,name=self._name+"_act")
    @tf.Module.with_name_scope
    def __call__(self,input_ts):
        x=self._spbconv(input_ts)
        if(self._use_bn==True):x=self._bn(x)
        x=self._act(x)
        output_ts=x
        return output_ts

class Resize(tf.Module):
    def __init__(self,output_hw,name="resize"):
        super(Resize,self).__init__(name=name)
        self._output_hw=output_hw
        self._name=name
    @tf.Module.with_name_scope
    def __call__(self,input_ts):
        output_ts=tf.image.resize(input_ts,self._output_hw,method=tf.image.ResizeMethod.BILINEAR)
        return output_ts

class Identity(tf.Module):
    def __init__(self,name="identity"):
        super(Identity,self).__init__(name=name)
        self._name=name
    @tf.Module.with_name_scope
    def __call__(self,input_ts):
        output_ts=tf.identity(input_ts)
        return output_ts

class ReLUConvBN(tf.Module):
    def __init__(self,
                 filters,
                 kernel_size,
                 strides=(1,1),
                 padding="same",
                 bias=False,
                 use_bn=True,
                 activation=tf.nn.relu,
                 name="convbn"):
        super(ReLUConvBN,self).__init__()
        self._filters=filters
        self._kernel_size=kernel_size
        self._strides=strides
        self._padding=padding
        self._bias=bias
        self._use_bn=use_bn
        self._activation=activation
        self._name=name
        self._Build()
    @tf.Module.with_name_scope
    def _Build(self):
        self._conv=tf.keras.layers.Conv2D(filters=self._filters,
                                          kernel_size=self._kernel_size,
                                          strides=self._strides,
                                          padding=self._padding,
                                          use_bias=self._bias,
                                          name=self._name+"_conv")
        if(self._use_bn==True):self._bn=tf.keras.layers.BatchNormalization(momentum=0.997,epsilon=1e-4,name=self._name+"_bn")
        self._act=tf.keras.layers.Activation(self._activation,name=self._name+"_act")
    @tf.Module.with_name_scope
    def __call__(self,input_ts):
        x=self._act(input_ts)
        x=self._conv(x)
        if(self._use_bn==True):x=self._bn(x)
        output_ts=x
        return output_ts
    
class Zeros(tf.Module):
    def __init__(self,name="zeros"):
        super(Zeros,self).__init__(name=name)
        self._name=name
    @tf.Module.with_name_scope
    def __call__(self,input_ts):
        output_ts=tf.abs(input_ts)-tf.abs(input_ts)
        return output_ts
    
class ExpDWConvs(tf.Module):
    def __init__(self,t,activation=gelu,name="expdwconvs"):
        super(ExpDWConvs,self).__init__(name=name)
        self._t=t
        self._activation=activation
        self._name=name
        self._Build()
    def _Build(self):
        self._skip_expands=[]
        for i in range(self._t):
            dconv=DepthConvBN(kernel_size=(3,3),strides=(1,1),use_bn=True,activation=None,name=self._name+"_skip_expand_"+str(i))
            self._skip_expands.append(dconv)
        self._expand_bn=tf.keras.layers.BatchNormalization(momentum=0.997,epsilon=1e-4,name=self._name+"_expand_bn")
        self._expand_act=tf.keras.layers.Activation(self._activation,name=self._name+"_expand_act")
    @tf.Module.with_name_scope
    def __call__(self,input_ts):
        skip_expand_x=[]
        skip_x=input_ts
        for i in range(self._t):
            skip_x=self._skip_expands[i](skip_x)
            skip_expand_x.append(skip_x)
        x=tf.concat(skip_expand_x,axis=-1)
        x=self._expand_bn(x)
        x=self._expand_act(x)
        output_ts=x
        return output_ts
    
class SEModule(tf.Module):
    def __init__(self,name="sem"):
        super(SEModule,self).__init__(name=name)
        self._name=name
    @tf.Module.with_name_scope
    def _Build(self,input_ch):
        self._gap=tf.keras.layers.GlobalAveragePooling2D(keepdims=True,name=self._name+"_gap")
        self._conv1=ConvBN(input_ch//4,kernel_size=(1,1),use_bn=False,activation=tf.nn.relu,bias=True,name=self._name+"_conv1")
        self._conv2=ConvBN(input_ch,kernel_size=(1,1),use_bn=False,activation=hard_sigmoid,bias=True,name=self._name+"_conv2")
    @tf.Module.with_name_scope
    def __call__(self,input_ts):
        input_ch=input_ts.get_shape().as_list()[3]
        self._Build(input_ch)
        x=self._gap(input_ts)
        x=self._conv1(x)
        x=self._conv2(x)
        output_ts=input_ts*x
        return output_ts

class DepthAttention(tf.Module):
    def __init__(self,name="depthatten"):
        super(DepthAttention,self).__init__(name=name)
        self._name=name
    @tf.Module.with_name_scope
    def _Build(self,input_ch):
        self._gap=tf.keras.layers.GlobalAveragePooling2D(name=self._name+"_gap")
        self._convq=ConvBN(input_ch//4,kernel_size=(1,1),use_bn=False,activation=gelu,name=self._name+"_convq")
        self._convk=ConvBN(input_ch//4,kernel_size=(1,1),use_bn=False,activation=gelu,name=self._name+"_convk")
        self._convv=ConvBN(input_ch//4,kernel_size=(1,1),use_bn=False,activation=gelu,name=self._name+"_convv")
        self._conv_confd=ConvBN(input_ch,kernel_size=(1,1),use_bn=False,activation=tf.nn.sigmoid,name=self._name+"_conv2")
    @tf.Module.with_name_scope
    def __call__(self,input_ts):
        input_ch=input_ts.get_shape().as_list()[3]
        self._Build(input_ch)
        x=self._gap(input_ts)
        x=tf.reshape(x,[-1,1,1,input_ch])
        q=self._convq(x)
        k=self._convk(x)
        v=self._convv(x)
        q=tf.reshape(q,[-1,input_ch//4,1])
        k=tf.reshape(k,[-1,1,input_ch//4])
        v=tf.reshape(v,[-1,input_ch//4,1])
        qk=tf.matmul(q,k)
        qkv=tf.matmul(tf.nn.softmax(qk),v)
        qkv=tf.reshape(qkv,[-1,1,1,input_ch//4])
        x=self._conv_confd(qkv)
        output_ts=input_ts*x
        return output_ts
