import tensorflow as tf
from modules import ConvBN,DepthConvBN,gelu
from mobile_cell import MobileCell

class CellBN(tf.Module):
    def __init__(self,filters,get_cell_function=None,activation=None,use_bn=False,name="cellbn"):
        super(CellBN,self).__init__(name=name)
        self._filters=filters
        self._get_cell_function=get_cell_function
        self._activation=activation
        self._use_bn=use_bn
        self._name=name
        self._Build()
    @tf.Module.with_name_scope
    def _Build(self):
        if(self._get_cell_function!=None):
            self._cell=self._get_cell_function(self._filters,name=self._name+"_cell")
        self._bn=tf.keras.layers.BatchNormalization(momentum=0.997,epsilon=1e-4,name=self._name+"_bn")
        self._act=tf.keras.layers.Activation(self._activation,name=self._name+"_act")
    @tf.Module.with_name_scope
    def __call__(self,input_ts):
        x=input_ts
        if(self._get_cell_function!=None):
            x=self._cell(x)
        if(self._use_bn==True):
            x=self._bn(x)
        output_ts=self._act(x)
        return output_ts

class TinyMetaModel(tf.Module):
    def __init__(self,get_cell_function=None,labels_len=10,name="tinymetamode"):
        super(TinyMetaModel,self).__init__(name=name)
        self._get_cell_function=get_cell_function
        self._labels_len=labels_len
        self._name=name
        self._Build()
    @tf.Module.with_name_scope
    def _Build(self):
        self._stem=ConvBN(32,(3,3),(1,1),activation=None,name=self._name+"_stem")
        self._cell=CellBN(32,self._get_cell_function,name=self._name+"_cell")
        self._pool=tf.keras.layers.AveragePooling2D((4,4),name=self._name+"_pool")
        self._cell_2=CellBN(32,self._get_cell_function,name=self._name+"_cell_2")
        self._gap=tf.keras.layers.GlobalAveragePooling2D(name=self._name+"_gap")
        self._dsout=tf.keras.layers.Dense(self._labels_len,activation=None,name=self._name+"_dsout")
    @tf.Module.with_name_scope
    def __call__(self,input_ts):
        x=self._stem(input_ts)
        x=self._cell(x)
        x=self._pool(x)
        x=self._cell_2(x)
        x=self._gap(x)
        proxy_ts=self._dsout(x)
        return proxy_ts
    
class TinyDARTSMetaModel(tf.Module):
    def __init__(self,normal_cell_function=None,reduce_cell_function=None,labels_len=10,name="tinydartsmetamode"):
        super(TinyDARTSMetaModel,self).__init__(name=name)
        self._normal_cell_function=normal_cell_function
        self._reduce_cell_function=reduce_cell_function
        self._labels_len=labels_len
        self._name=name
        self._Build()
    @tf.Module.with_name_scope
    def _Build(self):
        self._stem=ConvBN(32,(3,3),(1,1),activation=None,name=self._name+"_stem")
        self._norm_cell=CellBN(32,self._normal_cell_function,name=self._name+"_cell")
        self._pool=tf.keras.layers.AveragePooling2D((4,4),name=self._name+"_pool")
        self._rdce_cell=CellBN(32,self._reduce_cell_function,name=self._name+"_rdce_cell")
        self._gap=tf.keras.layers.GlobalAveragePooling2D(name=self._name+"_gap")
        self._dsout=tf.keras.layers.Dense(self._labels_len,activation=None,name=self._name+"_dsout")
    @tf.Module.with_name_scope
    def __call__(self,input_ts):
        x=self._stem(input_ts)
        x=self._norm_cell([x,x])
        x=self._pool(x)
        x=self._rdce_cell([x,x])
        x=self._gap(x)
        proxy_ts=self._dsout(x)
        return proxy_ts
    
class ResBlock(tf.Module):
    def __init__(self,filters,name="resblock"):
        super(ResBlock,self).__init__(name=name)
        self._filters=filters
        self._name=name
        self._Build()
    @tf.Module.with_name_scope
    def _Build(self):
        self._conv_1=ConvBN(self._filters,(3,3),(2,2),activation=tf.nn.relu,name=self._name+"_conv_1")
        self._conv_2=ConvBN(self._filters,(3,3),(1,1),activation=tf.nn.relu,name=self._name+"_conv_2")
        self._pooling=tf.keras.layers.AveragePooling2D((2,2),name=self._name+"_pooling")
        self._conv_3=ConvBN(self._filters,(1,1),(1,1),use_bn=False,activation=None,name=self._name+"_conv_3")
    @tf.Module.with_name_scope
    def __call__(self,input_ts):
        x=self._conv_1(input_ts)
        x=self._conv_2(x)
        shortcut=self._pooling(input_ts)
        shortcut=self._conv_3(shortcut)
        output_ts=x+shortcut
        return output_ts
    
class Nas201Block(tf.Module):
    def __init__(self,filters,get_cell_function=None,blck_len=1,name="nas201block"):
        super(Nas201Block,self).__init__(name=name)
        self._filters=filters
        self._get_cell_function=get_cell_function
        self._blck_len=blck_len
        self._name=name
        self._Build()
    @tf.Module.with_name_scope
    def _Build(self):
        self._cell_list=[]
        self._first_cell=CellBN(self._filters,self._get_cell_function,name=self._name+"_first_cell")
        for i in range(self._blck_len-1):
            self._cell_list.append(CellBN(self._filters,self._get_cell_function,name=self._name+"_cell_"+str(i)))
    @tf.Module.with_name_scope
    def __call__(self,input_ts):
        first_x=self._first_cell(input_ts)
        x=first_x
        for i in range(self._blck_len-1):
            x=self._cell_list[i](x)
        output_ts=x
        return output_ts
    
class NAS201Model(tf.Module):
    def __init__(self,get_cell_function=None,labels_len=10,name="nas201model"):
        super(NAS201Model,self).__init__(name=name)
        self._init_channel=16
        self._get_cell_function=get_cell_function
        self._labels_len=labels_len
        self._name=name
        self._Build()
    @tf.Module.with_name_scope
    def _Build(self):
        self._convbn=ConvBN(self._init_channel,(3,3),(1,1),activation=None,name=self._name+"_convbn")
        self._cells_1=Nas201Block(self._init_channel,self._get_cell_function,5,name=self._name+"_cells_1")
        self._resblck_1=ResBlock(int(self._init_channel*2),name=self._name+"_resblck_1")
        self._cells_2=Nas201Block(int(self._init_channel*2),self._get_cell_function,5,name=self._name+"_cells_2")
        self._resblck_2=ResBlock(int(self._init_channel*4),name=self._name+"_resblck_2")
        self._cells_3=Nas201Block(int(self._init_channel*4),self._get_cell_function,5,name=self._name+"_cells_3")
        self._gap=tf.keras.layers.GlobalAveragePooling2D(name=self._name+"_gap")
        self._dsout=tf.keras.layers.Dense(self._labels_len,activation=None,name=self._name+"_dsout")
    @tf.Module.with_name_scope
    def __call__(self,input_ts):
        x=self._convbn(input_ts)
        x=self._cells_1(x)
        x=self._resblck_1(x)
        x=self._cells_2(x)
        x=self._resblck_2(x)
        x=self._cells_3(x)
        x=self._gap(x)
        out_ts=self._dsout(x)
        return out_ts

class NAS101Model(tf.Module):
    def __init__(self,get_cell_function=None,labels_len=10,name="nas101model"):
        super(NAS101Model,self).__init__(name=name)
        self._get_cell_function=get_cell_function
        self._labels_len=labels_len
        self._name=name
        self._Build()
    @tf.Module.with_name_scope
    def _Build(self):
        self._stem=ConvBN(128,(3,3),(1,1),activation=None,name=self._name+"_convbn")
        self._cell_1=CellBN(128,self._get_cell_function,name=self._name+"_cell_1")
        self._cell_2=CellBN(128,self._get_cell_function,name=self._name+"_cell_2")
        self._cell_3=CellBN(128,self._get_cell_function,name=self._name+"_cell_3")
        self._pool_1=tf.keras.layers.AveragePooling2D((2,2),name=self._name+"_pool_1")
        self._cell_4=CellBN(128,self._get_cell_function,name=self._name+"_cell_4")
        self._cell_5=CellBN(128,self._get_cell_function,name=self._name+"_cell_5")
        self._cell_6=CellBN(128,self._get_cell_function,name=self._name+"_cell_6")
        self._pool_2=tf.keras.layers.AveragePooling2D((2,2),name=self._name+"_pool_2")
        self._cell_7=CellBN(128,self._get_cell_function,name=self._name+"_cell_7")
        self._cell_8=CellBN(128,self._get_cell_function,name=self._name+"_cell_8")
        self._cell_9=CellBN(128,self._get_cell_function,name=self._name+"_cell_9")
        self._gap=tf.keras.layers.GlobalAveragePooling2D(name=self._name+"_gap")
        self._dsout=tf.keras.layers.Dense(self._labels_len,activation=None,name=self._name+"_dsout")
    @tf.Module.with_name_scope
    def __call__(self,input_ts):
        x=self._stem(input_ts)
        x=self._cell_1(x)
        x=self._cell_2(x)
        x=self._cell_3(x)
        x=self._pool_1(x)
        x=self._cell_4(x)
        x=self._cell_5(x)
        x=self._cell_6(x)
        x=self._pool_2(x)
        x=self._cell_7(x)
        x=self._cell_8(x)
        x=self._cell_9(x)
        x=self._gap(x)
        out_ts=self._dsout(x)
        return out_ts

class MobileMetaModel(tf.Module):
    def __init__(self,ops_name_list,alpha=1.0,labels_len=10,name="mobilemetamodel"):
        super(MobileMetaModel,self).__init__(name=name)
        self._ops_name_list=ops_name_list
        self._alpha=alpha
        self._labels_len=labels_len
        self._name=name
        self._Build()
    @tf.Module.with_name_scope
    def _Build(self):
        self._stem=ConvBN(int(8*self._alpha),(3,3),(1,1),activation=None,name=self._name+"_stem")
        self._cell1=MobileCell(ops_name=self._ops_name_list[0],filters=int(16*self._alpha),t=2,strides=(2,2),name=self._name+"_cell1")
        self._cell2=MobileCell(ops_name=self._ops_name_list[1],filters=int(16*self._alpha),t=2,name=self._name+"_cell2")
        self._cell3=MobileCell(ops_name=self._ops_name_list[2],filters=int(32*self._alpha),t=2,strides=(2,2),name=self._name+"_cell3")
        self._cell4=MobileCell(ops_name=self._ops_name_list[3],filters=int(32*self._alpha),t=2,name=self._name+"_cell4")
        self._cell5=MobileCell(ops_name=self._ops_name_list[4],filters=int(32*self._alpha),t=2,name=self._name+"_cell5")
        self._gap=tf.keras.layers.GlobalAveragePooling2D(name=self._name+"_gap")
        self._dsout=tf.keras.layers.Dense(self._labels_len,activation=None,name=self._name+"_dsout")
    @tf.Module.with_name_scope
    def __call__(self,input_ts):
        x=self._stem(input_ts)
        x=self._cell1(x)
        x=self._cell2(x)
        x=self._cell3(x)
        x=self._cell4(x)
        x=self._cell5(x)
        x=self._gap(x)
        output_ts=self._dsout(x)
        return output_ts
    

class CELoss(tf.Module):
    def __init__(self,name="celoss"):
        super(CELoss,self).__init__(name=name)
        self._name=name
    @tf.Module.with_name_scope
    def __call__(self):
        def _CELoss(true_y,pred_y):
            pred_y=tf.nn.softmax(pred_y)
            labels_len=tf.shape(pred_y)[-1]
            label_idx=tf.cast(true_y[...,0],tf.int32)
            one_hot_label=tf.one_hot(label_idx,labels_len)
            ce_loss=tf.keras.losses.categorical_crossentropy(one_hot_label,pred_y,label_smoothing=0.01)
            return ce_loss
        return _CELoss

def CreateTinyMetaModel(get_cell_function=None,labels_len=10):
    input_shape=(32,32,3)
    input_ts=tf.keras.Input(shape=input_shape)
    proxy_ts=TinyMetaModel(get_cell_function,labels_len)(input_ts)
    proxy_model=tf.keras.Model(inputs=input_ts,outputs=proxy_ts)
    return proxy_model


def CreateTinyDARTSMetaModel(norm_cell_function=None,reduce_cell_function=None,labels_len=10):
    input_shape=(32,32,3)
    input_ts=tf.keras.Input(shape=input_shape)
    proxy_ts=TinyDARTSMetaModel(norm_cell_function,reduce_cell_function,labels_len)(input_ts)
    proxy_model=tf.keras.Model(inputs=input_ts,outputs=proxy_ts)
    return proxy_model


def CreateNAS101Model(get_cell_function=None,labels_len=10):
    input_shape=(32,32,3)
    input_ts=tf.keras.Input(shape=input_shape)
    out_ts=NAS101Model(get_cell_function,labels_len)(input_ts)
    model=tf.keras.Model(inputs=input_ts,outputs=out_ts)
    return model

def CreateNAS201Model(get_cell_function=None,labels_len=10):
    input_shape=(32,32,3)
    input_ts=tf.keras.Input(shape=input_shape)
    out_ts=NAS201Model(get_cell_function,labels_len)(input_ts)
    model=tf.keras.Model(inputs=input_ts,outputs=out_ts)
    return model

def CreateMobileMetaModel(operations_list,alpha=1.0,labels_len=10):
    input_shape=(32,32,3)
    input_ts=tf.keras.Input(shape=input_shape)
    out_ts_list=MobileMetaModel(operations_list,alpha,labels_len)(input_ts)
    model=tf.keras.Model(inputs=input_ts,outputs=out_ts_list)
    return model

def CompileModel(model,lr=0.01):
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),loss=CELoss()())
    return model