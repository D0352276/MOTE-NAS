import random
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import seaborn as sns
from scipy.interpolate import make_interp_spline
import numpy as np
import os
from json_io import JSON2Dict,Dict2JSON
from nas_prcss import RankingCellPths,SamplingCellPths
from brokenaxes import brokenaxes

class PltFig:
    def __init__(self,x_label,y_label,title=None,xlims=None,ylims=None,grid=True,x_hid=False,y_hid=False,figsize=(8,6),space=0.05):
        plt.figure(figsize=figsize, dpi=80)
        self._ax=brokenaxes(xlims=xlims,ylims=ylims,
                            wspace=space,hspace=space)
        # self._ax=plt.gca()
        # self._ax.spines["right"].set_visible(False)
        # self._ax.spines["top"].set_visible(False)

        if(grid==True):self._ax.grid()
        if(x_hid==True):self._ax.set_xticks([])
        if(y_hid==True):self._ax.set_yticks([])
        self._ax.set_xlabel(x_label,fontsize=12)
        self._ax.set_ylabel(y_label,fontsize=12)
        self._ax.set_title(title)
        
        plt.setp(self._ax.get_yticklabels(),rotation=50,ha="right",rotation_mode="anchor")
    def _CurveSmooth(self,x_arr,y_arr):
        x_arr=x_arr.copy()
        y_arr=y_arr.copy()
        orig_len=len(x_arr)
        model=make_interp_spline(x_arr,y_arr)
        _x_arr=np.linspace(x_arr[0],x_arr[-1],int(orig_len*0.5))
        y_arr=model(_x_arr)
        model=make_interp_spline(_x_arr,y_arr)
        _x_arr=np.linspace(_x_arr[0],_x_arr[-1],orig_len)
        y_arr=model(_x_arr)
        return x_arr,y_arr
    def PlotCurve(self,x_arr,y_arr,alpha=1,linewidth=2,label="",linestyle="-",marker="",markersize=8,color="gray",smooth=False):
        if(smooth==True):
            x_arr,y_arr=self._CurveSmooth(x_arr,y_arr)
        self._ax.plot(x_arr,y_arr,marker=marker,markersize=markersize,linestyle=linestyle,linewidth=linewidth,label=label,color=color,alpha=alpha)
        return
    def PlotCrtrnLine(self,x_val,y_lim,alpha=0.5,linewidth=2,color="gray"):
        self._ax.plot([x_val,x_val],y_lim,marker="",linestyle="--",linewidth=linewidth,color=color,alpha=alpha)
        return
    def PlotSymbol(self,x_arr,y_arr,marker="^",markersize=10,label="",alpha=1.0,color="gray"):
        self._ax.plot(x_arr,y_arr,marker=marker,markersize=markersize,linestyle="None",label=label,color=color,alpha=alpha)
        return
    def PlotDensity(self,x_arr,y_arr,alpha=0.25):
        sns.kdeplot(x=x_arr,y=y_arr,cmap="Greys",fill=True,cbar=False,alpha=alpha)
        return
    def PlotStd(self,x_arr,y_arr,std_arr,std_scale=0.1,alpha=0.125,color="gray",smooth=False):
        if(smooth==True):
            x_arr,y_arr=self._CurveSmooth(x_arr,y_arr)
        std_arr=std_arr*std_scale
        self._ax.fill_between(x_arr,y_arr+std_arr,y_arr-std_arr,color=color,alpha=alpha)
        return
    def Save(self,save_path,legend_loc="upper left",fontsize=10):
        if(legend_loc!="None"):
            self._ax.legend(fontsize=fontsize,loc=legend_loc)
        # plt.tight_layout()
        plt.savefig(save_path,bbox_inches="tight")
        return

def SpearmanRank(gt_ranks,proxy_ranks):
    ids_len=len(proxy_ranks)
    ids_diff=0
    for i,proxy_rank in enumerate(proxy_ranks):
        gt_rank=gt_ranks[i]
        id_diff=(proxy_rank-gt_rank)**2
        ids_diff+=id_diff
    psp=1-((6*ids_diff)/(ids_len*(ids_len**2-1)))
    return psp

def RankXY(data_dir,proxy_test_acc_key="proxy_test_accuracy",test_acc_key="test_accuracy_200"):
    id_accs=[]
    id_proxy_accs=[]
    id_rank={}
    proxy_id_rank={}

    alljs=os.listdir(data_dir)
    random.shuffle(alljs)

    for js_name in alljs:
        js_path=data_dir+"/"+js_name
        cell=JSON2Dict(js_path)
        if(proxy_test_acc_key not in cell):continue

        id_accs.append([cell["id"],cell[test_acc_key]])
        id_proxy_accs.append([cell["id"],cell[proxy_test_acc_key]])

    id_accs=sorted(id_accs,key=lambda x:x[1],reverse=True)
    for i,id_acc in enumerate(id_accs):
        id,acc=id_acc
        id_rank[id]=i
    id_proxy_accs=sorted(id_proxy_accs,key=lambda x:x[1],reverse=True)
    for i,id_proxy_acc in enumerate(id_proxy_accs):
        id,acc=id_proxy_acc
        proxy_id_rank[id]=i
    x=[]
    y=[]
    for id in id_rank:
        x.append(id_rank[id])
        y.append(proxy_id_rank[id])
    return x,y,SpearmanRank(x,y)

def AccXY(data_dir):
    x=[]
    y=[]
    alljs=os.listdir(data_dir)
    for js_name in alljs:
        js_path=data_dir+"/"+js_name
        cell=JSON2Dict(js_path)
        if("proxy_test_accuracy" not in cell):continue
        x.append(cell["test_accuracy_200"])
        # y.append(cell["test_accuracy_12"])
        y.append(cell["proxy_test_accuracy"])
    print(len(x))
    return x,y

def Filtering2NASData(data_dir,out_dir):
    alljs=os.listdir(data_dir)
    for js_name in alljs:
        js_path=data_dir+"/"+js_name
        out_path=out_dir+"/"+js_name
        cell=JSON2Dict(js_path)
        if("proxy_test_accuracy" not in cell):continue
        Dict2JSON(cell,out_path)