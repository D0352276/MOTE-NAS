import random
import numpy as np
from nas_prcss import CellPthsInit,CellPth2Cell,RankingCellPths
from json_io import Dict2JSON

def SetProxyAccKey(proxy_acc_key):
    global PROXYACCURACYKEY
    PROXYACCURACYKEY=proxy_acc_key
    return PROXYACCURACYKEY

def SetProxyTrainTimeKey(proxy_train_time_key):
    global PROXYTRAINTIMEKEY
    PROXYTRAINTIMEKEY=proxy_train_time_key
    return PROXYTRAINTIMEKEY

def SetGTAccKey(gt_acc_key):
    global GTACCURACYKEY
    GTACCURACYKEY=gt_acc_key
    return GTACCURACYKEY

def SetGTTrainTimeKey(gt_train_time_key):
    global GTTRAINTIMEKEY
    GTTRAINTIMEKEY=gt_train_time_key
    return GTTRAINTIMEKEY

def GetAccKey(pth_type):
    global GTACCURACYKEY
    global PROXYACCURACYKEY
    acc_key=GTACCURACYKEY if pth_type=="gt" else PROXYACCURACYKEY
    return acc_key

def GetTrainTimeKey(pth_type):
    global GTTRAINTIMEKEY
    global PROXYTRAINTIMEKEY
    time_key=GTTRAINTIMEKEY if pth_type=="gt" else PROXYTRAINTIMEKEY
    return time_key



class NAS201CellsPool:
    def __init__(self):
        self._cell_pths_pool=[]
        self._cell_pths_info={}
        self._log=[]
        self._cur_best_acc=0
        self._cur_best_cell={"test_accuracy_200":0}
        self._test_accs=[]
    def CheckPth(self,cell_pth):
        if(cell_pth not in self._cell_pths_pool):
            return True
        else:
            return False
    def CheckPths(self,cell_pths):
        checked_pths=[]
        for cell_pth in cell_pths:
            if(self.CheckPth(cell_pth)!=False):
                checked_pths.append(cell_pth)
        return checked_pths
    def AppendPth(self,cell_pth,pth_type="gt"):
        if(cell_pth not in self._cell_pths_pool):
            self._cell_pths_pool.append(cell_pth)
            cell=CellPth2Cell(cell_pth)
            self._cell_pths_info[cell_pth]=[pth_type,cell]
            # self._log.append([len(self._cell_pths_pool),self.CostTime(),self._cur_best_acc])
            self._log.append([len(self._cell_pths_pool),self.CostTime(),self._cur_best_cell["test_accuracy_200"]])

            return True
        return False
    def AppendPths(self,cell_pths,pth_type="gt"):
        act_cell_pths=[]
        for cell_pth in cell_pths:
            if(self.AppendPth(cell_pth,pth_type)):
                act_cell_pths.append(cell_pth)
        return act_cell_pths
    def UpdateCellPth(self,cell_pth,pth_type="gt"):
        assert cell_pth in self._cell_pths_pool,"This 'cell_pth' not in pool."
        cell=CellPth2Cell(cell_pth)
        self._cell_pths_info[cell_pth]=[pth_type,cell]
        return
    def Get(self,k=-1):
        cell_pths=self._cell_pths_pool.copy()
        if(k==-1):
            return cell_pths
        else:
            return random.choices(cell_pths,k=k)
    def Len(self):
        return len(self._cell_pths_pool)
    def CostTime(self):
        cost_time=0
        for cell_pth in self._cell_pths_pool:
            pth_type,cell=self._cell_pths_info[cell_pth]
            cost_time+=cell[GetTrainTimeKey(pth_type)]
        return cost_time
    def Log(self):
        return self._log
    def UpdateBestGTAcc(self):
        for cell_pth in self._cell_pths_pool:
            pth_type,cell=self._cell_pths_info[cell_pth]
            self.UpdateCellPth(cell_pth,"gt")
            acc=cell[GetAccKey("gt")]
            if(acc>self._cur_best_acc):
                self._cur_best_acc=acc
                self._cur_best_cell=cell
        del self._log[-1]
        # self._log.append([len(self._cell_pths_pool),self.CostTime(),self._cur_best_acc])
        self._log.append([len(self._cell_pths_pool),self.CostTime(),self._cur_best_cell["test_accuracy_200"]])

        return self._cur_best_acc
    def UpdateBestPorxyAcc(self):
        cell_pths=self._cell_pths_pool.copy()
        cell_pths=RankingCellPths(cell_pths,GetAccKey("proxy"))
        
        for top_cell_pth in cell_pths:
            pth_type,cell=self._cell_pths_info[top_cell_pth]
            if(pth_type=="gt"):
                continue
            else:
                self.UpdateCellPth(top_cell_pth,"gt")
                acc=cell[GetAccKey("gt")]
                if(acc>self._cur_best_acc):
                    self._cur_best_acc=acc
                break
        del self._log[-1]
        self._log.append([len(self._cell_pths_pool),self.CostTime(),self._cur_best_acc])
        return self._cur_best_acc
    def UpdateBestAcc(self,pth_type="gt"):
        if(pth_type=="gt"):
            self._cur_best_acc=self.UpdateBestGTAcc()
        else:
            self._cur_best_acc=self.UpdateBestPorxyAcc()
        return self._cur_best_acc
    def GetCurBestAcc(self):
        return self._cur_best_acc


def SetGlobalCellsPool():
    global GLOBALCELLSPOOL
    GLOBALCELLSPOOL=NAS201CellsPool()
    return GLOBALCELLSPOOL

def GetGlobalCellsPool():
    global GLOBALCELLSPOOL
    if(GLOBALCELLSPOOL==None):
        raise Exception("GetGlobalCellsPool Error: Please 'SetGlobalCellsPool' first.")
    return GLOBALCELLSPOOL