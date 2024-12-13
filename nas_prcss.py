import os
import numpy as np
import random
from json_io import Dict2JSON,JSON2Dict

def ADJMatrix(adj,max_nodes):
    pad_num=max_nodes-len(adj)
    for elemt in adj:
        for i in range(pad_num):elemt.append(0)
    for i in range(pad_num):adj.append([0 for j in range(max_nodes)])
    adj_matrix=np.array(adj)
    return adj_matrix

def OPsMatrix(all_ops,chosen_ops,max_nodes):
    pad_num=max_nodes-len(chosen_ops)
    if(pad_num>0):
        for i in range(pad_num):chosen_ops.append('none')
    op_matrix=[]
    for op in chosen_ops:
        one_hot=[0 for i in range(len(all_ops))]
        one_hot[all_ops.index(op)]=1
        op_matrix.append(one_hot)
    return np.array(op_matrix)

def TransNas201ADJ(adj,max_nodes):
    adj_m=np.zeros([8,8])
    connections=[[0,1],[0,2],[0,4],[1,3],[1,5],[2,3],[3,6],[4,6],[5,6],[6,7]]
    for connection in connections:
        start_idx,end_idx=connection
        adj_m[end_idx][start_idx]=1
    return ADJMatrix(adj_m,max_nodes)

def TransNas201OPs(all_ops,chosen_ops,max_nodes):
    chosen_ops=["input"]+chosen_ops+["output"]
    return OPsMatrix(all_ops,chosen_ops,max_nodes)

def CellPth2Cell(cell_pth,all_ops=[],max_nodes=8,preprcss=False,cell_pth_type="nas201"):
    cell=JSON2Dict(cell_pth)
    if(preprcss==True and cell_pth_type=="nas201"):
        cell["adj_matrix"]=TransNas201ADJ(cell["adj_matrix"],max_nodes)
        cell["operations"]=TransNas201OPs(all_ops,cell["operations"],max_nodes)
    elif(preprcss==True and cell_pth_type=="nas101"):
        cell["adj_matrix"]=ADJMatrix(cell["adj_matrix"],max_nodes)
        cell["operations"]=OPsMatrix(all_ops,cell["operations"],max_nodes)
    elif(preprcss==True and cell_pth_type=="darts"):
        cell["norm_adj_matrix"]=ADJMatrix(cell["norm_adj_matrix"],max_nodes)
        cell["norm_operations"]=OPsMatrix(all_ops,cell["norm_operations"],max_nodes)
        cell["rdce_adj_matrix"]=ADJMatrix(cell["rdce_adj_matrix"],max_nodes)
        cell["rdce_operations"]=OPsMatrix(all_ops,cell["rdce_operations"],max_nodes)
    elif(preprcss==True and cell_pth_type=="mbnas"):
        cell["adj_matrix"]=ADJMatrix(cell["adj_matrix"],max_nodes)
        cell["operations"]=OPsMatrix(all_ops,cell["operations"],max_nodes)
    return cell

def CellPths2Cells(cell_pths,all_ops=[],max_nodes=7,preprcss=False,cell_pth_type="nas201"):
    return list(map(lambda x:CellPth2Cell(x,all_ops,max_nodes,preprcss,cell_pth_type),cell_pths))

def CellPthInit(cell_pth):
    cell=CellPth2Cell(cell_pth)
    cell["pred_accuracy"]=-1
    cell["confidence"]=1
    Dict2JSON(cell,cell_pth)
    return cell_pth

def CellPthsInit(cell_pths):
    return list(map(lambda x:CellPthInit(x),cell_pths))

def CellPthPredicting(cell_pth,predictor,all_ops=[],max_nodes=8,cell_pth_type="nas201"):
    cell=CellPth2Cell(cell_pth,all_ops=all_ops,max_nodes=max_nodes,preprcss=True,cell_pth_type=cell_pth_type)
    preds=predictor.predict_on_batch((np.array([cell["adj_matrix"]]),np.array([cell["operations"]])))
    pred_acc=preds[0][0]
    cell=CellPth2Cell(cell_pth,max_nodes=max_nodes,preprcss=False)
    cell["pred_accuracy"]=float(pred_acc)
    Dict2JSON(cell,cell_pth)
    return

def CellPredicting(cell,predictor):
    preds=predictor.predict_on_batch((np.array([cell["adj_matrix"]]),np.array([cell["operations"]])))
    pred_acc=preds[0][0]
    cell["pred_accuracy"]=float(pred_acc)
    return cell

def CellPthsPredicting(cell_pths,predictor,all_ops=[],max_nodes=8,cell_pth_type="nas201"):
    for cell_pth in cell_pths:
        CellPthPredicting(cell_pth,predictor,all_ops,max_nodes,cell_pth_type)
    return

def CellsPredicting(cells,predictor):
    for cell in cells:
        CellPredicting(cell,predictor)
    return

def RankingCellPths(cell_pths,rank_key="pred_accuracy",sorting="largest",ignore=False):
    reverse=True if sorting=="largest" else False
    cell_pths=cell_pths.copy()
    ranking_cell_pths=[]
    for cell_path in cell_pths:
        cell=CellPth2Cell(cell_path)
        if(ignore==True and rank_key not in cell):continue
        metric=cell[rank_key]
        ranking_cell_pths.append([cell_path,metric])
    ranking_cell_pths=sorted(ranking_cell_pths,key=lambda x:x[1],reverse=reverse)
    ranking_cell_pths=list(map(lambda x:x[0],ranking_cell_pths))
    return ranking_cell_pths

def RankingCells(cells,rank_key="pred_accuracy",sorting="largest"):
    reverse=True if sorting=="largest" else False
    ranking_cells=[]
    for cell in cells:
        accuracy=cell[rank_key]
        ranking_cells.append([cell,accuracy])
    ranking_cells=sorted(ranking_cells,key=lambda x:x[1],reverse=reverse)
    ranking_cells=list(map(lambda x:x[0],ranking_cells))
    return ranking_cells

def FilteringCellPths(cell_pths,key=None):
    if(key==None):return cell_pths
    _cell_pths=[]
    for cell_pth in cell_pths:
        cell=CellPth2Cell(cell_pth)
        if(key in cell):_cell_pths.append(cell_pth)
    return _cell_pths

def SamplingCellPths(cells_dir,k=-1,shuffle=True):
    cell_pths=[]
    all_cells=os.listdir(cells_dir)
    if(shuffle==True):random.shuffle(all_cells)
    if(k==-1):k=len(all_cells)
    act_count=0
    for cell_name in all_cells:
        cell_path=cells_dir+"/"+cell_name
        if(os.path.isfile(cell_path)!=True):continue
        cell_pths.append(cell_path)
        act_count+=1
        if(act_count==k):break
    return cell_pths

def PartialSamplingCellPths(cells_dir,k=-1):
    if(k==-1):return SamplingCellPths(cells_dir,k=-1)
    cell_pths=SamplingCellPths(cells_dir,k=-1)
    cell_pths=RankingCellPths(cell_pths,"flops")
    partial_len=int(len(cell_pths)/k)
    chosen_pths=[]
    start_idx=0
    for i in range(k):
        end_idx=start_idx+partial_len
        if(i==k-1):end_idx=len(cell_pths)-1
        batch_cell_pths=cell_pths[start_idx:end_idx]
        chosen_pths.append(random.choice(batch_cell_pths))
        start_idx=end_idx
    return chosen_pths

def Cells2MinMaxVal(cells,key):
    m_vals=[]
    for cell in cells:
        m_vals.append(cell[key])
    m_vals=np.array(m_vals)
    return min(m_vals),max(m_vals)

def CellPths2MinMaxVal(cell_pths,key):
    cells=CellPths2Cells(cell_pths)
    return Cells2MinMaxVal(cells,key)

def Cells2MeanStdVal(cells,key):
    m_vals=[]
    for cell in cells:
        m_vals.append(cell[key])
    m_vals=np.array(m_vals)
    return np.mean(m_vals),np.std(m_vals)

def CellPths2MeanStdVal(cell_pths,key):
    cells=CellPths2Cells(cell_pths)
    return Cells2MeanStdVal(cells,key)

def ResetDirtyBit(cell_pths,dirty_bit=0):
    for cell_pth in cell_pths:
        cell=CellPth2Cell(cell_pth)
        cell["dirty_bit"]=dirty_bit
        Dict2JSON(cell,cell_pth)
    return cell_pths
def FilteringByDirtyBit(cell_pths,dirty_bit=0):
    _cell_pths=[]
    for cell_pth in cell_pths:
        cell=CellPth2Cell(cell_pth)
        if(cell["dirty_bit"]==dirty_bit):
            _cell_pths.append(cell_pth)
    return _cell_pths