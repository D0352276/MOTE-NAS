from nas_prcss import SamplingCellPths,RankingCellPths,CellPths2Cells,RankingCells
from eval_cell import CellPths2Psp,CellPth2Cell
from json_io import Dict2JSON
import random
import numpy as np


CELLSIDPOOL=[]
def CellDist(cell_1,cell_2):
    adj_dist=np.array(cell_1["adj_matrix"])-np.array(cell_2["adj_matrix"])
    adj_dist=np.sqrt(np.sum(adj_dist**2))
    ops_dist=np.array(cell_1["operations"])-np.array(cell_2["operations"])
    ops_dist=np.sqrt(np.sum(ops_dist**2))
    return (adj_dist+ops_dist)/2

def RemoveMutationCopy(mutations):
    global CELLSIDPOOL
    _mutations=[]
    _repeat_ids=[]
    for mutation in mutations:
        if(mutation["id"] not in CELLSIDPOOL and mutation["id"] not in _repeat_ids):
            _mutations.append(mutation)
            _repeat_ids.append(mutation["id"])
    return _mutations

def CellMutation(base_cells,possible_cells,max_dist=1.0,mutation_k=1):
    mutation_cells=[]
    for cell in base_cells:
        _mutation_cells=[]
        for _cell in possible_cells:
            cell_dist=CellDist(cell,_cell)
            if(cell_dist<=max_dist):
                _mutation_cells.append(_cell)
        mutation_cells+=_mutation_cells
    mutation_cells=RemoveMutationCopy(mutation_cells)
    random.shuffle(mutation_cells)
    return mutation_cells[:mutation_k]

def Cells2CellPths(cells_dir,cells):
    cell_pths=[]
    for cell in cells:
        cell_pth=cells_dir+"/"+str(cell["id"])+".json"
        cell_pths.append(cell_pth)
    return cell_pths



# all_ops=["void","output","input","skip_connect","zeros","none","nor_conv_3x3","nor_conv_1x1","avg_pool_3x3"]
# data_type="cifar10"
# data_dir="data/nasbench201_"+data_type+"-10"

# proxy_key="proxy_unitsest"
# proxy_cost_key="proxy_train_time_50"
# repeat=1

all_ops=["output","input","skip_connect","zeros","none","nor_conv_3x3","nor_conv_1x1","avg_pool_3x3"]
data_type="img16"
data_dir="data/test_"+data_type+"-10"

proxy_key="mote"
proxy_cost_key="proxy_train_time"
repeat=10


topk=10
budgets=max(topk,10)

save_path="nas_results/"+data_type+"_"+str(topk)+".json"

best_accs=[]
costs=[]
for r in range(repeat):
    CELLSIDPOOL=[]
    PROXYIDPOOL=[]
    cells_pool=[]
    best_acc=0
    cost=0
    for i in range(budgets):
        batch_size=i//10*10+10
        if(i==0):
            cell_pths=SamplingCellPths(data_dir,batch_size)
            cells=CellPths2Cells(cell_pths,all_ops=all_ops,max_nodes=8,preprcss=True)
        else:
            #mutation phase
            cell_pths=SamplingCellPths(data_dir,batch_size*10)
            cells=CellPths2Cells(cell_pths,all_ops=all_ops,max_nodes=8,preprcss=True)
            cells_pool=RankingCells(cells_pool,proxy_key)

            top_cells=[]
            topn=max(1,round(len(cells_pool)*0.01))
            for top_cell in cells_pool[:topn]:
                top_cells.append(top_cell)
            cells=CellMutation(top_cells,cells,mutation_k=batch_size*10)

        #filtering by UTRE
        cells=RankingCells(cells,proxy_key)
        promising_cells=cells[:]
        for promising_cell in promising_cells:
            cells_pool.append(promising_cell)
            CELLSIDPOOL.append(promising_cell["id"])

        #calculate cost
        for cell in cells:
            if(cell["id"] not in PROXYIDPOOL):
                cost+=cell[proxy_cost_key]
                PROXYIDPOOL.append(cell["id"])

        print(i,cost/1000,len(cells_pool))
        
    topk_cells=RankingCells(cells_pool,proxy_key)[:topk]

    topk_cells=RankingCells(topk_cells,"test_accuracy_12")
    for cell in topk_cells:
        cost+=cell["train_time_12"]

    best_acc=topk_cells[0]["test_accuracy_200"]
    best_accs.append(best_acc)
    costs.append(cost)
    print(r,best_acc,cost/1000,"\n")




