
from eval_cell import CellPths2Psp
from nas_prcss import SamplingCellPths,FilteringByDirtyBit
from gen_mote import GetProxyC100TrainData, CellPth2MOTE

train_x,train_y=GetProxyC100TrainData(labels_len=10)
cells_dir="data/nasbench201_img16-10"
cell_type="nas201"

if(cell_type=="nas101"):
    gt_key="test_accuracy_108"
elif(cell_type=="nas201"):
    gt_key="test_accuracy_200"

cell_pths=SamplingCellPths(cells_dir,shuffle=True)
total_cells=len(cell_pths)
act_cells=0

#retrain option
# cell_pths=ResetDirtyBit(cell_pths,0)


cell_pths=FilteringByDirtyBit(cell_pths,1)
act_cells=total_cells-len(cell_pths)

for i,cell_pth in enumerate(cell_pths):
    print(i,cell_pth,"Processed:"+str(act_cells/total_cells)+"%")
    CellPth2MOTE(train_x,train_y,cell_pth,cell_type,10)
    act_cells+=1

psp=CellPths2Psp(cell_pths,gt_key,"mote")
print(psp)

