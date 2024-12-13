import numpy as np
import random
from nas_prcss import CellPth2Cell,SamplingCellPths,PartialSamplingCellPths
from cells_pool import GetGlobalCellsPool,GetAccKey
from nas_augment import RandMutation

class NasDataGenerator:
    def __init__(self,data_dir,all_ops,max_nodes=8,init_cells=10,read_type="gt",init_method="grids"):
        self._data_dir=data_dir
        self._all_ops=all_ops
        self._max_nodes=max_nodes
        self._read_type=read_type
        self._cells_pool=GetGlobalCellsPool()
        if(init_method=="grids"):
            self._cells_pool.AppendPths(PartialSamplingCellPths(data_dir,k=init_cells),self._read_type)
        elif(init_method=="rands"):
            self._cells_pool.AppendPths(SamplingCellPths(data_dir,k=init_cells),self._read_type)
        self._cells_pool.UpdateBestAcc(self._read_type)
    def Read(self,batch_size=16):
        act_cell_pths=self._cells_pool.Get(k=batch_size)
        adj_matrix_list=[]
        op_matrix_list=[]
        acc_cnfd_list=[]
        for i,cell_path in enumerate(act_cell_pths):
            cell_dict=CellPth2Cell(cell_path,self._all_ops,self._max_nodes,preprcss=True)
            ######
            gt_acc=cell_dict[GetAccKey(self._read_type)]
            # gt_acc=cell_dict[GetAccKey(self._read_type)]/10
            ######
            adj_mat=cell_dict["adj_matrix"]
            ops_mat=cell_dict["operations"]
            cnfd=cell_dict.get("confidence",1.0)
            adj_mat,ops_mat,cnfd=RandMutation(adj_mat,ops_mat,cnfd)
            adj_matrix_list.append(adj_mat)
            op_matrix_list.append(ops_mat)
            acc_cnfd_list.append([gt_acc,cnfd])
        output_xy=(np.array(adj_matrix_list),np.array(op_matrix_list)),np.array(acc_cnfd_list)
        return output_xy
    def Gen(self,batch_size=16):
        while(1):
            yield self.Read(batch_size)

class FixedNasDataGenerator:
    def __init__(self,data_dir,all_ops,max_nodes=8,cell_pths=[],gt_key="test_accuracy_200"):
        self._data_dir=data_dir
        self._all_ops=all_ops
        self._max_nodes=max_nodes
        self._cell_pths=cell_pths
        self._gt_key=gt_key
    def Read(self,batch_size=16):
        random.shuffle(self._cell_pths)
        act_cell_pths=self._cell_pths[:batch_size]
        adj_matrix_list=[]
        op_matrix_list=[]
        acc_cnfd_list=[]
        for i,cell_path in enumerate(act_cell_pths):
            cell_dict=CellPth2Cell(cell_path,self._all_ops,self._max_nodes,preprcss=True)
            gt_acc=cell_dict[self._gt_key]
            adj_mat=cell_dict["adj_matrix"]
            ops_mat=cell_dict["operations"]
            cnfd=cell_dict.get("confidence",1.0)
            adj_matrix_list.append(adj_mat)
            op_matrix_list.append(ops_mat)
            acc_cnfd_list.append([gt_acc,cnfd])
        output_xy=(np.array(adj_matrix_list),np.array(op_matrix_list)),np.array(acc_cnfd_list)
        return output_xy
    def Gen(self,batch_size=16):
        while(1):
            yield self.Read(batch_size)