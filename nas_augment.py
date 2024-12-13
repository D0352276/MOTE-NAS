import random
import numpy as np

def Identity(adj_mat,op_mat,cnfd):
    return adj_mat,op_mat,cnfd

def AddConnection(adj_mat,op_mat,cnfd):
    nodes_len,_=np.shape(adj_mat)[:2]
    while(1):
        idx_1=random.randint(0,nodes_len-1)
        idx_2=random.randint(0,nodes_len-1)
        if(adj_mat[idx_1][idx_2]==0):
            break
    adj_mat[idx_1][idx_2]=1
    return adj_mat,op_mat,cnfd*0.5

def ChangeOperation(adj_mat,op_mat,cnfd):
    nodes_len,ops_len=np.shape(op_mat)[:2]
    node_idx=random.randint(0,nodes_len-1)
    op_idx=random.randint(0,ops_len-1)
    op_mat[node_idx]=np.where(op_mat[node_idx]==1,0,0)
    op_mat[node_idx][op_idx]=1
    return adj_mat,op_mat,cnfd*0.5

def RandMutation(adj_mat,op_mat,cnfd):
    if(random.random()>0.5):
        adj_mat,op_mat,cnfd=AddConnection(adj_mat,op_mat,cnfd)
    if(random.random()>0.5):
        adj_mat,op_mat,cnfd=ChangeOperation(adj_mat,op_mat,cnfd)
    return adj_mat,op_mat,cnfd
