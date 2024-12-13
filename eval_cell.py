from nas_prcss import RankingCellPths,CellPth2Cell,FilteringCellPths,SamplingCellPths
import scipy.stats as stats

def KandallTauRank(gt_ranks,proxy_ranks):
    tau,p_value=stats.kendalltau(gt_ranks,proxy_ranks)
    return tau

def SpearmanRank(gt_ranks,proxy_ranks):
    ids_len=len(proxy_ranks)
    ids_diff=0
    for i,proxy_rank in enumerate(proxy_ranks):
        gt_rank=gt_ranks[i]
        id_diff=(proxy_rank-gt_rank)**2
        ids_diff+=id_diff
    psp=1-((6*ids_diff)/(ids_len*(ids_len**2-1)))
    return psp

def CellPths2Psp(cell_pths,gt_key,proxy_key,sorting="largest",ignore=True):
    if(ignore==True):
        cell_pths=FilteringCellPths(cell_pths,proxy_key)

    gt_ranked_cell_pths=RankingCellPths(cell_pths,gt_key)
    proxy_ranked_cell_pths=RankingCellPths(cell_pths,proxy_key,sorting)
    ranking_dict={}
    for pred_rank,ranked_cell_pth in enumerate(proxy_ranked_cell_pths):
        id=CellPth2Cell(ranked_cell_pth)["id"]
        ranking_dict[id]=[pred_rank]
    for gt_rank,ranked_cell_pth in enumerate(gt_ranked_cell_pths):
        id=CellPth2Cell(ranked_cell_pth)["id"]
        ranking_dict[id].append(gt_rank)

    gt_ranking=[]
    pred_ranking=[]
    for id in ranking_dict:
        pred,gt=ranking_dict[id]
        gt_ranking.append(gt)
        pred_ranking.append(pred)
    psp=KandallTauRank(gt_ranking,pred_ranking)
    return psp

def CellPthsAvgCost(cell_pths,cost_key="proxy_train_time",ignore=True):
    if(ignore==True):
        cell_pths=FilteringCellPths(cell_pths,cost_key)
    cost_time=0
    for i,cell_pth in enumerate(cell_pths):
        cost_time+=CellPth2Cell(cell_pth)[cost_key]
    return cost_time/len(cell_pths)

def KLabelsEst2Psp(klabels,est_key,cost_key="proxy_train_time",cell_pth_type="nas201",data_type="cifar100"):
    if(cell_pth_type=="nas201"):
        gt_key="test_accuracy_200"
    else:
        gt_key="test_accuracy_108"
    psps=[]
    costs=[]
    for k_label in klabels:
        if(cell_pth_type=="nas201"):
            cell_pths=SamplingCellPths("data/forUA/nasbench201OneK_"+data_type+"_"+str(k_label))
        else:
            cell_pths=SamplingCellPths("data/forUA/nasbench101OneK_cifar10_"+str(k_label))
        psp=abs(CellPths2Psp(cell_pths,gt_key,est_key))
        cost_time=CellPthsAvgCost(cell_pths,cost_key)
        psps.append(psp)
        costs.append(cost_time)
    return costs,psps



