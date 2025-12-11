import torch
import torch.nn.functional as F
from collections import defaultdict


def compute_dists(
    unlab_actis,          
    kmeans_protos,
    cluster_nums_tensor,  
    topk,
    C,
    D
):
    ambi_activations_detach = unlab_actis.detach()
    N = ambi_activations_detach.size(0)
    proto_clusters = []
    for c in range(C):
        cnt = cluster_nums_tensor[c].item()
        full_edges = kmeans_protos[c]
        edges = full_edges[:cnt]
        proto_clusters.append(edges)
    protoC_max_size = max(pc.size(0) for pc in proto_clusters)
    padded = torch.zeros(C, protoC_max_size, D).cuda()
    for c, pc in enumerate(proto_clusters):
        padded[c, : pc.size(0)] = pc
    protos = padded.unsqueeze(0)                               
    a = ambi_activations_detach.unsqueeze(1).unsqueeze(1)       
    diff = a - protos                                         
    km_dists = torch.norm(diff, dim=3)                         

    valid_mask = torch.zeros_like(km_dists, dtype=torch.bool) 
    for c in range(C):
        valid_len = cluster_nums_tensor[c].item()
        if valid_len > 0:
            valid_mask[:, c, :valid_len] = True  

    masked_dists = km_dists.clone()
    masked_dists[~valid_mask] = float('inf') 

    dist_flat = masked_dists.view(N, -1)                          
    topk_vals, topk_idx = torch.topk(dist_flat, topk, dim=1, largest=False)
    km_topk_vals = topk_vals                                    
    km_topk_class_idx = topk_idx // protoC_max_size              

    return km_topk_class_idx, km_topk_vals




def PDEV_MPUS_masks(
    unlab_probs,              
    major_classes,            
    km_topk_class_idx,
    km_topk_vals,
    gamma=0.4       
):

    N, C = unlab_probs.size(0), unlab_probs.size(1)
    device = unlab_probs.device
    dists = km_topk_vals  
    mean_d = dists.mean(dim=1, keepdim=True)
    centered = dists - mean_d   
    max_abs = centered.abs().max(dim=1, keepdim=True)[0].clamp_min(1e-6)
    norm = centered / max_abs   
    weights = 1.0 - gamma * norm  
    km_topk_class_idx = km_topk_class_idx.to(device)   
    votes_tensor = torch.zeros(N, C, device=device)
    votes_tensor.scatter_add_(1, km_topk_class_idx, weights)

    _, km_preds = votes_tensor.topk(1, dim=1)
    km_preds = km_preds.flatten()  
    top2_votes, _ = votes_tensor.topk(2, dim=1)
    diff=top2_votes[:, 0] - top2_votes[:, 1]
    diff_thr=diff.mean()
    vote_diff_mask = (top2_votes[:, 0] - top2_votes[:, 1]) >= diff_thr  

    masked_preds = km_preds[vote_diff_mask]                
    masked_topk_classes = km_topk_class_idx[vote_diff_mask]   
    masked_topk_vals = km_topk_vals[vote_diff_mask]           

    class_dist_min = torch.full((masked_preds.size(0),), float('inf'), device=masked_preds.device)
    for i in range(masked_preds.size(0)):
        cls = masked_preds[i]  
        topk_cls = masked_topk_classes[i]  
        topk_val = masked_topk_vals[i]   
        mask = topk_cls == cls
        if mask.any():
            class_dist_min[i] = topk_val[mask].min()  

    final_mask = torch.ones_like(masked_preds, dtype=torch.bool)  
    for cls in major_classes:
        cls_mask = masked_preds == cls
        cls_dists = class_dist_min[cls_mask]
        if cls_dists.numel() == 0:
            continue
        cls_mean = cls_dists.mean() 
        final_mask[cls_mask] = cls_dists >= cls_mean

    final_vote_mask = vote_diff_mask.clone()
    final_vote_mask[vote_diff_mask] = final_mask
    km_labels = F.one_hot(km_preds[final_vote_mask], num_classes=C).int()

    return final_vote_mask, km_labels 

def has_gradient_flow(tensor):
    return tensor.requires_grad and tensor.grad_fn is not None
def has_gradient_flow_list(tensor_list):
    if not isinstance(tensor_list, list):
        return False
    return any(t.requires_grad and t.grad_fn is not None for t in tensor_list)

