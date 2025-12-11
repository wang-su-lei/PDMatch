import torch
import torch.nn.functional as F
from collections import defaultdict
def update_maw(labels, per_c_weights,m=0.9,scale=0.4):
    labels_detach = labels.argmax(dim=1).detach() 
    i_c_counts = torch.bincount(labels_detach, minlength=labels.shape[1]).float()  
    nonzero_mask = i_c_counts != 0
    nonzero_vals = i_c_counts[nonzero_mask]
    mean = nonzero_vals.mean()
    std = nonzero_vals.std(unbiased=False)
    cur_c_weights = torch.ones_like(i_c_counts)  
    z = (mean-i_c_counts[nonzero_mask]) / (std + 1e-8)
    cur_c_weights[nonzero_mask] = 1.0 + z * scale  
    maw=m*per_c_weights+(1-m)*cur_c_weights
    return maw



def update_daw(kmeans_protos,cluster_nums_tensor,actis,labels,probs,major_classes,C,D,scale = 0.4):
    proto_clusters = []
    for c in range(C):
        cnt = cluster_nums_tensor[c].item()
        full_edges = kmeans_protos[c]
        edges = full_edges[:cnt]
        proto_clusters.append(edges)
    protoC_max_size = max(pc.size(0) for pc in proto_clusters)
    padded = torch.zeros(C, protoC_max_size, D, device=actis.device)
    for c, pc in enumerate(proto_clusters):
        padded[c, : pc.size(0)] = pc
    final_km_preds = labels.argmax(dim=1).detach() 
    final_activations = actis.detach() 
    final_edge_probs = probs.detach() 
    dist_w_list = []

    for i in range(final_km_preds.size(0)):
        cls = final_km_preds[i].item()             
        act = final_activations[i]                 
        probs = final_edge_probs[i].clone()        
        protos_c = padded[cls]                 
        mask_valid_c = protos_c.abs().sum(dim=1) != 0
        dists_c = torch.norm(protos_c[mask_valid_c] - act, dim=1)
        distc = dists_c.min()
        probs[cls] = -1.0
        ambi_c = probs.argmax().item()
        protos_ambi = padded[ambi_c]             
        mask_valid_ambi = protos_ambi.abs().sum(dim=1) != 0
        dists_ambi = torch.norm(protos_ambi[mask_valid_ambi] - act, dim=1)
        distmean = dists_ambi.mean()
        dist_w = distc / (distmean + 1e-8)
        dist_w_list.append(dist_w)

    dist_w_raw = torch.stack(dist_w_list).to(actis.device)  
    mean = dist_w_raw .mean()
    std = dist_w_raw .std(unbiased=False)
    dist_w_norm = (dist_w_raw  - mean) / (std + 1e-8)  
    daw_raw = 1.0 + dist_w_norm * scale
    daw= daw_raw.clamp(min=0.1, max=2.0).detach()
    return daw

