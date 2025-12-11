import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
import numpy as np


def update_prototypes_epoch(feats_np, labels_onehot_np,  CLASS_NUM, seed,topk,beta=0.999):
    n_min = int(min(CLASS_NUM))
    effective_nums = []
    cluster_nums = []
    for n_i in CLASS_NUM:
        if n_i == n_min:
            E_i = 1.0
        else:
            E_i = (1 - beta**n_i) / (1 - beta**n_min)
        effective_nums.append(E_i)
        cluster_nums.append(max(topk, int(round(E_i))))
    effective_nums = np.array(effective_nums, dtype=np.float32)
    cluster_nums = np.array(cluster_nums, dtype=np.int64)
    labels = np.argmax(labels_onehot_np, axis=1) 
    kmeans_protos = []
    for c, num_clusters in enumerate(cluster_nums):
        mask = (labels == c)
        feats_c = feats_np[mask]
        kmeans = KMeans(n_clusters=int(num_clusters), n_init=20,  random_state=seed).fit(feats_c)
        assignments = kmeans.labels_
        protos = []
        for k in range(int(num_clusters)):
            member_mask = (assignments == k)
            centroid = feats_c[member_mask].mean(axis=0)
            protos.append(torch.tensor(centroid, dtype=torch.float32).cuda())   
        kmeans_protos.append(torch.stack(protos, dim=0))
    cluster_nums_tensor = torch.tensor(cluster_nums, dtype=torch.long).cuda()
    return  kmeans_protos, cluster_nums_tensor

def update_prototypes_batch(
    kmeans_protos,
    cluster_nums,
    lab_activations,
    lab_labels,
    m = 0.9
) :
    lab_activations_detach=lab_activations.detach()
    lab_labels_detach=lab_labels.detach()
    B, D = lab_activations.shape

    if lab_labels_detach.dim() == 2:
        lab_indices = lab_labels_detach.argmax(dim=1)
    else:
        lab_indices = lab_labels_detach

    if isinstance(cluster_nums, torch.Tensor):
        cluster_nums = cluster_nums.tolist()

    for i in range(B):
        z = lab_activations_detach[i]     
        c = lab_indices[i].item()      
        r = cluster_nums[c]            
        proto_full = kmeans_protos[c]  
        protos_c = proto_full[:r]      

        z_norm = F.normalize(z, dim=0)
        protos_c_norm = F.normalize(protos_c, dim=1)  
        sims = protos_c_norm @ z_norm                 
        idx = sims.argmax().item()                  

        proto_full[idx] = m * proto_full[idx] + (1 - m) * z

    return kmeans_protos