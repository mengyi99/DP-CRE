import torch
import torch.nn as nn
import random
import numpy as np
import random
from sklearn.cluster import KMeans
from data_loader import get_data_loader


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def select_data(config, encoder, sample_set,id2num, select_num=None):
    data_loader = get_data_loader(config,id2num, sample_set, batch_size=128)
    encoder.eval()
    features = []
    for _, (_, _,tokens,_,_,_) in enumerate(data_loader):
        tokens = torch.stack([x.to(config.device) for x in tokens], dim=0)
        with torch.no_grad():
            feature = encoder(tokens)
        features.append(feature)
    features = torch.cat(features, dim=0).cpu()  
    num_clusters = min(config.memory_size, len(sample_set))
    if select_num is not None:
        num_clusters = min(select_num, len(sample_set))

    kmeans = KMeans(n_clusters=num_clusters, random_state=0)
    distances = kmeans.fit_transform(features)
    cluster_id = kmeans.labels_.tolist()
    mem_set = []
    current_feat = []
    for k in range(num_clusters):
        sel_index = np.argmin(distances[:, k])
        instance = sample_set[sel_index]
        count = cluster_id.count(k) / len(cluster_id)
        instance['weight'] = count
        mem_set.append(instance)
        current_feat.append(features[sel_index])     
    current_feat = np.stack(current_feat, axis=0)
    current_feat = torch.from_numpy(current_feat).to(config.device)
    mean = torch.mean(features,dim=0,keepdim=True).to(config.device)
    return sample_set,mem_set, current_feat, mean,features


def get_proto(config, encoder, mem_set,id2num):
    data_loader = get_data_loader(config,id2num, mem_set, batch_size=config.memory_size)
    weight = torch.tensor([[item['weight']] for item in mem_set]).to(config.device)
    encoder.eval()
    with torch.no_grad():
        for _, (_, _,tokens,_,_,_) in enumerate(data_loader):
            tokens = torch.stack([x.to(config.device) for x in tokens], dim=0)
            with torch.no_grad():
                features = encoder(tokens)
    return features,weight
    

class SupConLoss(nn.Module):
    def __init__(self, device, temperature=0.1, contrast_mode='all'):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.device = device

    def forward(self, features, labels=None, mask=None):
        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)
        batch_size = features.shape[0]

        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(self.device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(self.device)
        else:
            mask = mask.float().to(self.device)
        labels = torch.reshape(labels, (-1, 1))
        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(torch.matmul(anchor_feature, contrast_feature.T),self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        
        # mask-out self-contrast cases
        logits_mask = torch.scatter(torch.ones_like(mask),1,torch.arange(batch_size * anchor_count).view(-1, 1).to(self.device),0)
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        

        # compute mean of log-likelihood over positive and loss
        loss = -(mask * log_prob).sum(1) / mask.sum(1)
        loss = loss.view(anchor_count, batch_size).mean(dim=0)
        return loss
    

