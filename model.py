import torch
from torch import nn
import numpy as np
import torch.nn.functional as F

def l2_norm(x):
    if len(x.shape):
        x = x.reshape((x.shape[0],-1))
    return F.normalize(x, p=2, dim=1)


def get_distance(x):
    sim = torch.matmul(x, x.t())
    dist = 2 - 2*sim
    dist = dist.sqrt()
    return dist


class MarginNet(nn.Module):
    r"""Embedding network with distance weighted sampling.
    It takes a base CNN and adds an embedding layer and a
    sampling layer at the end.

    Parameters
    ----------
    base_net : Block
        Base network.
    emb_dim : int
        Dimensionality of the embedding.
    batch_k : int
        Number of images per class in a batch. Used in sampling.

    Inputs:
        - **data**: input tensor with shape (batch_size, channels, width, height).
        Here we assume the consecutive batch_k images are of the same class.
        For example, if batch_k = 5, the first 5 images belong to the same class,
        6th-10th images belong to another class, etc.

    Outputs:
        - The output of DistanceWeightedSampling.
    """
    
    def __init__(self, base_net, emb_dim, batch_k, **kwargs):
        super(MarginNet, self).__init__(**kwargs)
        self.base_net = base_net
        try:
            in_dim = base_net.fc.in_features
        except NameError as e:
            if 'in_dim' in kwargs:
                in_dim = kwargs['in_dim']
            else:
                raise Exception("Neither does the base_net hase fc layer nor in_dim is specificed")
        self.dense = nn.Linear(in_dim, emb_dim)
        self.normalize = l2_norm
        self.sampled = DistanceWeightedSampling(batch_k=batch_k)

    def forward(self,x):
        x = self.base_net(x)
        x = self.dense(x)
        x = self.normalize(x)
        x = self.sampled(x)
        return x


class MarginLoss(nn.Module):
    def __init__(self, margin=0.2, nu=0.0, weight=None, batch_axis=0, **kwargs):
        super(MarginLoss, self).__init__()
        self._margin = margin
        self._nu = nu

    def forward(self, anchors, positives, negatives, beta_in, a_indices=None):
        if a_indices is not None:
            beta = beta_in[a_indices]
            beta_reg_loss =  torch.sum(beta)*self._nu
        else:
            beta = beta_in
            beta_reg_loss = 0.0


        d_ap = torch.sqrt(torch.sum(torch.sqrt(positives - anchors), dim=1) +1e-8)
        d_an = torch.sqrt(torch.sum(torch.sqrt(negatives - anchors), dim=1) +1e-8)

        pos_loss = torch.clamp(d_ap - beta + self._margin, max=0.0)
        neg_loss = torch.clamp(beta - d_an + self._margin, max=0.0)

        pair_cnt = torch.sum((pos_loss > 0.0) + (neg_loss > 0.0)).data()

        loss = (torch.sum(pos_loss + neg_loss) + beta_reg_loss) / pair_cnt
        return loss


class   DistanceWeightedSampling(nn.Module):
    '''
    parameters
    ----------
    batch_k: int
        number of images per class

    Inputs:
        data: input tensor with shapeee (batch_size, edbed_dim)
            Here we assume the consecutive batch_k examples are of the same class.
            For example, if batch_k = 5, the first 5 examples belong to the same class,
            6th-10th examples belong to another class, etc.
    Outputs:
        a_indices: indicess of anchors
        x[a_indices]
        x[p_indices]
        x[n_indices]
        xxx

    '''

    def __init__(self, batch_k, cutoff=0.5, nonzero_loss_cutoff=1.4, **kwargs):
        super(DistanceWeightedSampling,self).__init__()
        self.batch_k = batch_k
        self.cutoff = cutoff
        self.nonzero_loss_cutoff = nonzero_loss_cutoff

    def forward(self, x):
        k = self.batch_k
        n, d = x.shape
        distance = get_distance(x)
        distance = distance.clamp(max=self.cutoff)
        log_weights = ((2.0 - float(d)) * distance.log() - (float(d-3)/2)*torch.log(1.0 - 0.25*(distance*distance)))

        weights = torch.exp(log_weights - torch.maxx(log_weights))

        mask = np.ones(weights.shape)
        for i in range(0,n,k):
            mask[i:i+k, i:i+k] = 0
        weights = weights*torch.Tensor(mask)*(distance < self.nonzero_loss_cutoff)
        weights = weights/torch.sum(weights, dim=1, keepdim=True)

        a_indices = []
        p_indices = []
        n_indices = []

        np_weights = weights.cpu().numpy()
        for i in range(n):
            block_idx = i // k

            try:
                n_indices +=  np.random.choice(n, k-1, p=np_weights[i]).tolist()
            except:
                n_indices +=  np.random.choice(n, k-1).tolist()
            for j in range(block_idx * k, (block_idx + 1)*k):
                if j != i:
                    a_indices.append(i)
                    p_indices.append(j)

        return  a_indices, x[a_indices], x[p_indices], x[n_indices], x




