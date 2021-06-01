

import math
import torch
import torch.nn as nn
import itertools
import torch.nn.functional as F
from tools import euclidean_dist, cosine_dist

def get_all_feature_list(local_feature_list, global_feature):

    local_feature_list.append(global_feature)
    feature_list = local_feature_list
    return feature_list

def get_features(local_feature_list, global_feature):
    local_feature_list.append(global_feature)
    cated_features = [feature.unsqueeze(1) for feature in local_feature_list]
    cated_features = torch.cat(cated_features, dim=1)
    return cated_features

class GraphConvolutional(nn.Module):

    def __init__(self, in_dim, out_dim, bias=True):
        super(GraphConvolutional, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.weight = nn.Parameter(torch.FloatTensor(self.in_dim, self.out_dim))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_dim))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def learn_adj(self, inputs, adj):
        similarities = torch.zeros([inputs.size(0), 7, 7])
        for k in range(inputs.size(0)):
            for i, j in itertools.product(list(range(7)), list(range(7))):
                if i < j:
                    if adj[i, j] > 0:
                        #similarity = euclidean_dist(inputs[k, i, :].unsqueeze(0), inputs[k, j, :].unsqueeze(0))
                        similarity = torch.exp(-torch.norm((inputs[k, i, :].unsqueeze(0) -
                                                            inputs[k, j, :].unsqueeze(0)), p=2) / 2)
                        #similarity = cosine_dist(inputs[k, i, :].unsqueeze(0), inputs[k, j, :].unsqueeze(0))

                        similarities[k, i, j] = similarity
                elif i == j:
                    similarities[k, i, j] = 1

        similarities = F.normalize(similarities, p=1)
        for k in range(inputs.size(0)):
            for i, j in itertools.product(list(range(7)), list(range(7))):
                if i == j:
                    similarities[k, i, j] = 1
        mask = adj.unsqueeze(0).repeat([inputs.size(0), 1, 1])
        new_adj = similarities * mask
        return new_adj

    def forward(self, x, adj):
        feature = x.squeeze()
        new_adj = self.learn_adj(feature, adj).cuda()
        support = torch.matmul(feature, self.weight)
        output = torch.matmul(new_adj, support)
        if self.bias is not None:
            output = output + self.bias
        else:
            output = output

        return output

    def __repr__(self):
        return self.__class__.__name__+'(' \
            + str(self.in_dim) + '->' \
            + str(self.out_dim) + ')'

class GraphConvNet(nn.Module):
    def __init__(self, nfeat, nhid, adj):
        super(GraphConvNet, self).__init__()

        self.adj = adj
        self.gc = GraphConvolutional(nfeat, nhid)

    def forward(self, local_feature_list, global_feature):

        features = get_features(local_feature_list, global_feature)
        output = self.gc(features, self.adj)
        new_local_feature, new_global_feature = torch.split(output, split_size_or_sections=[6, 1], dim=1)

        return new_global_feature