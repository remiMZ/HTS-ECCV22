import torch
import torch.nn as nn
import sys 
sys.path.append("..") 
sys.path.append("../..")
sys.path.append("../../..")
from global_utils import get_backbone

## backbone
class Backbone(nn.Module):
    def __init__(self, backbone):
        super(Backbone, self).__init__()

        self.encoder = get_backbone(backbone)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, inputs):
        features = self.encoder(inputs) 
        features = self.avg_pool(features)
        features = features.view(features.shape[0], -1)   
        return features   


class TreeLSTM(nn.Module):
    def __init__(self, in_features):
        super(TreeLSTM, self).__init__()
        self.in_features = in_features
        self.out_features = in_features

        self.W_iou = nn.Linear(self.in_features, 3 * self.out_features)
        self.U_iou = nn.Linear(self.out_features, 3 * self.out_features, bias=False)
        
        self.W_f = nn.Linear(self.in_features, self.out_features)
        self.U_f = nn.Linear(self.out_features, self.out_features, bias=False)

    def forward(self, data, device):
        features = data['features']
        node_order = data['node_order']
        adjacency_list = data['adjacency_list']
        edge_order = data['edge_order']
        
        batch_size = node_order.shape[0]

        h = torch.zeros(batch_size, self.out_features, device=device)
        c = torch.zeros(batch_size, self.out_features, device=device)

        for n in range(node_order.max() + 1):
            self.node_forward(n, h, c, features, node_order, adjacency_list, edge_order)
        return h

    def node_forward(self, iteration, h, c, features, node_order, adjacency_list, edge_order):
        node_mask = node_order == iteration
        edge_mask = edge_order == iteration

        x = features[node_mask, :]

        if iteration == 0:
            iou = self.W_iou(x)
        else:
            adjacency_list = adjacency_list[edge_mask, :]

            parent_indexes = adjacency_list[:, 0]
            child_indexes = adjacency_list[:, 1]

            child_h = h[child_indexes, :]
            child_c = c[child_indexes, :]

            _, child_counts = torch.unique_consecutive(parent_indexes, return_counts=True)
            child_counts = tuple(child_counts)

            parent_children = torch.split(child_h, child_counts)
            parent_list = [item.sum(0) for item in parent_children]

            h_sum = torch.stack(parent_list)
            iou = self.W_iou(x) + self.U_iou(h_sum)

        i, o, u = torch.split(iou, iou.size(1) // 3, dim=1)
        i = torch.sigmoid(i)
        o = torch.sigmoid(o)
        u = torch.tanh(u)

        if iteration == 0:
            c[node_mask, :] = i * u
        else:
            f = self.W_f(features[parent_indexes, :]) + self.U_f(child_h)
            f = torch.sigmoid(f)

            fc = f * child_c

            parent_children = torch.split(fc, child_counts)
            parent_list = [item.sum(0) for item in parent_children]

            c_sum = torch.stack(parent_list)
            c[node_mask, :] = i * u + c_sum

        h[node_mask, :] = o * torch.tanh(c[node_mask])






        





        


        


