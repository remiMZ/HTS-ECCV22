import torch
import torch.nn as nn
import torch.nn.functional as F

import sys 
sys.path.append("..") 
sys.path.append("../..")

from global_utils import get_backbone

## Backbone
class Backbone(nn.Module):
    def __init__(self, backbone):
        super(Backbone, self).__init__()

        self.encoder = get_backbone(backbone)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, inputs):
        # features = self.encoder(inputs.view(-1, *inputs.shape[2:])) 
        features = self.encoder(inputs)
        features = self.avg_pool(features)
        features = features.view(features.shape[0], -1)   
        return features   
    
## TreeLSTM
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

## GCN
def gmul(inputs, matrix):
    num_samples = inputs.shape[1]
    matrix = torch.split(matrix, 1, 3)
    matrix = torch.cat(matrix, 1).squeeze(3)    
    outputs = torch.bmm(matrix, inputs)    
    outputs = torch.split(outputs, num_samples, 1)
    outputs = torch.cat(outputs, 2)    
    return outputs


class GConv(nn.Module):
    def __init__(self, in_features, out_features, J=2, has_bn=True):
        super(GConv, self).__init__()
        self.J = J
        self.in_features = in_features * J
        self.out_features = out_features
        self.fc = nn.Linear(self.in_features, self.out_features)

        self.has_bn = has_bn
        if self.has_bn:
            self.bn = nn.BatchNorm1d(self.out_features)

    def forward(self, inputs, matrix):
        x = gmul(inputs, matrix)    
        x_shape = x.shape
        x = x.contiguous()
        x = x.view(-1, self.in_features)
        x = self.fc(x)    

        if self.has_bn:
            x = self.bn(x)

        x = x.view(*x_shape[:-1], self.out_features)   
        return x

class MatrixGenerator(nn.Module):
    def __init__(self, in_features, hid_features, ratio=[2, 2, 1, 1], operator='J2', activation='softmax'):
        super(MatrixGenerator, self).__init__()

        self.operator = operator
        self.activation = activation

        self.conv2d_1 = nn.Conv2d(in_features, int(hid_features * ratio[0]), kernel_size=1, stride=1)
        self.bn_1 = nn.BatchNorm2d(int(hid_features * ratio[0]))
        self.conv2d_2 = nn.Conv2d(int(hid_features * ratio[0]), int(hid_features * ratio[1]), kernel_size=1, stride=1)
        self.bn_2 = nn.BatchNorm2d(int(hid_features * ratio[1]))
        self.conv2d_3 = nn.Conv2d(int(hid_features * ratio[1]), int(hid_features * ratio[2]), kernel_size=1, stride=1)
        self.bn_3 = nn.BatchNorm2d(int(hid_features * ratio[2]))
        self.conv2d_4 = nn.Conv2d(int(hid_features * ratio[2]), int(hid_features * ratio[3]), kernel_size=1, stride=1)
        self.bn_4 = nn.BatchNorm2d(int(hid_features * ratio[3]))

        self.conv2d_last = nn.Conv2d(int(hid_features * ratio[3]), 1, kernel_size=1, stride=1)

    def forward(self, inputs, matrix_id):
        inputs_i = inputs.unsqueeze(2)
        inputs_j = torch.transpose(inputs_i, 1, 2)
        matrix_new = torch.abs(inputs_i - inputs_j)    
        matrix_new = torch.transpose(matrix_new, 1, 3)    
        matrix_new = self.conv2d_1(matrix_new)
        matrix_new = self.bn_1(matrix_new)
        matrix_new = F.leaky_relu(matrix_new)

        matrix_new = self.conv2d_2(matrix_new)
        matrix_new = self.bn_2(matrix_new)
        matrix_new = F.leaky_relu(matrix_new)

        matrix_new = self.conv2d_3(matrix_new)
        matrix_new = self.bn_3(matrix_new)
        matrix_new = F.leaky_relu(matrix_new)

        matrix_new = self.conv2d_4(matrix_new)
        matrix_new = self.bn_4(matrix_new)
        matrix_new = F.leaky_relu(matrix_new)

        matrix_new = self.conv2d_last(matrix_new)
        matrix_new = torch.transpose(matrix_new, 1, 3)     
        
        if self.activation == 'softmax':
            matrix_new = matrix_new - matrix_id.expand_as(matrix_new) * 1e8   
            matrix_new = torch.transpose(matrix_new, 2, 3)

            matrix_new = matrix_new.contiguous()
            matrix_new_shape = matrix_new.shape
            matrix_new = matrix_new.view(-1, matrix_new.size(3))
            matrix_new = F.softmax(matrix_new, dim=-1)
            matrix_new = matrix_new.view(*matrix_new_shape)

            matrix_new = torch.transpose(matrix_new, 2, 3)
        elif self.activation == 'sigmoid':    
            matrix_new = torch.sigmoid(matrix_new)
            matrix_new *= (1 - matrix_id)
        elif self.activation == 'none':    
            matrix_new *= (1 - matrix_id)
        else:
            raise ValueError('Non-supported Activation.')

        if self.operator == 'laplace':    
            matrix_new = matrix_id - matrix_new
        elif self.operator == 'J2':
            matrix_new = torch.cat((matrix_id, matrix_new), -1)
        else:
            raise ValueError('Non-supported Operator.')

        return matrix_new

class GraphNeuralNetwork(nn.Module):
    def __init__(self, out_channels, num_ways, hid_features=96, J=2):
        super(GraphNeuralNetwork, self).__init__()
        self.out_channels = out_channels
        self.num_ways = num_ways
        self.in_features = self.out_channels + num_ways
        self.J = J

        self.num_layers = 2
        for i_layer in range(self.num_layers):
            module_matrix = MatrixGenerator(
                in_features=self.in_features + int(hid_features / 2) * i_layer, 
                hid_features=hid_features)
            module_gconv = GConv(
                in_features=self.in_features + int(hid_features / 2) * i_layer, 
                out_features=int(hid_features / 2), 
                J=2)
            self.add_module('module_matrix_{}'.format(i_layer), module_matrix)
            self.add_module('module_gconv_{}'.format(i_layer), module_gconv)

        self.module_matrix_last = MatrixGenerator(
                in_features=self.in_features + int(hid_features / 2) * self.num_layers, 
                hid_features=hid_features)
        self.module_gconv_last = GConv(
                in_features=self.in_features + int(hid_features / 2) * self.num_layers, 
                out_features=self.num_ways, 
                J=2,
                has_bn=False)

    def forward(self, support_embeddings, support_one_hot_targets, query_embeddings):
        support_node_embeddings = torch.cat((support_embeddings, support_one_hot_targets), -1)    

        predictions = None
        for i_query in range(query_embeddings.shape[1]):
            single_query_embedding = query_embeddings[:, i_query].unsqueeze(1)
            pseudo_query_one_hot_target = torch.ones(*single_query_embedding.shape[:2], self.num_ways, device=support_one_hot_targets.device) / self.num_ways
            query_node_embedding = torch.cat((single_query_embedding, pseudo_query_one_hot_target), -1)
            node_embeddings = torch.cat((query_node_embedding, support_node_embeddings), 1)   

            matrix_init = torch.eye(node_embeddings.shape[1], device=support_embeddings.device).unsqueeze(0).repeat(node_embeddings.shape[0], 1, 1).unsqueeze(3)    

            for i_layer in range(self.num_layers):
                matrix_i = self._modules['module_matrix_{}'.format(i_layer)](node_embeddings, matrix_init)

                node_embeddings_new = F.leaky_relu(self._modules['module_gconv_{}'.format(i_layer)](node_embeddings, matrix_i))
                node_embeddings = torch.cat((node_embeddings, node_embeddings_new), -1)

            matrix_last = self.module_matrix_last(node_embeddings, matrix_init)
            prediction = self.module_gconv_last(node_embeddings, matrix_last)[:, 0, :].unsqueeze(1)  
            if predictions is None:
                predictions = prediction
            else:
                predictions = torch.cat((predictions, prediction), 1)
        del support_embeddings, query_embeddings, support_node_embeddings
        return predictions    




        





        


        


