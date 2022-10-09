import torch
import numpy as np

def calculate_evaluation_orders(adjacency_list, tree_size):
    adjacency_list = np.array(adjacency_list)

    node_ids = np.arange(tree_size, dtype=int)

    node_order = np.zeros(tree_size, dtype=int)
    unevaluated_nodes = np.ones(tree_size, dtype=bool)

    parent_nodes = adjacency_list[:, 0]
    child_nodes = adjacency_list[:, 1]

    n = 0
    while unevaluated_nodes.any():
        unevaluated_mask = unevaluated_nodes[child_nodes]

        unready_parents = parent_nodes[unevaluated_mask]

        nodes_to_evaluate = unevaluated_nodes & ~np.isin(node_ids, unready_parents)
        node_order[nodes_to_evaluate] = n
        unevaluated_nodes[nodes_to_evaluate] = False

        n += 1

    edge_order = node_order[parent_nodes]

    return node_order, edge_order

def batch_tree_input(batch):
    tree_sizes = [b['features'].shape[0] for b in batch]

    batched_features = torch.cat([b['features'] for b in batch])
    batched_node_order = torch.cat([b['node_order'] for b in batch])
    batched_edge_order = torch.cat([b['edge_order'] for b in batch])

    batched_adjacency_list = []
    offset = 0
    for n, b in zip(tree_sizes, batch):
        batched_adjacency_list.append(b['adjacency_list'] + offset)
        offset += n
    batched_adjacency_list = torch.cat(batched_adjacency_list)

    return {
        'features': batched_features,
        'node_order': batched_node_order,
        'edge_order': batched_edge_order,
        'adjacency_list': batched_adjacency_list,
        'tree_sizes': tree_sizes
    }

def unbatch_tree_tensor(tensor, tree_sizes):
    return torch.split(tensor, tree_sizes, dim=0)

def data_list(idx, features, num_parent=None):
    state = {}
    state['id'] = idx
    state['features'] = features
    state['num_parent'] = num_parent
    return state

class BTree(object):
    def __init__(self, tree):
        self.tree = tree

    def convert_tensors_to_tree(self, parent_id=None):
        tree_data = []
        if not len(self.tree):
            return []
        for item in self.tree:
            if parent_id:
                if item['id'] == parent_id:
                    children = self.search_children_by_pid(item['id'],[])
                    item['children'] = children
                    break
            elif item['num_parent'] == None:
                children = self.search_children_by_pid(item['id'],[])          
                item['children'] = children
                tree_data.append(item)

        return tree_data       

    def search_children_by_pid(self, pid, id_results:list):
        if not len(self.tree):
            return []
        children = []
        for item in self.tree:
            if item['id'] not in id_results:
                if item['num_parent'] == pid:
                    id_results.append(item['id'])
                    item['children'] = self.search_children_by_pid(item['id'],id_results)
                    children.append(item)

        return children
    
def get_tree(ssl_ways, shape, features):
    data = features.split(1)
    tree = []    
    for i in range(shape[0]):
        state = data_list(str(i), data[i])
        tree.append(state)       
        for j in range(shape[0]+(i*ssl_ways[0]), shape[0]+((i+1)*ssl_ways[0])):
            state = data_list(str(j), data[j], num_parent=str(i))
            tree.append(state)
            if len(shape) == 2:
                continue
            for k in range(shape[0]+shape[1]+(i*ssl_ways[1]), shape[0]+shape[1]+((i+1)*ssl_ways[1])):
                state = data_list(str(k), data[k], num_parent=str(j))
                tree.append(state)
                if len(shape) == 3:
                    continue
                
    tree = BTree(tree)
    tree = tree.convert_tensors_to_tree()
    
    return tree

def _label_node_index(node, n=0):
    node['index'] = n
    for child in node['children']:
        n +=1
        n = _label_node_index(child, n)
    return n
        
def _gather_node_attributes(node, key):
    features = [node[key]]
    for child in node['children']:
        features.extend(_gather_node_attributes(child, key))
    return features

def _gather_adjacency_list(node):
    adjacency_list = []
    for child in node['children']:
        adjacency_list.append([node['index'], child['index']])
        adjacency_list.extend(_gather_adjacency_list(child))

    return adjacency_list

def convert_tree_to_tensors(trees, device):
    data_all = []
    for _, tree in enumerate(trees):
        _label_node_index(tree)
        features = _gather_node_attributes(tree, 'features')
        features = torch.cat(features, 0)
        adjacency_list = _gather_adjacency_list(tree)
        
        node_order, edge_order = calculate_evaluation_orders(adjacency_list, len(features))

        data = {
            'features': features, 
            'adjacency_list': torch.tensor(adjacency_list, device=device, dtype=torch.int64),
            'node_order': torch.tensor(node_order, device=device, dtype=torch.int64),
            'edge_order': torch.tensor(edge_order, device=device, dtype=torch.int64),
        }
        data_all.append(data)
    
    batch_data = batch_tree_input(data_all)
        
    return batch_data