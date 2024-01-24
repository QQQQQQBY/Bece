import pickle
import numpy as np
from collections import defaultdict
import torch
import torch_geometric
from torch_geometric.utils import remove_self_loops

def sparse_to_adjlist(relation, filename,node_labels,filename_label):
    adj_lists = {}
    label_lists = {}
    for index, node in enumerate(relation[0]):
        if node != relation[1][index]:
            if node not in adj_lists.keys():
                adj_lists[node] = []
            if relation[1][index] not in adj_lists[node]:      # follower 
                adj_lists[node].append(relation[1][index])
                if node_labels[node] == node_labels[relation[1][index]]: # same users
                    if node not in label_lists.keys():
                        label_lists[node] = []
                    label_lists[node].append(0)
                if node_labels[node] != node_labels[relation[1][index]]: # different users
                    if node not in label_lists.keys():
                        label_lists[node] = []
                    label_lists[node].append(1)

            # friends
            if relation[1][index] not in adj_lists.keys():
                adj_lists[relation[1][index]] = []
            if node not in adj_lists[relation[1][index]]:
                adj_lists[relation[1][index]].append(node) # no direction
                if node_labels[node] == node_labels[relation[1][index]]:
                    if relation[1][index] not in label_lists.keys():
                        label_lists[relation[1][index]] = []
                    label_lists[relation[1][index]].append(0)
                if node_labels[node] != node_labels[relation[1][index]]:
                    if relation[1][index] not in label_lists.keys():
                        label_lists[relation[1][index]] = []
                    label_lists[relation[1][index]].append(1)


        if node == relation[1][index]:
            if node not in adj_lists.keys():
                adj_lists[node] = []
            if relation[1][index] not in adj_lists[node]:
                adj_lists[node].append(relation[1][index])
                if node not in label_lists.keys():
                    label_lists[node] = []            
                label_lists[node].append(0)

    with open(filename_label, 'wb') as file:
        pickle.dump(label_lists, file)
    with open(filename, 'wb') as file:
        pickle.dump(adj_lists, file)
    file.close()    

# 加载边
edge_index = torch.load('/Dataset/MGTAB/edge_index_0_1.pt').cpu()
edge_type = torch.load('/Dataset/MGTAB/edge_type_01.pt').cpu()
edge_index,edge_type = remove_self_loops(edge_index,edge_type)
edge_index = edge_index.t() # torch.Size([2, 1700108])

node_labels = torch.load('/Dataset/MGTAB/labels_bot.pt')
edge_label = []
relation0 = []
relation1 = []
edge_index = edge_index.numpy().tolist()
for i in range(edge_type.shape[0]):
    if edge_type[i] == 0:
        relation0.append(edge_index[i])
    if edge_type[i] == 1:
        relation1.append(edge_index[i])
for i in range(node_labels.shape[0]):
    relation0.append([i, i])
    relation1.append([i, i])

relation0 = np.array(relation0).T
relation1 = np.array(relation1).T
# No direction
sparse_to_adjlist(relation0, ' /Dataset/MGTAB/relation0.pickle',node_labels,' /Dataset/MGTAB/relation0_label.pickle')
sparse_to_adjlist(relation1,  ' /Dataset/MGTAB/relation1.pickle',node_labels,' /Dataset/MGTAB/relation1_label.pickle')
