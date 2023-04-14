import dgl.data
import dgl.function
import dgl
import torch.nn.functional as F
import torch.nn as nn
import torch
import os
os.environ['DGLBACKEND'] = 'pytorch'


class MyNN(nn.Module):
    def __init__(self, num_node_features, node_embedding_size, num_edge_features, edge_embedding_sizes):
        super(MyNN, self).__init__()

        if len(num_edge_features) != len(edge_embedding_sizes):
            raise Exception("Edge type mismatch.")

        self.nodes_in = nn.Sequential(
            nn.Linear(num_node_features, node_embedding_size),
            nn.ReLU(),
            nn.Linear(node_embedding_size, node_embedding_size),
            nn.ReLU(),
            nn.Linear(node_embedding_size, node_embedding_size))

        self.edges_in = nn.ModuleList((
            nn.Sequential(
                nn.Linear(size, features),
                nn.ReLU(),
                nn.Linear(features, features),
                nn.ReLU(),
                nn.Linear(features, features),
            ) for size, features in zip(num_edge_features, edge_embedding_sizes)
        ))

        self.conv1 = GraphConv(in_feats, h_feats)
        self.conv2 = GraphConv(h_feats, 1)

    def forward(self, graph):
        with graph.local_scope():

            # Initial node embedding
            hn = dgl.function.u_add_v

        # h = self.conv1(g, in_feat)
        # h = F.relu(h)
        # h = self.conv2(g, h)
        # return h

        # Create the model with given dimensions
model = MyNN(g.ndata["feat"].shape[1], 16)

#
# Railway GNN concept:
#  - input the conflict relaxation where
#   * nodes represent events (entering a block)
#   * edges represent presedences between consecutive events
#   * edges represent all possible conflicts
#  - decode the delay at the final event
#

#
# Simplified experiment 1:
#  a single train,
#     * edge feature: minimum travel time
#     * edge feature: slack at relaxation?
#
