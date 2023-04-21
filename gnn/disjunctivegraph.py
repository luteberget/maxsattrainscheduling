from collections import defaultdict
from dataclasses import dataclass
from typing import List, Tuple
import torch
import torch.nn
import torch.functional as F

import torch_geometric
import torch_geometric.loader
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, summary, MLP
from torch_geometric.nn.conv import GATv2Conv



data = random_data(5, 2, 9, 25)
print(data)

print("X")
for key, value in data.x_dict.items():
    print("  key", key, "shape", value)
print(".")

print("EDGE_INDEX")
for key, value in data.edge_index_dict.items():
    print("  key", key, "shape", value)
print(".")

print("EDGE_ATTR")
for key, value in data.edge_attr_dict.items():
    print("  key", key, "shape", value)
print(".")


class MyGNN(torch.nn.Module):
    def __init__(self):
        super().__init__()

        # hyperparameters
        hidden_channels = 4
        out_channels = 1
        num_layers = 3

        self.encode = MLP(
            in_channels=-1, hidden_channels=4,
            out_channels=hidden_channels*3, num_layers=3)

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HeteroConv({
                ('operation', 'precedes', 'operation'): GATv2Conv((-1, -1), hidden_channels, add_self_loops=False),
                ('operation', 'rev_precedes', 'operation'): GATv2Conv((-1, -1), hidden_channels, add_self_loops=False),
                ('operation', 'conflictswith', 'operation'): GATv2Conv((-1, -1), hidden_channels, add_self_loops=False),
            }, aggr='cat')
            self.convs.append(conv)

        self.decode = MLP(
            in_channels=hidden_channels * len(self.convs) * num_layers, hidden_channels=4,
            out_channels=out_channels, num_layers=3)

    def forward(self, x_dict, edge_index_dict, edge_attr_dict):
        x_dict["operation"] = self.encode(x_dict["operation"])

        # layer_sums = x_dict["operation"]
        layer_xdicts = defaultdict(list)
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict, edge_attr_dict)
            x_dict = {key: x.relu() for key, x in x_dict.items()}
            # print(x_dict["operation"].shape)
            for key, x in x_dict.items():
                layer_xdicts[key].append(x)
            # layer_sums += x_dict["operation"]

        # "skip connections"?
        cat_xs = {key: torch.cat(x, dim=-1) for key, x in layer_xdicts.items()}

        edge_index = edge_index_dict["operation", "conflictswith", "operation"]
        edge_feat_first = cat_xs['operation'][edge_index[0]]
        edge_feat_second = cat_xs['operation'][edge_index[1]]
        prod = (edge_feat_first * edge_feat_second)

        output = self.decode(prod)
        # print(output)

        num_edges, num_output_features = output.shape
        # print(num_edges)

        output_pairs = output.view(num_edges//2, 2, num_output_features)
        softmax = torch.nn.Softmax(dim=1)(output_pairs)
        softmax = softmax.view(num_edges, num_output_features)

        # print(output_pairs)
        # print(softmax)
        return softmax


model = MyGNN()

with torch.no_grad():  # Initialize lazy modules.
    data_loader = torch_geometric.loader.DataLoader([data], 1)
    print(data_loader)
    for x in data_loader:
        edge_attrs = x.edge_attr_dict
        del edge_attrs[("operation", "conflictswith", "operation")]
        out = model(x.x_dict, x.edge_index_dict, edge_attrs)

# print(summary(model, data.x_dict, data.edge_index_dict,
#       data.edge_attr_dict, max_depth=99, leaf_module=None))

optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
dataset = [data] * 100
train_loader = torch_geometric.loader.DataLoader(dataset, 50)


def apply(model, batch):
    # Remove ground truth from model input.
    edge_attrs = batch.edge_attr_dict
    del edge_attrs[("operation", "conflictswith", "operation")]
    out = model(batch.x_dict, batch.edge_index_dict, edge_attrs)
    return out


def train(train_loader):
    model.train()
    for batch in train_loader:
        optimizer.zero_grad()

        out = apply(model, batch)
        truth = batch.edge_attr_dict["operation", "conflictswith", "operation"]
        loss = torch.nn.functional.mse_loss(out,
                                            truth)
        # print(loss)
        loss.backward()
        optimizer.step()


def test(loader):
    model.eval()

    correct = 0
    count = 0
    for batch in loader:  # Iterate in batches over the training/test dataset.

        out = apply(model, batch)
        truth = batch.edge_attr_dict["operation",
                                     "conflictswith", "operation"].round().int()
        pred = out.round().int()  # Use the class with highest probability.

        # Check against ground-truth labels.
        correct += int((pred == truth).sum())
        count += pred.shape[0]

    return correct / count  # Derive ratio of correct predictions.


for epoch in range(200):
    train(train_loader)
    train_acc = test(train_loader)
    test_acc = 0.0
    print(
        f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')
