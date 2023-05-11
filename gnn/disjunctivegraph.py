import os
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

from jobshopdata import random_data


N_MACHINES = 4
N_JOBS = 3
DUR_LO = 5
DUR_HI = 50

N_SAMPLES = 2000



data = random_data(None, N_JOBS, N_MACHINES, DUR_LO, DUR_HI)
print(data)

print("X")
for key, value in data.x_dict.items():
    print("  key", key, "shape",value.shape, value)
print(".")

print("EDGE_INDEX")
for key, value in data.edge_index_dict.items():
    print("  key", key, "shape", value.shape, value)
print(".")

print("EDGE_ATTR")
for key, value in data.edge_attr_dict.items():
    print("  key", key, "shape", value.shape, value)
print(".")


class MyGNN(torch.nn.Module):
    def __init__(self):
        super().__init__()

        # hyperparameters
        hidden_channels = 8
        out_channels = 1
        num_layers = 4

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

        self.decode1 = MLP(
            in_channels=hidden_channels * len(self.convs) * 3, hidden_channels=4,
            out_channels=5, num_layers=3)

        self.decode2 = MLP(
            in_channels=10, hidden_channels=4,
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
        # prod = torch.cat([edge_feat_first, edge_feat_second], dim=-1)
        
        # prod = edge_feat_first - edge_feat_second
        #     # print("edge first", edge_feat_first.shape)
        #     # print("edge second", edge_feat_second.shape)
        #     # print("dot prod shape", prod.shape)

        e1 = self.decode1(edge_feat_first)
        e2 = self.decode1(edge_feat_second)
        output = self.decode2(torch.cat([e1,e2], dim=-1))
        output = 2.0 * torch.special.expit(output) - 1.0

        # output = self.decode(cat_xs["operation"])
        # print("out shape", output.shape)

        # num_edges, num_output_features = output.shape
        # # print(num_edges)

        # output_pairs = output.view(num_edges//2, 2, num_output_features)
        # softmax = torch.nn.Softmax(dim=1)(output_pairs)
        # softmax = softmax.view(num_edges, num_output_features)

        # print(output_pairs)
        # print(softmax)
        return output


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


filename = f"jobshop-{N_MACHINES}-{N_JOBS}-{DUR_LO}-{DUR_HI}-{N_SAMPLES}.dat"

if not os.path.exists(filename):
    print("Generating data")
    dataset = [random_data(i, N_JOBS, N_MACHINES, DUR_LO, DUR_HI)
                for i in range(N_SAMPLES)]
    torch.save(dataset, filename)
else:
    print("Loading data...")
    dataset = torch.load(filename)


# dataset = [dataset[0]]
print("Training...")

train_loader = torch_geometric.loader.DataLoader(dataset, 25)


def apply(model, batch):
    # Remove ground truth from model input.
    edge_attrs = batch.edge_attr_dict
    del edge_attrs[("operation", "conflictswith", "operation")]
    out = model(batch.x_dict, batch.edge_index_dict, edge_attrs)
    return out


def train(train_loader):
    model.train()
    for batch_idx, batch in enumerate(train_loader):
        optimizer.zero_grad()

        out = apply(model, batch)
        truth = batch.edge_attr_dict["operation", "conflictswith", "operation"]
        #truth = batch.y_dict["operation"]
        loss = torch.nn.functional.mse_loss(out,
                                            truth)
        if batch_idx % 100 == 0:
            print(f"batch{batch_idx} loss={loss}")
        loss.backward()
        optimizer.step()


def test(loader):
    model.eval()

    correct = 0
    count = 0
    for batch in loader:  # Iterate in batches over the training/test dataset.

        out = apply(model, batch)
        #truth = batch.y_dict["operation"]
        truth = batch.edge_attr_dict["operation",
                                     "conflictswith", "operation"].round().int()
        # pred = out.round().int()  # Use the class with highest probability.
        pred = torch.tensor([[1 if x >= 0.0 else -1] for x in out])

        # Check against ground-truth labels.
        correct += int((pred == truth).sum())
        count += pred.shape[0]

    return correct / count  # Derive ratio of correct predictions.


for epoch in range(10000):
    try:
        train(train_loader)
        train_acc = test(train_loader)
        test_acc = 0.0
        print(
            f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')
    except KeyboardInterrupt:
        print("Cancelling...")
        break



# Single point test

data = random_data(None, N_JOBS, N_MACHINES, DUR_LO, DUR_HI)
print(data)

out = model(data.x_dict, data.edge_index_dict, edge_attrs)
print(data.y_dict)
print(out)
print(torch.tensor([[1 if x >= 0.0 else -1] for x in out]))
print("number ", sum([1 if x >= 0.0 else 0 for x in out]), len([x for x in out]))
