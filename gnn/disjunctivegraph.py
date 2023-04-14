from dataclasses import dataclass
from typing import List, Tuple
import torch
import torch.nn

import torch_geometric.transforms as T
from torch_geometric.datasets import OGB_MAG
from torch_geometric.nn import HeteroConv, GCNConv, SAGEConv, GATConv, Linear, summary
from torch_geometric.nn.conv import GATv2Conv


@dataclass
class Operation:
    machine_id: int
    duration: float


@dataclass
class Job:
    operations: List[Operation]


@dataclass
class JobShopProblem:
    jobs: List[Job]


example_problem = JobShopProblem([
    Job([Operation(machine_id=0, duration=3),
         Operation(machine_id=1, duration=2),
         Operation(machine_id=2, duration=2)]),
    Job([Operation(machine_id=0, duration=2),
         Operation(machine_id=2, duration=1),
         Operation(machine_id=1, duration=4)]),
    Job([Operation(machine_id=1, duration=4),
         Operation(machine_id=2, duration=3)])
])

def convert_jobshopproblem_to_dataset(p :JobShopProblem) -> Data:
    return


dataset = OGB_MAG(root='./data', preprocess='metapath2vec',
                  transform=T.ToUndirected())
data = dataset[0]


data["author", "writes", "paper"].edge_attr = torch.tensor([
    [0.1, 0.2, 0.3] for _ in data["author", "writes", "paper"].edge_index.T
])
data["paper", "rev_writes", "author"].edge_attr = torch.tensor([
    [0.1, 0.2, 0.3, 0.4] for _ in data["paper", "rev_writes", "author"].edge_index.T
])

data["author", "likes", "paper"].edge_index = data["author",
                                                   "writes", "paper"].edge_index
data["author", "likes", "paper"].edge_attr = data["author",
                                                  "writes", "paper"].edge_attr

data["author", "likes", "paper"].edge_attr = torch.tensor([
    [0.1, 0.2, 0.3, 0.4, 0.5] for _ in data["author", "likes", "paper"].edge_index.T
])


print("X")
for key, value in data.x_dict.items():
    print("  key", key, "shape", value.shape)

print("EDGE_INDEX")
for key, value in data.edge_index_dict.items():
    print("  key", key, "shape", value.shape)


print("EDGE_ATTR")
for key, value in data.edge_attr_dict.items():
    print("  key", key, "shape", value.shape)


class HeteroGNN(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, num_layers):
        super().__init__()

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HeteroConv({
                ('author', 'writes', 'paper'): GATv2Conv((-1, -1), hidden_channels, add_self_loops=False, edge_dim=3),
                ('author', 'likes', 'paper'): GATv2Conv((-1, -1), hidden_channels, add_self_loops=False, edge_dim=5),
                ('paper', 'rev_writes', 'author'): GATv2Conv((-1, -1), hidden_channels, add_self_loops=False, edge_dim=4),
            }, aggr='cat')
            self.convs.append(conv)

        self.lin = Linear(hidden_channels, out_channels)

    def forward(self, x_dict, edge_index_dict, edge_attr_dict):
        for conv in self.convs:
            print("author shape before", x_dict["author"].shape)
            print("paper shape before", x_dict["paper"].shape)
            x_dict = conv(x_dict, edge_index_dict, edge_attr_dict)
            x_dict = {key: x.relu() for key, x in x_dict.items()}
            print("author shape after", x_dict["author"].shape)
            print("paper shape after", x_dict["paper"].shape)
        return self.lin(x_dict['author'])


model = HeteroGNN(hidden_channels=7, out_channels=11,
                  num_layers=2)

# print(summary(model))

with torch.no_grad():  # Initialize lazy modules.
    out = model(data.x_dict, data.edge_index_dict, data.edge_attr_dict)

print(summary(model, data.x_dict, data.edge_index_dict,
      data.edge_attr_dict, max_depth=99, leaf_module=None))


def train():
    model.train()
    optimizer.zero_grad()
    out = model(data.x_dict, data.edge_index_dict)
    mask = data['paper'].train_mask
    loss = F.cross_entropy(out['paper'][mask], data['paper'].y[mask])
    loss.backward()
    optimizer.step()
    return float(loss)


train()
