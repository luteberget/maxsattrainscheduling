from collections import defaultdict
from dataclasses import dataclass
from typing import List
import numpy as np
import torch
from torch_geometric.data import HeteroData
import gurobipy as gp


def permute_rows(x):
    '''
    x is a np array
    '''
    ix_i = np.tile(np.arange(x.shape[0]), (x.shape[1], 1)).T
    ix_j = np.random.sample(x.shape).argsort(axis=1)
    return x[ix_i, ix_j]


def uni_instance_gen(n_j, n_m, low, high):
    times = np.random.randint(low=low, high=high, size=(n_j, n_m))
    machines = np.expand_dims(
        np.arange(1, n_m+1), axis=0).repeat(repeats=n_j, axis=0)
    machines = permute_rows(machines)
    return times, machines


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


ex_jsp_1 = JobShopProblem([
    Job([Operation(machine_id=0, duration=3),
         Operation(machine_id=1, duration=2),
         Operation(machine_id=2, duration=2)]),
    Job([Operation(machine_id=0, duration=2),
         Operation(machine_id=2, duration=1),
         Operation(machine_id=1, duration=4)]),
    Job([Operation(machine_id=1, duration=4),
         Operation(machine_id=2, duration=3)])
])


def convert_jobshopproblem_to_dataset(p: JobShopProblem) -> HeteroData:
    op_ids = {}
    res_use = defaultdict(list)

    for job_idx, job in enumerate(p.jobs):
        for op_idx, op in enumerate(job.operations):
            op_id = len(op_ids)
            op_ids[(job_idx, op_idx)] = op_id
            res_use[op.machine_id].append(op_id)

    nodes = torch.tensor([[op.duration]
                          for job in p.jobs for op in job.operations], dtype=torch.float)

    precedence = torch.tensor(
        [[op_ids[(job_idx, op_idx)], op_ids[(job_idx, op_idx+1)]]
            for job_idx, job in enumerate(p.jobs)
            for op_idx in range(len(job.operations)-1)],
        dtype=torch.long).t().contiguous()

    rev_precedence = torch.tensor(
        [[op_ids[(job_idx, op_idx+1)], op_ids[(job_idx, op_idx)]]
            for job_idx, job in enumerate(p.jobs)
            for op_idx in range(len(job.operations)-1)],
        dtype=torch.long).t().contiguous()

    conflicts = torch.tensor(
        [[uses[i], uses[j]]
            for uses in res_use.values()
            for i in range(len(uses))
            for j in range(len(uses))
            if i != j],
        dtype=torch.long).t().contiguous()

    data = HeteroData()
    data['operation'].x = nodes
    data['operation', 'precedes', 'operation'].edge_index = precedence
    data['operation', 'rev_precedes', 'operation'].edge_index = rev_precedence
    data['operation', 'conflictswith', 'operation'].edge_index = conflicts

    data['operation', 'conflictswith', 'operation'].edge_attr = torch.tensor(
        [[0.0] for _ in range(conflicts.shape[1])])

    # data = torch_geometric.transforms.ToUndirected(merge=False)(data)
    return data


def random_data(n_jobs, n_machines, lo, hi) -> HeteroData:
    times, machines = uni_instance_gen(n_jobs, n_machines, lo, hi)
    assert (times.shape[0] == n_jobs)
    assert (times.shape[1] == n_machines)

    p = JobShopProblem(
        [
            Job(
                [
                    Operation(machines[job_idx, op_idx],
                              float(times[job_idx, op_idx]))
                    for op_idx in range(n_machines)
                ]
            )
            for job_idx in range(n_jobs)
        ]
    )

    return convert_jobshopproblem_to_dataset(p)


@dataclass
class SolveResult:
    optimal: bool
    conflict_vector: torch.Tensor


def solve_jobshop(p: JobShopProblem) -> SolveResult:
    op_ids = {}
    res_use = defaultdict(list)

    for job_idx, job in enumerate(p.jobs):
        for op_idx, op in enumerate(job.operations):
            op_id = len(op_ids)
            op_ids[(job_idx, op_idx)] = op_id
            res_use[op.machine_id].append(op_id)

    m = gp.Model()
    start_times = {(job_idx, op_idx): m.addVar(lb=0.0)
                   for (job_idx, op_idx) in op_ids.keys()}

    # 1. fixed precedences
    for job_idx, job in enumerate(p.jobs):
        for op_idx in range(len(job.operations)-1):
            m.addConstr(
                start_times[(job_idx, op_idx)] + job.operations[op_idx].duration <=
                start_times[(job_idx, op_idx+1)]
            )

    # 2. big-m disjunctive precedences
    conflicts = []
    for uses in res_use.values():
        for i in range(len(uses)):
            for j in range(len(uses)):
            [[uses[i], uses[j]]
            for uses in res_use.values()
            for i in range(len(uses))
            for j in range(len(uses))
            if i != j],

    print("Optimizing...")
    m.optimize()

    print(f"Optimal objective value: {m.objVal}")
    print(f"Solution values: x={x.X}, y={y.X}, z={z.X}")

if __name__ == '__main__':
    pass