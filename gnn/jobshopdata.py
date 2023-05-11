from collections import defaultdict
from dataclasses import dataclass
from typing import List
import numpy as np
import torch
from torch_geometric.data import HeteroData
import gurobipy as gp

M = 5000.0


def permute_rows(x):
    '''
    x is a np array
    '''
    ix_i = np.tile(np.arange(x.shape[0]), (x.shape[1], 1)).T
    ix_j = np.random.sample(x.shape).argsort(axis=1)
    return x[ix_i, ix_j]


def uni_instance_gen(n_j, n_m, low, high):
    times = np.random.randint(low=low, high=high, size=(n_j, n_m))
    print("TIMES", times)
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

    def conflicts(self):
        cs=[]
        for job_idx1, job1 in enumerate(self.jobs):
            for op_idx1, op1 in enumerate(job1.operations):
                t1 = (job_idx1, op_idx1)
                for job_idx2, job2 in enumerate(self.jobs):
                    for op_idx2, op2 in enumerate(job2.operations):
                        t2 = (job_idx2, op_idx2)
                        if t1 == t2:
                            continue
                        cs.append((t1,t2))
        return cs




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


def convert_jobshopproblem_to_dataset(p: JobShopProblem, solution: List[int]) -> HeteroData:
    op_ids = {}
    res_use = defaultdict(list)

    for job_idx, job in enumerate(p.jobs):
        for op_idx, op in enumerate(job.operations):
            op_id = len(op_ids)
            op_ids[(job_idx, op_idx)] = op_id
            res_use[op.machine_id].append(op_id)

    nodes = torch.tensor([[op.duration]
                          for job in p.jobs for op in job.operations], dtype=torch.float)
    
    sequence = torch.tensor([[op_idx]
                          for job in p.jobs for op_idx,op in enumerate(job.operations)], dtype=torch.float)

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
        [[op_ids[a], op_ids[b]]
            for a,b in p.conflicts()],
        dtype=torch.long).t().contiguous()

    data = HeteroData()
    data['operation'].x = nodes
    data['operation'].y = sequence
    data['operation', 'precedes', 'operation'].edge_index = precedence
    data['operation', 'rev_precedes', 'operation'].edge_index = rev_precedence
    data['operation', 'conflictswith', 'operation'].edge_index = conflicts

    assert (conflicts.shape[1] == len(solution))
    data['operation', 'conflictswith', 'operation'].edge_attr = torch.tensor([[float(x)] for x in solution])

    # data = torch_geometric.transforms.ToUndirected(merge=False)(data)
    return data


def random_data(idx, n_jobs, n_machines, lo, hi) -> HeteroData:
    print("Generating data ", idx)
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

    solution = solve_jobshop(p)

    return convert_jobshopproblem_to_dataset(p, solution)


@dataclass
class SolveResult:
    optimal: bool
    conflict_vector: torch.Tensor


def solve_jobshop(p: JobShopProblem) -> SolveResult:
    op_ids = {}
    res_use = defaultdict(set)

    for job_idx, job in enumerate(p.jobs):
        for op_idx, op in enumerate(job.operations):
            op_id = len(op_ids)
            op_ids[(job_idx, op_idx)] = op_id
            res_use[op.machine_id].add((job_idx, op_idx))

    m = gp.Model()
    m.setParam('OutputFlag', 0)
    start_times = {(job_idx, op_idx): m.addVar(lb=0.0)
                    for job_idx, job in enumerate(p.jobs)
                    for op_idx, _op in enumerate(job.operations)}


    # 1. fixed precedences
    for job_idx, job in enumerate(p.jobs):
        for op_idx in range(len(job.operations)-1):
            m.addConstr(
                start_times[(job_idx, op_idx)] + job.operations[op_idx].duration <=
                start_times[(job_idx, op_idx+1)]
            )

    # 2. big-m disjunctive precedences
    conflicts = p.conflicts()
    conflict_vars = {}

    for c in conflicts:
        t1,t2 = c
        j1,op1 = t1
        j2,op2 = t2
        
        if c not in conflict_vars:
            var = m.addVar(vtype=gp.GRB.BINARY)
            conflict_vars[(t1,t2)] = gp.LinExpr(var)
            conflict_vars[(t2,t1)] = 1.0 - gp.LinExpr(var)

        m.addConstr(
            start_times[(
                j1, op1)] + p.jobs[j1].operations[op1].duration <= start_times[(j2, op2)] + M*conflict_vars[c]
        )

    print("Optimizing...")
    m.setObjective(sum(start_times.values()))
    m.optimize()
    print(f"DONE {m.Runtime*1000.0:.1f} ms")
    if m.status == gp.GRB.OPTIMAL:
        print(f"Optimal objective value: {m.objVal}")
        return [1 if conflict_vars[c].getValue() >= 0.5 else -1 for c in conflicts]
    elif m.status == gp.GRB.INFEASIBLE:
        m.write("kjeks.lp")
        m.computeIIS()
        m.write("infeasible.ilp")
        raise Exception("Infeasible")
    else:
        raise Exception("Infeasible")


if __name__ == '__main__':
    pass
