use grb::parameter::IntParam::NetworkCuts;

use super::SolverError;
use crate::problem::{DelayCostType, Problem};
use std::collections::VecDeque;

pub fn spawn_heuristic_thread(
    mk_env: impl Fn() -> grb::prelude::Env + Send + 'static,
    sol_in_rx: std::sync::mpsc::Receiver<Vec<Vec<i32>>>,
    problem: Problem,
    delay_cost_type: DelayCostType,
    sol_out_tx: std::sync::mpsc::Sender<(i32, Vec<Vec<i32>>)>,
) {
    std::thread::spawn(move || {
        let env = mk_env();
        while let Ok(mut sol) = sol_in_rx.recv() {
            loop {
                while let Ok(more_recent_sol) = sol_in_rx.try_recv() {
                    sol = more_recent_sol;
                }
                // let ub_sol =
                //     crate::solvers::heuristic::solve_heuristic(&env, &problem, &sol).unwrap();
                let ub_sol = crate::solvers::heuristic::solve_heuristic_better(
                    &env,
                    &problem,
                    delay_cost_type,
                    false,
                    Some(&sol),
                )
                .unwrap();
                if let Some(ub_sol) = ub_sol {
                    let ub_cost = problem.verify_solution(&ub_sol, delay_cost_type).unwrap();
                    if sol_out_tx.send((ub_cost, ub_sol)).is_ok() {
                        println!("HEUR.FEAS. {}", ub_cost);
                    }
                } else {
                }

                match sol_in_rx.try_recv() {
                    Ok(more_recent_sol) => {
                        sol = more_recent_sol;
                        continue;
                    }
                    Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                        break;
                    }
                    _ => {}
                }

                let ub_sol = crate::solvers::heuristic::solve_heuristic_better(
                    &env,
                    &problem,
                    delay_cost_type,
                    true,
                    Some(&sol),
                )
                .unwrap();
                if let Some(ub_sol) = ub_sol {
                    let ub_cost = problem.verify_solution(&ub_sol, delay_cost_type).unwrap();
                    if sol_out_tx.send((ub_cost, ub_sol)).is_ok() {
                        println!("HEUR.FEAS. {}", ub_cost);
                    }
                }

                break;
            }
        }
    });
}

pub fn solve_heuristic_better(
    env: &grb::Env,
    problem: &Problem,
    delay_cost_type: crate::problem::DelayCostType,
    use_strong_branching: bool,
    solution: Option<&Vec<Vec<i32>>>,
) -> Result<Option<Vec<Vec<i32>>>, SolverError> {
    let mut train_start_idx: Vec<u32> = Default::default();
    let mut t_vars: Vec<i32> = Default::default();
    let mut total_cost = 0;
    let tiebreak_obj = crate::problem::DelayCostType::Continuous;

    let _p = hprof::enter("solve_heuristic_better");
    let _p0 = hprof::enter("solve_heuristic_better");
    if let Some(sol) = solution {
        for (train_idx, vs) in sol.iter().enumerate() {
            train_start_idx.push(t_vars.len() as u32);
            assert!(vs.len() == problem.trains[train_idx].visits.len() + 1);
            t_vars.extend(vs.iter().copied());
        }
    } else {
        for train in problem.trains.iter() {
            train_start_idx.push(t_vars.len() as u32);
            t_vars.extend(train.visits.iter().map(|v| v.earliest));
            t_vars.push(t_vars.last().unwrap() + train.visits.last().unwrap().travel_time);
        }
    }

    for train_idx in 0..problem.trains.len() {
        for visit_idx in 0..problem.trains[train_idx].visits.len() {
            // Lower bounds should already have been taken into account.
            let t = t_vars[train_start_idx[train_idx] as usize + visit_idx];
            assert!(problem.trains[train_idx].visits[visit_idx].earliest <= t);
            total_cost +=
                problem.trains[train_idx].visit_delay_cost(delay_cost_type, visit_idx, t) as i32;
        }
    }

    // Check travel time constraints and add resource usage.
    for (train_idx, train) in problem.trains.iter().enumerate() {
        for visit_idx in 0..train.visits.len() {
            let visit = train.visits[visit_idx];
            let t1 = t_vars[train_start_idx[train_idx] as usize + visit_idx];
            let t2 = &mut t_vars[train_start_idx[train_idx] as usize + (visit_idx + 1)];
            let earliest_out = t1 + visit.travel_time;
            if *t2 < earliest_out {
                *t2 = earliest_out;
            }
        }
    }

    type NodeRef = (u32, u32);
    type Interval = (i32, i32);

    let mut conflict_edges: Vec<(NodeRef, NodeRef)> = Default::default();
    let mut resource_usage: Vec<Vec<(Interval, NodeRef)>> = Default::default();

    drop(_p0);
    loop {
        let _p0 = hprof::enter("conflict check");

        // println!("iter {}", conflict_edges.len());
        // Find the chronologically first resource conflict
        for res in resource_usage.iter_mut() {
            res.clear();
        }

        // println!("find res");
        // Check travel time constraints and add resource usage.
        for (train_idx, train) in problem.trains.iter().enumerate() {
            for visit_idx in 0..train.visits.len() - 1 {
                let visit = train.visits[visit_idx];
                let t1 = t_vars[train_start_idx[train_idx] as usize + visit_idx];
                let t2 = t_vars[train_start_idx[train_idx] as usize + visit_idx + 1];

                while visit.resource_id >= resource_usage.len() {
                    resource_usage.push(Default::default());
                }

                if visit.resource_id != 0 {
                    resource_usage[visit.resource_id]
                        .push(((t1, t2), (train_idx as u32, visit_idx as u32)));
                }
            }
        }
        // println!("sort res");
        for res in resource_usage.iter_mut() {
            res.sort();
        }

        // println!("find confl");
        let mut conflicts: Vec<(i32, (NodeRef, NodeRef))> = Default::default();
        for (res_id, res) in resource_usage.iter().enumerate() {
            for i in 1..res.len() {
                let ((prev_in, prev_out), prev_node) = &res[i - 1];
                let ((next_in, next_out), next_node) = &res[i];

                let overlaps = prev_in < next_out && next_in < prev_out;
                if overlaps {
                    // println!(
                    //     "  oerlap res {} time {}  ({:?}  - {:?})",
                    //     res_id,
                    //     prev_in,
                    //     res[i - 1],
                    //     res[i]
                    // );
                }
                if overlaps {
                    let add =
                        use_strong_branching || (conflicts.is_empty() || *prev_in < conflicts[0].0);

                    // println!(
                    //     "res {} time {}  ({:?}  - {:?})",
                    //     res_id,
                    //     prev_in,
                    //     res[i - 1],
                    //     res[i]
                    // );

                    if add {
                        if !use_strong_branching {
                            conflicts.clear();
                        }
                        conflicts.push((*prev_in, (*prev_node, *next_node)));
                    }
                }
            }
        }

        // println!("Conflict {:?}", conflicts);

        drop(_p0);
        let _p0 = hprof::enter("edge eval");

        if conflicts.is_empty() {
            // The solution is valid -- just return it:
            println!("Finishing with cost {}", total_cost);
            return Ok(Some(
                problem
                    .trains
                    .iter()
                    .enumerate()
                    .map(|(ti, t)| {
                        (0..(t.visits.len() + 1))
                            .map(|vi| t_vars[train_start_idx[ti] as usize + vi])
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>(),
            ));
        }

        conflicts.sort();
        const MAX_STRONG_BRANCHING_VARS: usize = 50;
        conflicts.truncate(MAX_STRONG_BRANCHING_VARS);

        let mut best: Option<(((i32, i32), (i32, i32)), Vec<i32>, (NodeRef, NodeRef))> =
            Default::default();

        for (_, (n1, n2)) in conflicts {
            assert!((n1.1 as usize) < problem.trains[n1.0 as usize].visits.len());
            assert!((n2.1 as usize) < problem.trains[n2.0 as usize].visits.len());
            let n1_next = (n1.0, n1.1 + 1);
            let n2_next = (n2.0, n2.1 + 1);

            let mut cases = [
                (t_vars.clone(), (n1_next, n2), (-1, -1)),
                (t_vars.clone(), (n2_next, n1), (-1, -1)),
            ];

            'case: for (case_idx, (ts, (n1, n2), cost_increase)) in cases.iter_mut().enumerate() {
                // println!("eval case {}", case_idx);
                *cost_increase = (0, 0);
                let t1 = ts[train_start_idx[n1.0 as usize] as usize + n1.1 as usize];
                assert!(t1 > ts[train_start_idx[n2.0 as usize] as usize + n2.1 as usize]);

                let mut queue = VecDeque::new();
                queue.push_back((*n2, t1));

                while let Some((n @ (ti, vi), new_time)) = queue.pop_front() {
                    let train = &problem.trains[ti as usize];
                    let this_t = &mut ts[train_start_idx[ti as usize] as usize + vi as usize];
                    if *this_t >= new_time {
                        continue; // This node has already been updated.
                    }

                    let mut old_cost = (-1, -1);
                    let mut new_cost = (-1, -1);
                    if (vi as usize) < train.visits.len() {
                        old_cost = (
                            train.visit_delay_cost(delay_cost_type, vi as usize, *this_t) as i32,
                            train.visit_delay_cost(tiebreak_obj, vi as usize, *this_t) as i32,
                        );
                        new_cost = (
                            train.visit_delay_cost(delay_cost_type, vi as usize, new_time) as i32,
                            train.visit_delay_cost(tiebreak_obj, vi as usize, new_time) as i32,
                        );
                        cost_increase.0 += new_cost.0 - old_cost.0;
                        cost_increase.1 += new_cost.1 - old_cost.1;
                    }

                    assert!(*this_t < new_time);

                    // println!(
                    //     "  Setting {:?} from {} to {}  (cost {:?}--{:?})  total {:?}",
                    //     n, this_t, new_time, old_cost, new_cost, *cost_increase
                    // );
                    if n == *n1 {
                        // println!(" This makes a cycle to {:?}", *n1);
                        cost_increase.0 = i32::MAX;
                        continue 'case;
                    }

                    *this_t = new_time;
                    let this_t = *this_t;

                    // Travel edge
                    let has_travel_time_edge = (vi as usize) < train.visits.len();
                    let travel_time_edge = has_travel_time_edge
                        .then(|| ((ti, vi + 1), train.visits[vi as usize].travel_time));

                    // Conflict edges
                    let first_e_idx = conflict_edges.partition_point(|(nx, _)| *nx < n);
                    let es = conflict_edges
                        .iter()
                        .skip(first_e_idx)
                        .take_while(|(nx, _)| *nx == n)
                        .map(|(_, nx)| (*nx, 0));

                    let outgoing_edges = travel_time_edge.iter().copied().chain(es);

                    for (other_n @ (other_t, other_v), dt) in outgoing_edges {
                        let next_t =
                            &mut ts[train_start_idx[other_t as usize] as usize + other_v as usize];
                        let next_earliest = this_t + dt;
                        // println!(
                        //     "trying to going to other node {:?}  {}+{}<={} (->{})",
                        //     other_n, this_t, dt, next_t, next_earliest
                        // );

                        if *next_t < next_earliest {
                            // println!("    it is critical");
                            queue.push_back((other_n, next_earliest));
                        } else {
                            // println!("    it is unchanged")
                        }
                    }
                }
            }

            // Both cases have been evaluated
            let (lb1, lb2) = (cases[0].2, cases[1].2);
            assert!(lb1.0 >= 0 && lb2.0 >= 0);

            if lb1.0 == i32::MAX && lb2.0 == i32::MAX {
                // Instance is infeasible.
                return Err(SolverError::NoSolution);
            }

            let branching_goodness = (
                -(lb1.0.min(lb2.0) + 5 * lb1.0.max(lb2.0)),
                -(lb1.1.min(lb2.1) + 5 * lb1.1.max(lb2.1)),
            );

            let case_idx = if lb1 < lb2 { 0 } else { 1 };
            let cost = (branching_goodness, cases[case_idx].2);

            if best.is_none() || cost < best.as_ref().unwrap().0 {
                best = Some((
                    cost,
                    std::mem::take(&mut cases[case_idx].0),
                    cases[case_idx].1,
                ));
            }
        }

        if let Some(((_branch_cost, added_cost), ts, cfl)) = best {
            t_vars = ts;
            conflict_edges.push(cfl);
            conflict_edges.sort();
            total_cost += added_cost.0;
        } else {
            panic!("unreachable");
        }
    }
}

pub fn solve_heuristic(
    env: &grb::Env,
    problem: &Problem,
    solution: &Vec<Vec<i32>>,
) -> Result<Option<Vec<Vec<i32>>>, SolverError> {
    let priorities = {
        let _p = hprof::enter("priorities");
        let mut priorities = Vec::new();
        for (a @ (t1, v1), b @ (t2, v2)) in crate::solvers::bigm::visit_conflicts(&problem) {
            if solution[t1][v1] <= solution[t2][v2] {
                priorities.push((a, b));
            } else {
                priorities.push((b, a));
            }
        }
        priorities
    };

    match crate::solvers::minimize::minimize_solution(env, problem, priorities) {
        Ok(s) => Ok(Some(s)),
        Err(SolverError::NoSolution) => Ok(None), // It is expected that the heuristic could fail to produce a solution.
        Err(x) => Err(x),
    }
}
