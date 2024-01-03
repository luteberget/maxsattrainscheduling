use serde::Serialize;
use std::{
    collections::HashMap,
    fmt::Display,
    io::{BufRead, BufReader, Read},
};

use super::{maxsatddd_ladder::SolveStats, SolverError};
use crate::{
    maxsatsolver::{External, MaxSatSolver},
    problem::{DelayCostType, Problem},
    solvers::bigm::visit_conflicts,
};

pub fn solve_maxsat_fixed_ti(
    mut solver: impl MaxSatSolver,
    problem: &Problem,
    timeout: f64,
    delay_cost_type: DelayCostType,
    time_discretization: Vec<Vec<Vec<i32>>>,
) -> Result<Vec<Vec<i32>>, SolverError> {
    assert!(problem.trains.len() == time_discretization.len());
    assert!(problem
        .trains
        .iter()
        .enumerate()
        .all(|(train_idx, t)| t.visits.len() == time_discretization[train_idx].len()));

    let start_time = std::time::Instant::now();

    let _p_init = hprof::enter("solve_maxsat_fixed_ti encode");

    let mut conflicting_resources: HashMap<usize, Vec<usize>> = HashMap::new();
    for (a, b) in problem.conflicts.iter() {
        conflicting_resources.entry(*a).or_default().push(*b);
        if *a != *b {
            conflicting_resources.entry(*b).or_default().push(*a);
        }
    }

    let mut resource_visits: Vec<Vec<(usize, usize)>> = Vec::new();

    let mut t_vars = Vec::new();

    // Discretize each visit's time point and compute the costs.
    for (train_idx, train) in problem.trains.iter().enumerate() {
        let mut train_ts = Vec::new();
        for (visit_idx, visit) in train.visits.iter().enumerate() {
            while resource_visits.len() <= visit.resource_id {
                resource_visits.push(Vec::new());
            }

            resource_visits[visit.resource_id].push((train_idx, visit_idx));

            let time_vars = time_discretization[train_idx][visit_idx]
                .iter()
                .copied()
                .map(|time| {
                    let v = solver.new_var();
                    let cost = train.visit_delay_cost(delay_cost_type, visit_idx, time) as u32;
                    solver.add_clause(Some(cost), vec![-v]);
                    (time, v)
                })
                .collect::<Vec<_>>();
            train_ts.push(time_vars);
        }
        t_vars.push(train_ts);
    }

    // CONSTRAINT 1: select one interval per time slot
    for (train_idx, train) in problem.trains.iter().enumerate() {
        for (visit_idx, visit) in train.visits.iter().enumerate() {
            solver.exactly_one(
                &t_vars[train_idx][visit_idx]
                    .iter()
                    .map(|(_, v)| *v)
                    .collect::<Vec<_>>(),
            );
        }
    }

    println!("C1 {}", solver.status());

    // CONSTRAINT 2: travel time constraints
    for (t_idx, train) in problem.trains.iter().enumerate() {
        for (v1_idx, visit) in train.visits.iter().enumerate() {
            if v1_idx + 1 < train.visits.len() {
                let v2_idx = v1_idx + 1;
                for (t1, v1) in t_vars[t_idx][v1_idx].iter() {
                    let mut cannot_reach = Vec::new();
                    for (t2, v2) in t_vars[t_idx][v2_idx].iter() {
                        let can_reach = *t1 + visit.travel_time <= *t2;
                        // if t_idx == 0 && v1_idx == 0 {
                        //     println!("  t0v0 {}-{} -- {}", t1, t2, can_reach);
                        // }
                        if !can_reach {
                            solver.add_clause(None, vec![-v1, -v2]);
                            cannot_reach.push(*v2);
                        }
                    }

                    let mut clause = cannot_reach;
                    clause.push(*v1);

                    // at_most_one(&mut satproblem, &clause);
                }
            }
        }
    }

    println!("C2 {}", solver.status());

    // CONSTRAINT 3: resource conflict
    for (train1_idx, train) in problem.trains.iter().enumerate() {
        println!("   c3 t{}", train1_idx);
        for (visit1_idx, visit) in train.visits.iter().enumerate() {
            // Find conflicting visits
            if let Some(conflicting_resources) = conflicting_resources.get(&visit.resource_id) {
                for other_resource in conflicting_resources.iter().copied() {
                    for (train2_idx, visit2_idx) in resource_visits[other_resource].iter().copied()
                    {
                        if train2_idx == train1_idx {
                            continue;
                        }

                        for (t1_in, var1) in t_vars[train1_idx][visit1_idx].iter() {
                            for (t2_in, var2) in t_vars[train2_idx][visit2_idx].iter() {
                                let t1_out = t1_in + visit.travel_time;
                                let t2_out = t2_in
                                    + problem.trains[train2_idx].visits[visit2_idx].travel_time;

                                let separation = (*t2_in - t1_out).max(*t1_in - t2_out);
                                let has_separation = separation >= 0;

                                if !has_separation {
                                    solver.add_clause(None, vec![-var1, -var2]);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    println!("C3 {}", solver.status());

    drop(_p_init);
    let _p_solve = hprof::enter("solve_maxsat_fixed_ti solve");
    let input_vec = solver
        .optimize(Some(timeout - start_time.elapsed().as_secs_f64()))
        .map_err(|e| match e {
            crate::maxsatsolver::MaxSatError::NoSolution => SolverError::NoSolution,
            crate::maxsatsolver::MaxSatError::Timeout => SolverError::Timeout,
        })?
        .1;

    let mut solution = Vec::new();
    for train_idx in 0..t_vars.len() {
        let mut train_sol = Vec::new();

        for visit_idx in 0..t_vars[train_idx].len() {
            for (t, var) in t_vars[train_idx][visit_idx].iter().copied() {
                assert!(var > 0);
                if input_vec[var as usize - 1] {
                    train_sol.push(t);
                }
            }
            // Workaround for avoiding conflict in the special relaxation where stations are uncapacitated:
            if problem.trains[train_idx].visits[visit_idx].resource_id == 0 && visit_idx > 0 {
                train_sol[visit_idx] = train_sol[visit_idx - 1]
                    + problem.trains[train_idx].visits[visit_idx - 1].travel_time;
            }

            assert!(train_sol.len() == visit_idx + 1);
        }
        train_sol.push(
            train_sol.last().unwrap()
                + problem.trains[train_idx].visits.last().unwrap().travel_time,
        );
        solution.push(train_sol);
    }

    Ok(solution)
}

pub fn solve(
    mut solver: impl MaxSatSolver,
    env: &grb::Env,
    problem: &Problem,
    timeout: f64,
    delay_cost_type: DelayCostType,
    output_stats: impl FnMut(String, serde_json::Value),
    discretization_interval: u32,
    big_m: u32,
) -> Result<(Vec<Vec<i32>>, SolveStats), SolverError> {
    let start_time = std::time::Instant::now();
    let mut solver_time = std::time::Duration::ZERO;

    let discretization_interval = discretization_interval as i32;
    let big_m = big_m as i32;
    let round = |t: i32| {
        ((t + discretization_interval / 2) / discretization_interval) * discretization_interval
    };

    let time_discretization = problem
        .trains
        .iter()
        .map(|train| {
            train
                .visits
                .iter()
                .map(|visit| {
                    let mut time_vars = Vec::new();
                    let mut time = round(visit.earliest);
                    while time < visit.earliest + big_m {
                        time_vars.push(time);
                        time += discretization_interval;
                    }
                    time_vars
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();

    let solution = solve_maxsat_fixed_ti(
        solver,
        problem,
        timeout,
        delay_cost_type,
        time_discretization,
    )?;

    let _p_post = hprof::enter("maxsat_ti postprocessing");

    let mut priorities = Vec::new();
    for (a @ (t1, v1), b @ (t2, v2)) in visit_conflicts(&problem) {
        if solution[t1][v1] <= solution[t2][v2] {
            priorities.push((a, b));
        } else {
            priorities.push((b, a));
        }
    }

    let sol = crate::solvers::minimize::minimize_solution(env, problem, priorities).unwrap();

    return Ok((sol, SolveStats::default()));
}
