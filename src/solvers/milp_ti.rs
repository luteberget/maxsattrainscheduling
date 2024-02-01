use grb::{add_binvar, c, expr::GurobiSum};
use std::collections::HashMap;

use super::{maxsatddd_ladder::SolveStats, SolverError};
use crate::{
    problem::{DelayCostType, Problem},
    solvers::bigm::visit_conflicts,
};

pub fn solve_milp_fixed_ti(
    env: &grb::Env,
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

    let _p_init = hprof::enter("solve_milp_fixed_ti encode");

    let mut solver = grb::Model::with_env("", env).unwrap();
    solver.set_param(grb::param::OutputFlag, 1).unwrap();
    solver.set_param(grb::param::LogToConsole, 1).unwrap();

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
                    let cost = train.visit_delay_cost(delay_cost_type, visit_idx, time) as u32;
                    // let v = solver.new_var();
                    // solver.add_clause(Some(cost), vec![-v]);
                    let v = add_binvar!(solver, obj: cost as f64).unwrap();
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
            solver
                .add_constr(
                    "select",
                    c!(t_vars[train_idx][visit_idx]
                        .iter()
                        .map(|(_, v)| v)
                        .grb_sum()
                        == 1),
                )
                .unwrap();
        }
    }

    // CONSTRAINT 2: travel time constraints
    for (t_idx, train) in problem.trains.iter().enumerate() {
        for (v1_idx, visit) in train.visits.iter().enumerate() {
            if v1_idx + 1 < train.visits.len() {
                let v2_idx = v1_idx + 1;
                for (t1, v1) in t_vars[t_idx][v1_idx].iter() {
                    let mut constraint = Vec::new();
                    for (t2, v2) in t_vars[t_idx][v2_idx].iter() {
                        let can_reach = *t1 + visit.travel_time <= *t2;
                        // if t_idx == 0 && v1_idx == 0 {
                        //     println!("  t0v0 {}-{} -- {}", t1, t2, can_reach);
                        // }
                        if !can_reach {
                            // solver.add_clause(None, vec![-v1, -v2]);
                            constraint.push(*v2);
                        }
                    }

                    if constraint.len() > 1 {
                        constraint.push(*v1);

                        solver
                            .add_constr("travel", c!(constraint.iter().grb_sum() <= 1))
                            .unwrap();
                    }
                }
            }
        }
    }

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
                                    // solver.add_clause(None, vec![-var1, -var2]);
                                    solver.add_constr("confl", c!(*var1 + *var2 <= 1)).unwrap();
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    drop(_p_init);
    let _p_solve = hprof::enter("solve_mip_fixed_ti solve");
    // solver.write("test.lp").unwrap();
    // panic!("ok");
    solver
        .set_param(
            grb::param::TimeLimit,
            timeout - start_time.elapsed().as_secs_f64(),
        )
        .unwrap();
    // let input_vec = solver
    //     .optimize(
    //         Some(),
    //         std::iter::empty(),
    //     )
    //     .map_err(|e| match e {
    //         crate::maxsatsolver::MaxSatError::NoSolution => SolverError::NoSolution,
    //         crate::maxsatsolver::MaxSatError::Timeout => SolverError::Timeout,
    //     })?
    //     .1;

    solver.optimize().unwrap();
    if solver.status().unwrap() == grb::Status::TimeLimit {
        return Err(SolverError::Timeout);
    } else if solver.status().unwrap() != grb::Status::Optimal {
        return Err(SolverError::NoSolution);
    }

    let mut solution = Vec::new();
    for train_idx in 0..t_vars.len() {
        let mut train_sol = Vec::new();

        for visit_idx in 0..t_vars[train_idx].len() {
            for (t, var) in t_vars[train_idx][visit_idx].iter().copied() {
                // assert!(var > 0);
                if solver.get_obj_attr(grb::attr::X, &var).unwrap() > 0.5 {
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

    let solution =
        solve_milp_fixed_ti(env, problem, timeout, delay_cost_type, time_discretization)?;

    let _p_post = hprof::enter("milp_ti postprocessing");

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
