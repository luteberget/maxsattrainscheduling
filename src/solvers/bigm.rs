use std::collections::HashSet;

use super::SolverError;
use crate::problem::{DelayCostType, Problem, DEFAULT_COST_THRESHOLDS};
const M: f64 = 100_000.0;

pub fn solve(
    env: &grb::Env,
    problem: &Problem,
    delay_cost_type: DelayCostType,
    lazy: bool,
    train_names: &[String],
    resource_names: &[String],
) -> Result<Vec<Vec<i32>>, SolverError> {
    let _p = hprof::enter("bigm solver");
    use grb::prelude::*;

    let mut model = Model::with_env("model1", env).map_err(SolverError::GurobiError)?;

    model
        .set_param(param::IntFeasTol, 1e-8)
        .map_err(SolverError::GurobiError)?;

    // timing variables
    let t_vars = problem
        .trains
        .iter()
        .enumerate()
        .map(|(train_idx, train)| {
            train
                .visits
                .iter()
                .enumerate()
                .map(|(visit_idx, visit)| {
                    add_ctsvar!(model,
                name : &format!("tn{}_v{}_tk{}", train_names[train_idx], visit_idx, resource_names[visit.resource_id]), 
                bounds: visit.earliest..)
                    .map_err(SolverError::GurobiError)
                })
                .collect::<Result<Vec<_>, _>>()
        })
        .collect::<Result<Vec<_>, _>>()?;

    // Travel time constraints
    for train_idx in 0..problem.trains.len() {
        for visit_idx in 0..problem.trains[train_idx].visits.len() - 1 {
            add_travel_constraint(
                problem,
                (train_idx, visit_idx),
                &mut model,
                &t_vars,
                train_names,
                resource_names,
            )?;
        }
    }

    // List all conflicting visits
    let visit_conflicts = visit_conflicts(problem);
    let mut added_conflicts = HashSet::new();

    if !lazy {
        for visit_pair in visit_conflicts.iter().copied() {
            add_conflict_constraint(visit_pair, &mut model, &t_vars, train_names, resource_names)?;
            assert!(added_conflicts.insert(visit_pair));
        }
    }

    // Objective
    match delay_cost_type {
        DelayCostType::Step123 => {
            let delay_cost = &DEFAULT_COST_THRESHOLDS;
            for (train_idx, train) in problem.trains.iter().enumerate() {
                for visit_idx in 0..train.visits.len() {
                    if let Some(aimed) = train.visits[visit_idx].aimed {
                        let time_var = t_vars[train_idx][visit_idx];

                        // create a variable for each delay threshold
                        let thresholds = &delay_cost.thresholds;
                        for threshold_idx in (0..thresholds.len()).rev() {
                            let (_prev_threshold, prev_cost) =
                                thresholds.get(threshold_idx + 1).unwrap_or(&(0, 0));
                            let (threshold, cost) = thresholds[threshold_idx];
                            let threshold = threshold;

                            let cost_diff = cost - prev_cost;
                            assert!(cost_diff > 0);

                            let threshold_var_name = format!(
                                "tn{}_v{}_tk{}_dly{}",
                                train_names[train_idx],
                                visit_idx,
                                resource_names[train.visits[visit_idx].resource_id],
                                threshold
                            );

                            // Add threshold_var to the objective with cost `diff_cost`.
                            #[allow(clippy::unnecessary_cast)]
                            let threshold_var =
                                add_intvar!(model, name: &threshold_var_name, bounds: 0..1, obj: cost_diff)
                                    .map_err(SolverError::GurobiError)?;

                            // If last_t - aimed >= threshold+1 then threshold_var must be 1

                            #[allow(clippy::useless_conversion)]
                            model
                                .add_constr(
                                    &format!("has_{}", threshold_var_name),
                                    c!(time_var - aimed <= threshold + M * threshold_var),
                                )
                                .map_err(SolverError::GurobiError)?;
                        }
                    }
                }
            }
        }
        DelayCostType::Continuous => {
            for (train_idx, train) in problem.trains.iter().enumerate() {
                for visit_idx in 0..train.visits.len() {
                    if let Some(aimed) = train.visits[visit_idx].aimed {
                        let time_var = t_vars[train_idx][visit_idx];
                        let objective_var_name = format!(
                            "tn{}_v{}_tk{}",
                            train_names[train_idx],
                            visit_idx,
                            resource_names[train.visits[visit_idx].resource_id],
                        );
                        let objective_var =
                            add_ctsvar!(model, name:&objective_var_name,bounds:0.., obj: 1.0)
                                .map_err(SolverError::GurobiError)?;

                        #[allow(clippy::useless_conversion)]
                        model
                            .add_constr(
                                &format!("bound_{}", objective_var_name),
                                c!(objective_var >= time_var - aimed),
                            )
                            .map_err(SolverError::GurobiError)?;
                    }
                }
            }
        }
    }

    let mut refinement_iterations = 0;
    loop {
        {
            let _p = hprof::enter("optimize");
            model.optimize().map_err(SolverError::GurobiError)?;
        }

        if model.status().map_err(SolverError::GurobiError)? != Status::Optimal {
            model.compute_iis().map_err(SolverError::GurobiError)?;
            model
                .write("infeasible.ilp")
                .map_err(SolverError::GurobiError)?;
            // let constrs = model.get_constrs().map_err(SolverError::GurobiError)?; // all constraints in model
            // let iis_constrs = model
            //     .get_obj_attr_batch(attr::IISConstr, constrs.iter().copied())
            //     .map_err(SolverError::GurobiError)?
            //     .into_iter()
            //     .zip(constrs)
            //     // IISConstr is 1 if constraint is in the IIS, 0 otherwise
            //     .filter_map(|(is_iis, c)| if is_iis > 0 { Some(*c) } else { None })
            //     .collect::<Vec<_>>();

            // println!("IIS {:?}", iis_constrs);
            return Err(SolverError::NoSolution);
        }
        let cost = model
            .get_attr(attr::ObjVal)
            .map_err(SolverError::GurobiError)?;

        // Check the conflicts
        let violated_conflicts = {
            let _p = hprof::enter("check conflicts");

            let mut cs = Vec::new();
            for visit_pair in visit_conflicts.iter().copied() {
                if !check_conflict(visit_pair, &model, &t_vars)? {
                    cs.push(visit_pair);
                }
            }
            cs
        };

        if !violated_conflicts.is_empty() {
            refinement_iterations += 1;
            for visit_pair in violated_conflicts {
                add_conflict_constraint(
                    visit_pair,
                    &mut model,
                    &t_vars,
                    train_names,
                    resource_names,
                )?;
                assert!(added_conflicts.insert(visit_pair));
            }
        } else {
            // success
            println!(
                "Solved with cost {} and {} conflict constraints after {} refinements",
                cost,
                added_conflicts.len(),
                refinement_iterations
            );

            model.write("model.lp").unwrap();
            model.write("model.sol").unwrap();

            let mut solution = Vec::new();
            for (train_idx, train_ts) in t_vars.iter().enumerate() {
                let mut train_solution = Vec::new();
                for visit_start_t_var in train_ts {
                    let t = model
                        .get_obj_attr(attr::X, visit_start_t_var)
                        .map_err(SolverError::GurobiError)?
                        .round() as i32;
                    train_solution.push(t);
                }

                train_solution.push(
                    train_solution.last().copied().unwrap()
                        + problem.trains[train_idx].visits.last().unwrap().travel_time,
                );
                solution.push(train_solution);
            }
            return Ok(solution);
        }
    }
}

fn add_travel_constraint(
    problem: &Problem,
    (train_idx, visit_idx): (usize, usize),
    model: &mut grb::Model,
    t_vars: &[Vec<grb::Var>],
    train_names: &[String],
    resource_names: &[String],
) -> Result<(), SolverError> {
    use grb::prelude::*;

    let visits = &problem.trains[train_idx].visits;
    #[allow(clippy::useless_conversion)]
    model
        .add_constr(
            &format!(
                "tn{}_v{}_tk{}_travel",
                train_names[train_idx],
                visit_idx,
                resource_names[problem.trains[train_idx].visits[visit_idx].resource_id]
            ),
            c!(
                (t_vars[train_idx][visit_idx + 1]) - (t_vars[train_idx][visit_idx])
                    >= visits[visit_idx].travel_time
            ),
        )
        .map_err(SolverError::GurobiError)?;
    Ok(())
}

pub fn visit_conflicts(problem: &Problem) -> Vec<((usize, usize), (usize, usize))> {
    let mut conflicts = Vec::new();
    let resource_conflicts = problem.conflicts.iter().copied().collect::<HashSet<_>>();
    for train_idx1 in 0..problem.trains.len() {
        for train_idx2 in (train_idx1 + 1)..problem.trains.len() {
            for visit_idx1 in 0..problem.trains[train_idx1].visits.len() {
                for visit_idx2 in 0..problem.trains[train_idx2].visits.len() {
                    let resource1 = problem.trains[train_idx1].visits[visit_idx1].resource_id;
                    let resource2 = problem.trains[train_idx2].visits[visit_idx2].resource_id;

                    let is_conflict1 = resource_conflicts.contains(&(resource1, resource2));
                    let is_conflict2 = resource_conflicts.contains(&(resource2, resource1));

                    assert!(is_conflict1 == is_conflict2);

                    if is_conflict1 || is_conflict2 {
                        conflicts.push(((train_idx1, visit_idx1), (train_idx2, visit_idx2)));
                    }
                }
            }
        }
    }
    conflicts
}

fn check_conflict(
    ((t1, v1), (t2, v2)): ((usize, usize), (usize, usize)),
    model: &grb::Model,
    t_vars: &[Vec<grb::Var>],
) -> Result<bool, SolverError> {
    use grb::prelude::*;

    let t_t1v1 = model
        .get_obj_attr(attr::X, &t_vars[t1][v1])
        .map_err(SolverError::GurobiError)?;
    let t_t1v1next = model
        .get_obj_attr(attr::X, &t_vars[t1][v1 + 1])
        .map_err(SolverError::GurobiError)?;
    let t_t2v2 = model
        .get_obj_attr(attr::X, &t_vars[t2][v2])
        .map_err(SolverError::GurobiError)?;
    let t_t2v2next = model
        .get_obj_attr(attr::X, &t_vars[t2][v2 + 1])
        .map_err(SolverError::GurobiError)?;

    let separation = (t_t2v2 - t_t1v1next).max(t_t1v1 - t_t2v2next);

    // let t1_first = t_t1v1next <= t_t2v2;
    // let t2_first = t_t2v2next <= t_t1v1;
    let has_separation = separation >= -1e-5;
    // println!("t1v1 {}-{}  t2v2 {}-{}  separation {} is_separated {}", t_t1v1, t_t1v1next, t_t2v2, t_t2v2next, separation, has_separation);
    // assert!((separation >= 1e-5) == has_separation);

    if has_separation {
        Ok(true)
    } else {
        // println!("conflict separation {:?}  {}", ((t1, v1), (t2, v2)), separation);

        Ok(false)
    }
}

fn add_conflict_constraint(
    ((t1, v1), (t2, v2)): ((usize, usize), (usize, usize)),
    model: &mut grb::Model,
    t_vars: &[Vec<grb::Var>],
    train_names: &[String],
    resource_names: &[String],
) -> Result<(), SolverError> {
    use grb::prelude::*;

    // println!("adding conflict {:?}", ((t1, v1), (t2, v2)));

    let choice_var_name = format!(
        "confl_tn{}_v{}_tn{}_v{}",
        train_names[t1], v1, train_names[t2], v2
    );

    #[allow(clippy::unnecessary_cast)]
    let choice_var = add_intvar!(model, name: &choice_var_name, bounds: 0..1)
        .map_err(SolverError::GurobiError)?;

    #[allow(clippy::useless_conversion)]
    model
        .add_constr(
            &format!("{}_first", choice_var_name),
            // t1 goes first: it reaches v1+1 before t2 reaches v2
            // if choice_var is 1, the constraint is disabled.
            c!(t_vars[t1][v1 + 1] <= t_vars[t2][v2] + M * choice_var),
        )
        .map_err(SolverError::GurobiError)?;

    #[allow(clippy::useless_conversion)]
    model
        .add_constr(
            &format!("{}_second", choice_var_name),
            // t2 goes first: it reaches v2+1 before t1 reaches v1
            c!(t_vars[t2][v2 + 1] <= t_vars[t1][v1] + M * (1 - choice_var)),
        )
        .map_err(SolverError::GurobiError)?;
    Ok(())
}
