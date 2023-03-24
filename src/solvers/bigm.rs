use std::{
    collections::{HashMap, HashSet},
    time::Instant,
};

use super::SolverError;
use crate::solvers::{
    maxsatddd::{self, TrainVisitCostBound},
    minimize,
};
use ddd_problem::problem::{iter_infinite_staircase, DelayCostThresholds, DelayCostType, Problem};
const M: f64 = 2.0 * 6.0 * 3600.0;

type ConflictHandler = fn(&mut grb::Model, &ConflictInformation) -> Result<grb::Var, SolverError>;

pub fn solve_bigm(
    env: &grb::Env,
    problem: &Problem,
    delay_cost_type: DelayCostType,
    lazy: bool,
    timeout: f64,
    train_names: &[String],
    resource_names: &[String],
    mut output_stats: impl FnMut(String, serde_json::Value),
    name: Option<&str>,
) -> Result<Vec<Vec<i32>>, SolverError> {
    solve(
        env,
        problem,
        delay_cost_type,
        lazy,
        timeout,
        train_names,
        resource_names,
        add_bigm_conflict_constraint,
        output_stats,
        name,
    )
}

pub fn solve_hull(
    env: &grb::Env,
    problem: &Problem,
    delay_cost_type: DelayCostType,
    lazy: bool,
    timeout: f64,

    train_names: &[String],
    resource_names: &[String],
    mut output_stats: impl FnMut(String, serde_json::Value),
    name: Option<&str>,
) -> Result<Vec<Vec<i32>>, SolverError> {
    solve(
        env,
        problem,
        delay_cost_type,
        lazy,
        timeout,
        train_names,
        resource_names,
        add_hull_conflict_constraint,
        output_stats,
        name,
    )
}

#[allow(clippy::too_many_arguments)]
fn solve(
    env: &grb::Env,
    problem: &Problem,
    delay_cost_type: DelayCostType,
    lazy_constraints: bool,
    timeout: f64,
    train_names: &[String],
    resource_names: &[String],
    add_conflict_constraint: ConflictHandler,
    mut output_stats: impl FnMut(String, serde_json::Value),
    name: Option<&str>,
) -> Result<Vec<Vec<i32>>, SolverError> {
    let _p = hprof::enter("bigm solver");
    use grb::prelude::*;

    let mut model = Model::with_env("model1", env).map_err(SolverError::GurobiError)?;
    let mut n_travel_constraints = 0;
    let mut n_resource_constraints = 0;
    let mut iteration = 0;
    let start_time = Instant::now();

    // model
    //     .set_param(param::IntFeasTol, 1e-8)
    //     .map_err(SolverError::GurobiError)?;

    // model
    // .set_param(param::LazyConstraints, 1)
    // .map_err(SolverError::GurobiError)?;

    model.set_param(grb::param::OutputFlag, 1);

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
            n_travel_constraints += 1;
        }
    }

    // List all conflicting visits
    let visit_conflicts = visit_conflicts(problem);
    let mut priority_vars = HashMap::new();
    let mut added_conflicts = HashSet::new();

    if !lazy_constraints {
        for visit_pair in visit_conflicts.iter().copied() {
            let conflict = ConflictInformation {
                problem,
                visit_pair,
                t_vars: &t_vars,
                train_names,
                resource_names,
            };
            let var = add_conflict_constraint(&mut model, &conflict)?;
            assert!(priority_vars.insert(visit_pair, var).is_none());
            n_resource_constraints += 1;
            assert!(added_conflicts.insert(visit_pair));
        }
    }

    let mut lazy_stepfunction: Option<HashMap<(usize, usize), Vec<()>>> = None;

    // Objective
    match delay_cost_type {
        DelayCostType::FiniteSteps1_3Min
        | DelayCostType::FiniteSteps1_5Min
        | DelayCostType::FiniteSteps123
        | DelayCostType::FiniteSteps12345
        | DelayCostType::FiniteSteps139 => {
            let delay_cost = match delay_cost_type {
                DelayCostType::FiniteSteps1_3Min => DelayCostThresholds::f1_3min(),
                DelayCostType::FiniteSteps1_5Min => DelayCostThresholds::f1_5min(),
                DelayCostType::FiniteSteps123 => DelayCostThresholds::f123(),
                DelayCostType::FiniteSteps12345 => DelayCostThresholds::f12345(),
                DelayCostType::FiniteSteps139 => DelayCostThresholds::f139(),
                _ => panic!(),
            };
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
        DelayCostType::InfiniteSteps60
        | DelayCostType::InfiniteSteps180
        | DelayCostType::InfiniteSteps360 => {
            // this is done lazily when a solution has been found
            //lazy_stepfunction = Some(Default::default());
            // The lazy_stepfunction uses big-M and can in princple have any shape, but since we know that
            // the step funtion has a fixed spacing and step increase, we can just use a single integer variable with cost.

            let interval = match delay_cost_type {
                DelayCostType::InfiniteSteps60 => 60,
                DelayCostType::InfiniteSteps180 => 180,
                DelayCostType::InfiniteSteps360 => 360,
                _ => panic!(),
            };

            let mut cost_vars: HashMap<(usize, usize), Var> = HashMap::new();
            let mut cost_threshold_vars: HashMap<TrainVisitCostBound, Var> = HashMap::new();

            for (train_idx, train) in problem.trains.iter().enumerate() {
                for visit_idx in 0..train.visits.len() {
                    if let Some(aimed) = train.visits[visit_idx].aimed {
                        let time_var = t_vars[train_idx][visit_idx];
                        let objective_var_name = format!(
                            "tn{}_v{}_tk{}_dlysteps",
                            train_names[train_idx],
                            visit_idx,
                            resource_names[train.visits[visit_idx].resource_id],
                        );
                        let objective_var =
                            add_intvar!(model, name:&objective_var_name,bounds:0.., obj: 1.0)
                                .map_err(SolverError::GurobiError)?;

                        cost_vars.insert((train_idx, visit_idx), objective_var);

                        #[allow(clippy::useless_conversion)]
                        model
                            .add_constr(
                                &format!("bound_{}", objective_var_name),
                                c!(interval * objective_var >= time_var - aimed),
                            )
                            .map_err(SolverError::GurobiError)?;
                    }
                }
            }

            const USE_MAXSAT_CUTS_WITH_TIMEOUT: Option<f64> = Some(10.0);

            let mut n_maxsat_cuts = 0;
            let mut total_cost = 0;
            if let Some(maxsat_timeout) = USE_MAXSAT_CUTS_WITH_TIMEOUT {
                for _ in 0..5 {
                    let _ = maxsatddd::solve(
                        satcoder::solvers::minisat::Solver::new(),
                        &problem,
                        maxsat_timeout,
                        delay_cost_type,
                        |_, _| {},
                        |tot_cost, coeffs, rhs| {
                            total_cost = tot_cost;
                            if coeffs.len() > 10 {
                                return;
                            }
                            let mut coeffs = coeffs.to_vec();
                            coeffs.sort_by_key(|(c, _)| (c.train, c.visit));
                            // Add missing variables
                            for (cost, _) in coeffs.iter() {
                                cost_threshold_vars
                            .entry(*cost)
                            .or_insert_with(|| {

                                let var = add_binvar!(model, name: &format!("t{}v{}c{}", cost.train, cost.visit, cost.cost), obj: 0.0)
                    .unwrap();
                model.add_constr(&format!("cost_t{}v{}c{}",  cost.train, cost.visit, cost.cost), c!(  cost_vars[&(cost.train, cost.visit)] >= var*cost.cost )).unwrap();
                                var
                            });
                            }

                            model
                                .add_constr(
                                    &format!("maxsatcut{}", n_maxsat_cuts),
                                    c!(coeffs
                                        .iter()
                                        .map(|(cost, coeff)| {
                                            (*coeff) * cost_threshold_vars[cost]
                                        })
                                        .sum::<Expr>()
                                        >= rhs),
                                )
                                .unwrap();

                            n_maxsat_cuts += 1;
                            println!("Got core {:?} {}", coeffs, rhs);
                        },
                    );
                }

                println!("Added {} maxsat cuts", n_maxsat_cuts);

                model
                    .add_constr(
                        &format!("maxsatcost"),
                        c!(cost_vars.iter().map(|(_, v)| { v }).sum::<Expr>() >= total_cost as f64),
                    )
                    .unwrap();
            }
        }
        DelayCostType::Continuous => {
            let mut cost_vars: HashMap<(usize, usize), Var> = HashMap::new();

            for (train_idx, train) in problem.trains.iter().enumerate() {
                for visit_idx in 0..train.visits.len() {
                    if let Some(aimed) = train.visits[visit_idx].aimed {
                        let time_var = t_vars[train_idx][visit_idx];
                        let objective_var_name = format!(
                            "tn{}_v{}_tk{}_dly",
                            train_names[train_idx],
                            visit_idx,
                            resource_names[train.visits[visit_idx].resource_id],
                        );
                        let objective_var =
                            add_ctsvar!(model, name:&objective_var_name,bounds:0.., obj: 1.0)
                                .map_err(SolverError::GurobiError)?;
                        cost_vars.insert((train_idx, visit_idx), objective_var);

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

            const USE_MAXSAT_CUTS_WITH_TIMEOUT: Option<f64> = Some(2.0);
            let mut cost_threshold_vars: HashMap<TrainVisitCostBound, Var> = HashMap::new();

            let mut n_maxsat_cuts = 0;
            if let Some(maxsat_timeout) = USE_MAXSAT_CUTS_WITH_TIMEOUT {
                let _ = maxsatddd::solve(
                    satcoder::solvers::minisat::Solver::new(),
                    &problem,
                    maxsat_timeout,
                    DelayCostType::InfiniteSteps180,
                    |_, _| {},
                    |_totalcost, coeffs, rhs| {
                        // Add missing variables
                        for (cost, _) in coeffs.iter() {
                            cost_threshold_vars
                            .entry(*cost)
                            .or_insert_with(|| {

                                let var = add_binvar!(model, name: &format!("t{}v{}c{}", cost.train, cost.visit, cost.cost), obj: 0.0)
                    .unwrap();
                model.add_constr(&format!("cost_t{}v{}c{}",  cost.train, cost.visit, cost.cost), c!(  cost_vars[&(cost.train, cost.visit)] >= var*180.0*cost.cost )).unwrap();
                                var
                            });
                        }

                        model
                            .add_constr(
                                &format!("maxsatcut{}", n_maxsat_cuts),
                                c!(coeffs
                                    .iter()
                                    .map(|(cost, coeff)| { (*coeff) * cost_threshold_vars[cost] })
                                    .sum::<Expr>()
                                    >= rhs),
                            )
                            .unwrap();

                        n_maxsat_cuts += 1;
                        println!("Got core {:?} {}", coeffs, rhs);
                    },
                );

                println!("Added {} maxsat cuts", n_maxsat_cuts);
            }
        }
    }

    let mut refinement_iterations = 0usize;
    // println!("INSTANCE");
    loop {
        {
            log::debug!(
                "Starting optimize on iteration {} with {} conflict constraints {:?} lazy objectives",
                refinement_iterations + 1,
                added_conflicts.len(),
                lazy_stepfunction.as_ref().map(|l| l.values().map(|x| x.len()).sum::<usize>()),
            );
            // model
            //     .write(&format!("model{}.lp", refinement_iterations + 1))
            //     .unwrap();
            // model
            //     .write(&format!("model{}.mps", refinement_iterations + 1))
            //     .unwrap()

            if !lazy_constraints {
                model
                    .write(&format!("{}_full_bigm.lp", name.unwrap()))
                    .unwrap()
            }

            let _p = hprof::enter("optimize");
            model
                .set_param(
                    param::TimeLimit,
                    timeout - start_time.elapsed().as_secs_f64(),
                )
                .map_err(SolverError::GurobiError)?;
            model.optimize().map_err(SolverError::GurobiError)?;

            // let n_nodes = model.get_attr(grb::attr::NodeCount).map_err(SolverError::GurobiError)?;
            // println!("NODECOUNT {}", n_nodes);
            iteration += 1;
        }

        let status = model.status().map_err(SolverError::GurobiError)?;
        if status == Status::TimeLimit {
            return Err(SolverError::Timeout);
        } else if status == Status::Infeasible {
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
        } else if status != Status::Optimal {
            dbg!(status);
            panic!("Unknown status type.");
        }

        let cost = model
            .get_attr(attr::ObjVal)
            .map_err(SolverError::GurobiError)?;

        let priorities = priority_vars
            .iter()
            .map(|((a, b), v)| {
                let choice = model
                    .get_obj_attr(attr::X, v)
                    .map_err(SolverError::GurobiError)
                    .unwrap()
                    > 0.5;

                if choice {
                    (*a, *b)
                } else {
                    (*b, *a)
                }
            })
            .collect::<Vec<_>>();

        const USE_MINIMIZE: bool = true;

        let solution = if USE_MINIMIZE {
            minimize::minimize_solution(env, problem, priorities)?
        } else {
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
            solution
        };

        // println!("Iteration {} cost {}", refinement_iterations, cost);

        // Check the conflicts
        let violated_conflicts = {
            let _p = hprof::enter("check conflicts");

            let mut cs: HashMap<(usize, usize), Vec<(usize, usize)>> = HashMap::new();
            for visit_pair @ ((t1, v1), (t2, v2)) in visit_conflicts.iter().copied() {
                if !check_conflict(visit_pair, |t, v| solution[t][v])? {
                    cs.entry((t1, t2)).or_default().push((v1, v2));
                }
            }
            cs
        };

        // Check the lazy objective
        let mut refined_objective = false;
        if let Some(lazy_stepfunction) = lazy_stepfunction.as_mut() {
            let interval = match delay_cost_type {
                DelayCostType::InfiniteSteps60 => 60,
                DelayCostType::InfiniteSteps180 => 180,
                DelayCostType::InfiniteSteps360 => 360,
                _ => panic!(),
            };

            for (train_idx, train) in problem.trains.iter().enumerate() {
                for visit_idx in 0..train.visits.len() {
                    if let Some(aimed) = train.visits[visit_idx].aimed {
                        let time_var = t_vars[train_idx][visit_idx];
                        let curr_t = solution[train_idx][visit_idx];
                        // println!("t{} {} @ {}  delay {}", train_idx, visit_idx, curr_t, (curr_t - aimed).max(0));

                        let steps = lazy_stepfunction.entry((train_idx, visit_idx)).or_default();
                        // let mut added_this = false;
                        for (threshold, cost) in iter_infinite_staircase(interval).skip(steps.len())
                        {
                            if cost > train.visit_delay_cost(delay_cost_type, visit_idx, curr_t) {
                                // println!("  next cost {} exceeds {}", cost, train.visit_delay_cost(delay_cost_type, visit_idx, curr_t));
                                break;
                            }

                            let threshold = threshold - 1;

                            // added_this = true;
                            // println!("Adding t{} v{} thr{} cost{} on {}", train_idx, visit_idx, threshold, cost, train.visit_delay_cost(delay_cost_type, visit_idx, curr_t));

                            let cost_diff = 1;
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
                                add_binvar!(model, name: &threshold_var_name, obj: cost_diff)
                                    .map_err(SolverError::GurobiError)?;

                            // If last_t - aimed >= threshold+1 then threshold_var must be 1

                            #[allow(clippy::useless_conversion)]
                            model
                                .add_constr(
                                    &format!("has_{}", threshold_var_name),
                                    c!(time_var - aimed <= threshold + M * threshold_var),
                                )
                                .map_err(SolverError::GurobiError)?;

                            // println!("added time step {} {} {} {} {}", train_idx, visit_idx, aimed, curr_t, threshold);

                            steps.push(());
                            refined_objective = true;
                        }

                        // if added_this {
                        //     println!("Train {} v {} steps {}", train_idx, visit_idx, steps.len());
                        // }
                    }
                }
            }
        };

        if !violated_conflicts.is_empty() || refined_objective {
            refinement_iterations += 1;
            for ((t1, t2), pairs) in violated_conflicts {
                let (v1, v2) = *pairs.iter().min_by_key(|(v1, v2)| v1 + v2).unwrap();
                let visit_pair = ((t1, v1), (t2, v2));
                let conflict = ConflictInformation {
                    problem,
                    visit_pair,
                    t_vars: &t_vars,
                    train_names,
                    resource_names,
                };
                let var = add_conflict_constraint(&mut model, &conflict)?;
                assert!(priority_vars.insert(visit_pair, var).is_none());
                n_resource_constraints += 1;
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

            output_stats("refinements".to_string(), refinement_iterations.into());
            output_stats(
                "added_conflict_pairs".to_string(),
                added_conflicts.len().into(),
            );
            output_stats("iteration".to_string(), iteration.into());
            output_stats(
                "travel_constraints".to_string(),
                n_travel_constraints.into(),
            );
            output_stats(
                "resource_constraints".to_string(),
                n_resource_constraints.into(),
            );
            output_stats("internal_cost".to_string(), cost.into());

            // let mip_objective =  model.get_attr(grb::attr::ObjVal).unwrap();
            // println!("MIP obj {}", mip_objective);

            // model.write("model_final.sol").unwrap();
            // model.write("model_final.lp").unwrap();

            // let mut solution = Vec::new();
            // for (train_idx, train_ts) in t_vars.iter().enumerate() {
            //     let mut train_solution = Vec::new();
            //     for visit_start_t_var in train_ts {
            //         let t = model
            //             .get_obj_attr(attr::X, visit_start_t_var)
            //             .map_err(SolverError::GurobiError)?
            //             .round() as i32;
            //         train_solution.push(t);
            //     }

            //     train_solution.push(
            //         train_solution.last().copied().unwrap()
            //             + problem.trains[train_idx].visits.last().unwrap().travel_time,
            //     );

            //     solution.push(train_solution);
            // }

            // // let t0v11 = solution[0][11];
            // //     println!("SOlutino [0][11] {} cost {}", t0v11, problem.trains[0].visit_delay_cost(delay_cost_type, 11, t0v11));

            println!(
                "BIGMSTATS {} {} {}",
                iteration, n_travel_constraints, n_resource_constraints
            );
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

pub fn check_conflict(
    ((t1, v1), (t2, v2)): ((usize, usize), (usize, usize)),
    t_vars: impl Fn(usize, usize) -> i32,
) -> Result<bool, SolverError> {
    let t_t1v1 = t_vars(t1, v1);
    let t_t1v1next = t_vars(t1, v1 + 1);
    let t_t2v2 = t_vars(t2, v2);
    let t_t2v2next = t_vars(t2, v2 + 1);

    let separation = (t_t2v2 - t_t1v1next).max(t_t1v1 - t_t2v2next);

    // let t1_first = t_t1v1next <= t_t2v2;
    // let t2_first = t_t2v2next <= t_t1v1;
    let has_separation = separation >= 0;
    // println!("t1v1 {}-{}  t2v2 {}-{}  separation {} is_separated {}", t_t1v1, t_t1v1next, t_t2v2, t_t2v2next, separation, has_separation);
    // assert!((separation >= 1e-5) == has_separation);

    if has_separation {
        Ok(true)
    } else {
        // println!("conflict separation {:?}  {}", ((t1, v1), (t2, v2)), separation);

        Ok(false)
    }
}

struct ConflictInformation<'a> {
    problem: &'a Problem,
    visit_pair: ((usize, usize), (usize, usize)),
    t_vars: &'a [Vec<grb::Var>],
    train_names: &'a [String],

    #[allow(dead_code)]
    resource_names: &'a [String],
}

fn add_bigm_conflict_constraint(
    model: &mut grb::Model,
    conflict: &ConflictInformation,
) -> Result<grb::Var, SolverError> {
    use grb::prelude::*;

    let &ConflictInformation {
        problem: _,
        visit_pair: ((t1, v1), (t2, v2)),
        t_vars,
        resource_names: _,
        train_names,
    } = conflict;

    // println!("adding conflict {:?}", ((t1, v1), (t2, v2)));

    let choice_var_name = format!(
        "confl_tn{}_v{}_tn{}_v{}",
        train_names[t1], v1, train_names[t2], v2
    );

    #[allow(clippy::unnecessary_cast)]
    let choice_var =
        add_binvar!(model, name: &choice_var_name).map_err(SolverError::GurobiError)?;

    #[allow(clippy::useless_conversion)]
    model
        .add_constr(
            &format!("{}_first", choice_var_name),
            // t1 goes first: it reaches v1+1 before t2 reaches v2
            // if choice_var is 1, the constraint is disabled.
            c!(t_vars[t1][v1 + 1] <= t_vars[t2][v2] + M * (1 - choice_var)),
        )
        .map_err(SolverError::GurobiError)?;

    #[allow(clippy::useless_conversion)]
    model
        .add_constr(
            &format!("{}_second", choice_var_name),
            // t2 goes first: it reaches v2+1 before t1 reaches v1
            c!(t_vars[t2][v2 + 1] <= t_vars[t1][v1] + M * (choice_var)),
        )
        .map_err(SolverError::GurobiError)?;
    Ok(choice_var)
}

fn add_hull_conflict_constraint(
    model: &mut grb::Model,
    conflict: &ConflictInformation,
) -> Result<grb::Var, SolverError> {
    use grb::prelude::*;

    let &ConflictInformation {
        problem,
        visit_pair: ((t1, v1), (t2, v2)),
        t_vars,
        resource_names: _,
        train_names: _,
    } = conflict;

    // println!("adding conflict {:?}", ((t1, v1), (t2, v2)));

    let choice_var_name = format!(
        "confl_tn{}_v{}_tn{}_v{}",
        //train_names[t1], v1, train_names[t2], v2
        t1,
        v1,
        t2,
        v2
    );

    #[allow(clippy::unnecessary_cast)]
    let choice_var = add_intvar!(model, name: &choice_var_name, bounds: 0..1)
        .map_err(SolverError::GurobiError)?;

    // #[allow(clippy::useless_conversion)]
    // model
    //     .add_constr(
    //         &format!("{}_first", choice_var_name),
    //         // t1 goes first: it reaches v1+1 before t2 reaches v2
    //         // if choice_var is 1, the constraint is disabled.
    //         c!(t_vars[t1][v1 + 1] <= t_vars[t2][v2] + M * choice_var),
    //     )
    //     .map_err(SolverError::GurobiError)?;

    // #[allow(clippy::useless_conversion)]
    // model
    //     .add_constr(
    //         &format!("{}_second", choice_var_name),
    //         // t2 goes first: it reaches v2+1 before t1 reaches v1
    //         c!(t_vars[t2][v2 + 1] <= t_vars[t1][v1] + M * (1 - choice_var)),
    //     )
    //     .map_err(SolverError::GurobiError)?;

    // we have
    //  OR (
    //    t_vars[t2][v2+1] <= t_vars[t1][v1],
    //    t_vars[t1][v1+1] <= t_vars[t2][v2],
    //  )
    //
    //  ... and instead of the Big-M formulation:
    //      t2f <= t1s + M*y
    //      t1f <= t2s + M*(1-y)
    //
    //  ... we will use the convex hull formulation:
    //
    //     t1s = t1s_a + t1s_b
    //     t1f = t1f_a + t1f_b
    //     t2s = t2s_a + t2s_b
    //     t2f = t2f_a + t2f_b
    //     t1s_a <= M * y
    //     t1f_a <= M * y
    //     t2s_a <= M * y
    //     t2f_a <= M * y
    //     t1s_b <= M * (1 - y)
    //     t1f_b <= M * (1 - y)
    //     t2s_b <= M * (1 - y)
    //     t2f_b <= M * (1 - y)
    //     constraint_A: -M(1-y) <= t2f_a - t1s_a <= 0
    //     constraint_B: -My     <= t1f_b - t2s_b <= 0
    //
    // See https://optimization.cbe.cornell.edu/index.php?title=Disjunctive_inequalities
    // for the general transformation.
    //

    let split_vars = [(t1, v1), (t1, v1 + 1), (t2, v2), (t2, v2 + 1)]
        .iter()
        .copied()
        .map(|(t, v)| {
            let t_a = add_ctsvar!(model, name :&format!("a_{}_{}", t, v), bounds: 0..)
                .map_err(SolverError::GurobiError)?;
            let t_b = add_ctsvar!(model, name :&format!("b_{}_{}", t, v), bounds: 0..)
                .map_err(SolverError::GurobiError)?;
            let lb = problem.trains[t].visits[v].earliest;

            #[allow(clippy::useless_conversion)]
            model
                .add_constr(
                    &format!("{}_t{}v{}_split", choice_var_name, t, v),
                    c!(t_vars[t][v] == lb + t_a + t_b),
                )
                .map_err(SolverError::GurobiError)?;
            #[allow(clippy::useless_conversion)]
            model
                .add_constr(
                    &format!("{}_t{}v{}_sel1", choice_var_name, t, v),
                    c!(t_a <= M * (1.0f64 - choice_var)),
                )
                .map_err(SolverError::GurobiError)?;
            #[allow(clippy::useless_conversion)]
            model
                .add_constr(
                    &format!("{}_t{}v{}_sel2", choice_var_name, t, v),
                    c!(t_b <= M * choice_var),
                )
                .map_err(SolverError::GurobiError)?;

            Ok((t_a, t_b))
        })
        .collect::<Result<Vec<(grb::Var, grb::Var)>, SolverError>>()?;

    let t1s_a = split_vars[0].0;
    let t1s_lb = problem.trains[t1].visits[v1].earliest;
    let t2f_a = split_vars[3].0;
    let t2f_lb = problem.trains[t2].visits[v2 + 1].earliest;

    let t2s_b = split_vars[2].1;
    let t2s_lb = problem.trains[t2].visits[v2].earliest;
    let t1f_b = split_vars[1].1;
    let t1f_lb = problem.trains[t1].visits[v1 + 1].earliest;

    #[allow(clippy::useless_conversion)]
    model
        .add_constr(
            &format!("{}_a", choice_var_name),
            c!((t2f_a + (1 - choice_var) * t2f_lb) - (t1s_a + (1 - choice_var) * t1s_lb) <= 0.0f64),
        )
        .map_err(SolverError::GurobiError)?;
    #[allow(clippy::useless_conversion)]
    model
        .add_constr(
            &format!("{}_b", choice_var_name),
            c!((t1f_b + choice_var * t1f_lb) - (t2s_b + choice_var * t2s_lb) <= 0.0f64),
        )
        .map_err(SolverError::GurobiError)?;

    Ok(choice_var)
}
