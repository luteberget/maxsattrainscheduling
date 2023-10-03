use std::{
    collections::{HashMap, HashSet},
    time::Instant,
};

use super::SolverError;
use crate::{
    problem::{DelayCostType, Problem},
    solvers::minimize,
};

const M: f64 = 6.0  * 3600.0;

struct TimeIndexedVar {
    vars: Vec<(i32, grb::Var)>,
    expr: grb::Expr,
}

fn round(x: i32, granularity: i32) -> i32 {
    (x as f32 / granularity as f32).round() as i32 * granularity
}

#[allow(clippy::too_many_arguments)]
pub fn solve_binarized_bigm(
    env: &grb::Env,
    problem: &Problem,
    delay_cost_type: DelayCostType,
    lazy_constraints: bool,
    ti_interval: i32,
    timeout: f64,
    train_names: &[String],
    resource_names: &[String],
    mut output_stats: impl FnMut(String, serde_json::Value),
) -> Result<Vec<Vec<i32>>, SolverError> {
    let _p = hprof::enter("binarized bigm solver");
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

    // model.set_param(grb::param::OutputFlag, 1);

    // Timing variables

    fn mk_ti_var(
        model: &mut Model,
        lb: i32,
        ub: i32,
        name: &str,
        ti_interval: i32,
        cost: impl Fn(i32) -> f64,
    ) -> TimeIndexedVar {
        let lb = round(lb, ti_interval);
        let ub = round(ub, ti_interval);

        let mut vars = Vec::new();
        let mut expr = Expr::Constant(0.0);

        let ts = (lb..=ub).step_by(ti_interval as usize);

        for t in ts {
            #[allow(clippy::unnecessary_cast)]
            let name = format!("{}_t{}", name, t);
            let v = add_binvar!(model, name: &name, obj: cost(t))
                .map_err(SolverError::GurobiError)
                .unwrap();

            vars.push((t, v));
            expr = expr + t * v;
        }

        model.add_constr(&format!("ti_{}", name),    c!(
            vars.iter().map(|(_,x)| *x).grb_sum() == 1
        )).unwrap();

        println!(
            "Time point {} with interval {} from lb={} to ub={} has {} binary vars",
            name,
            ti_interval,
            lb,
            ub,
            vars.len()
        );

        TimeIndexedVar { vars, expr }
    }

    // timing variables
    let t_vars: Vec<Vec<TimeIndexedVar>> = problem
        .trains
        .iter()
        .enumerate()
        .map(|(train_idx, train)| {
            train
                .visits
                .iter()
                .enumerate()
                .map(|(visit_idx, visit)| {
                    let name = format!(
                        "tn{}_v{}_tk{}",
                        train_names[train_idx], visit_idx, resource_names[visit.resource_id]
                    );
                    let lb = visit.earliest;
                    let ub = lb + M as i32;
                    mk_ti_var(&mut model, lb, ub, &name, ti_interval, |t| {
                        problem.trains[train_idx].visit_delay_cost(delay_cost_type, visit_idx, t)
                            as f64
                    })
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();

    model.update().unwrap();
    println!(
        "\n# VARS\nIn total {} vars for {} timepoints for {} trains.\n\n",
        model.get_vars().unwrap().len(),
        problem
            .trains
            .iter()
            .map(|t| t.visits.len() + 1)
            .sum::<usize>(),
        problem.trains.len(),
    );

    // Travel time constraints
    for train_idx in 0..problem.trains.len() {
        for visit_idx in 0..problem.trains[train_idx].visits.len() - 1 {
            add_travel_constraint(
                problem,
                (train_idx, visit_idx),
                &mut model,
                ti_interval,
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
            let var = add_bigm_conflict_constraint(&mut model, &conflict)?;
            assert!(priority_vars.insert(visit_pair, var).is_none());
            n_resource_constraints += 1;
            assert!(added_conflicts.insert(visit_pair));
        }
    }

    let mut refinement_iterations = 0usize;
    let mut solver_time = std::time::Duration::ZERO;
    // println!("INSTANCE");
    loop {
        {
            log::debug!(
                "Starting optimize on iteration {} with {} conflict constraints",
                refinement_iterations + 1,
                added_conflicts.len(),
            );
            println!("WRITING BIN BIGM");
            model
                .write(&format!("bin_bigm_model{}.lp", refinement_iterations + 1))
                .unwrap();
            model
                .write(&format!("bin_bigm_model{}.mps", refinement_iterations + 1))
                .unwrap();
            // model
            //     .write(&format!("model{}.mps", refinement_iterations + 1))
            //     .unwrap();
            let _p = hprof::enter("optimize");
            println!("Solving.");
            model
                .set_param(
                    param::TimeLimit,
                    timeout - start_time.elapsed().as_secs_f64(),
                )
                .map_err(SolverError::GurobiError)?;
            let start_solve = Instant::now();
            model.optimize().map_err(SolverError::GurobiError)?;
            solver_time += start_solve.elapsed();

            println!("Solve finished.");

            // let n_nodes = model.get_attr(grb::attr::NodeCount).map_err(SolverError::GurobiError)?;
            // println!("NODECOUNT {}", n_nodes);
            iteration += 1;
        }

        let status = model.status().map_err(SolverError::GurobiError)?;
        if status == Status::TimeLimit {
            return Err(SolverError::Timeout);
        } else if status == Status::Infeasible {
            println!("computing IIS");
            model.compute_iis().map_err(SolverError::GurobiError)?;
            println!("writing IIS.");
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

        const USE_MINIMIZE: bool = false;

        let solution = if USE_MINIMIZE {
            minimize::minimize_solution(env, problem, priorities)?
        } else {
            let mut solution = Vec::new();
            for (train_idx, train_ts) in t_vars.iter().enumerate() {
                let mut train_solution = Vec::new();
                for visit_start_t_var in train_ts {
                    let t = visit_start_t_var
                        .expr
                        .clone()
                        .into_quadexpr()
                        .get_value(&model)
                        .map_err(SolverError::GurobiError)? as i32;

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
        if !violated_conflicts.is_empty() {
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
                let var = add_bigm_conflict_constraint(&mut model, &conflict)?;
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
            output_stats(
                "total_time".to_string(),
                start_time.elapsed().as_secs_f64().into(),
            );
            output_stats("solver_time".to_string(), solver_time.as_secs_f64().into());
            output_stats(
                "algorithm_time".to_string(),
                (start_time.elapsed().as_secs_f64() - solver_time.as_secs_f64()).into(),
            );

            println!(
                "BINBIGMSTATS {} {} {}",
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
    ti_interval: i32,
    t_vars: &[Vec<TimeIndexedVar>],
    train_names: &[String],
    resource_names: &[String],
) -> Result<(), SolverError> {
    use grb::prelude::*;

    let visits = &problem.trains[train_idx].visits;

    let travel_time = round(visits[visit_idx].travel_time, ti_interval);

    #[allow(clippy::useless_conversion)]
    model
        .add_constr(
            &format!(
                "tn{}_v{}_tk{}_travel",
                train_names[train_idx],
                visit_idx,
                resource_names[problem.trains[train_idx].visits[visit_idx].resource_id]
            ),
            c!((t_vars[train_idx][visit_idx + 1].expr.clone())
                - (t_vars[train_idx][visit_idx].expr.clone())
                >= travel_time),
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
    t_vars: &'a [Vec<TimeIndexedVar>],
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
            c!(t_vars[t1][v1 + 1].expr.clone()
                <= t_vars[t2][v2].expr.clone() + M * (1 - choice_var)),
        )
        .map_err(SolverError::GurobiError)?;

    #[allow(clippy::useless_conversion)]
    model
        .add_constr(
            &format!("{}_second", choice_var_name),
            // t2 goes first: it reaches v2+1 before t1 reaches v1
            c!(t_vars[t2][v2 + 1].expr.clone() <= t_vars[t1][v1].expr.clone() + M * (choice_var)),
        )
        .map_err(SolverError::GurobiError)?;
    Ok(choice_var)
}
