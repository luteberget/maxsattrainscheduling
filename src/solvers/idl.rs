use std::collections::{HashMap, HashSet};

use satcoder::constraints::Totalizer;

use crate::problem::{DelayCostThresholds, Problem, DEFAULT_COST_THRESHOLDS};

use super::SolverError;

pub fn solve(problem: &Problem) -> Result<Vec<Vec<i32>>, SolverError> {
    let _p = hprof::enter("maxsat_idl solver");
    let mut solver = idl::IdlSolver::new();

    let zero = solver.zero();
    // time vars
    let t_vars = problem
        .trains
        .iter()
        .map(|t| {
            t.visits
                .iter()
                .map(|v| {
                    let var = solver.new_int();
                    solver.add_diff(None, zero, var, -v.earliest as i64);
                    var
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();

    // travel time constraint
    for (train_idx, train) in problem.trains.iter().enumerate() {
        for visit_idx in 0..train.visits.len() - 1 {
            // add_travel_constraint(problem, (train_idx, visit_idx), &mut model, &t_vars)?;
            let t1 = t_vars[train_idx][visit_idx];
            let t2 = t_vars[train_idx][visit_idx + 1];
            let dt = problem.trains[train_idx].visits[visit_idx].travel_time;
            solver.add_diff(None, t1, t2, -dt as i64);
        }
    }

    enum Soft<L: satcoder::Lit> {
        Delay,
        Totalizer(Totalizer<L>, usize),
    }

    struct SoftConstraint<L: satcoder::Lit> {
        weight: i32,
        original_weight: i32,
        constraint: Soft<L>,
    }

    let visit_conflicts = super::bigm::visit_conflicts(problem);
    let mut added_conflicts = HashSet::new();
    let mut soft_constraints: HashMap<idl::Lit, SoftConstraint<idl::Lit>> = HashMap::new();

    // Objective
    let delay_cost = &DEFAULT_COST_THRESHOLDS;
    for (train_idx, train) in problem.trains.iter().enumerate() {
        for visit_idx in 0..train.visits.len() {
            if let Some(aimed) = train.visits[visit_idx].aimed {
                let last_t = t_vars[train_idx][visit_idx];

                // create a variable for each delay threshold
                let thresholds = &delay_cost.thresholds;
                for threshold_idx in (0..thresholds.len()).rev() {
                    let (_prev_threshold, prev_cost) =
                        thresholds.get(threshold_idx + 1).unwrap_or(&(0, 0));
                    let (threshold, cost) = thresholds[threshold_idx];

                    let cost_diff = cost - prev_cost;
                    assert!(cost_diff > 0);

                    let threshold_var = solver.new_bool();
                    solver.add_diff(
                        Some(threshold_var),
                        last_t,
                        zero,
                        (aimed + threshold) as i64,
                    );
                    soft_constraints.insert(
                        threshold_var,
                        SoftConstraint {
                            weight: cost_diff as i32,
                            original_weight: cost_diff as i32,
                            constraint: Soft::Delay,
                        },
                    );
                }
            }
        }
    }

    let mut total_cost = 0;
    loop {
        enum Refinement {
            Conflicts(Vec<((usize, usize), (usize, usize))>),
            Core(Vec<idl::Lit>),
        }

        let refinement = {
            let result = {
                let _p = hprof::enter("idl solve_with_assumptions");
                let assumptions = soft_constraints.keys().copied().collect::<Vec<_>>();
                solver.solve_with_assumptions(&assumptions)
            };
            match result {
                Ok(model) => {
                    let mut conflicts = Vec::new();
                    {
                        let _p = hprof::enter("check conflicts");

                        for visit_pair in visit_conflicts.iter().copied() {
                            if !check_conflict(visit_pair, |v| model.get_int_value(v), &t_vars)? {
                                assert!(added_conflicts.insert(visit_pair));
                                conflicts.push(visit_pair);
                            }
                        }
                    }
                    if conflicts.is_empty() {
                        println!("Total cost {}", total_cost);
                        let zero_value = model.get_int_value(zero) as i32;
                        let mut solution = Vec::new();
                        for (train_idx, train_ts) in t_vars.iter().enumerate() {
                            let mut train_solution = Vec::new();
                            for visit_start_t_var in train_ts {
                                train_solution.push(
                                    model.get_int_value(*visit_start_t_var) as i32 - zero_value,
                                );
                            }

                            train_solution.push(
                                train_solution.last().copied().unwrap()
                                    + problem.trains[train_idx].visits.last().unwrap().travel_time,
                            );
                            solution.push(train_solution);
                        }
                        return Ok(solution);
                    } else {
                        Refinement::Conflicts(conflicts)
                    }
                }
                Err(core) => Refinement::Core(core.collect::<Vec<_>>()),
            }
        };

        match refinement {
            Refinement::Conflicts(c) => {
                for ((t1, v1), (t2, v2)) in c {
                    let choice = solver.new_bool();
                    solver.add_diff(Some(choice), t_vars[t1][v1 + 1], t_vars[t2][v2], 0);
                    solver.add_diff(Some(!choice), t_vars[t2][v2 + 1], t_vars[t1][v1], 0);
                }
            }
            Refinement::Core(core) => {
                if core.is_empty() {
                    return Err(SolverError::NoSolution); // UNSAT
                }

                let min_weight = core
                    .iter()
                    .map(|c| soft_constraints[c].weight)
                    .min()
                    .unwrap();
                println!("Core min weight {}", min_weight);

                for c in core.iter() {
                    let SoftConstraint {
                        weight,
                        original_weight,
                        constraint,
                    } = soft_constraints.remove(c).unwrap();

                    assert!(weight >= min_weight);
                    let new_weight = weight - min_weight;

                    if new_weight > 0 {
                        soft_constraints.insert(
                            *c,
                            SoftConstraint {
                                weight: new_weight,
                                original_weight,
                                constraint,
                            },
                        );
                    } else {
                        match constraint {
                            Soft::Delay => { /* primary soft constraint, when we relax we are done */
                            }
                            Soft::Totalizer(mut tot, bound) => {
                                // totalizer: need to extend its bound
                                let new_bound = bound + 1;
                                tot.increase_bound(&mut solver, new_bound as u32);
                                if new_bound < tot.rhs().len() {
                                    soft_constraints.insert(
                                        !tot.rhs()[new_bound].lit().unwrap(), // tot <= 2, 3, 4...
                                        SoftConstraint {
                                            weight: original_weight,
                                            original_weight,
                                            constraint: Soft::Totalizer(tot, new_bound),
                                        },
                                    );
                                }
                            }
                        }
                    }
                }

                total_cost += min_weight;
                if core.len() > 1 {
                    let bound = 1;
                    let tot = Totalizer::count(
                        &mut solver,
                        core.iter().map(|c| !satcoder::Bool::Lit(*c)),
                        bound as u32,
                    );
                    assert!(bound < tot.rhs().len());
                    soft_constraints.insert(
                        !(tot.rhs()[bound].lit().unwrap()), // tot <= 1
                        SoftConstraint {
                            weight: min_weight,
                            original_weight: min_weight,
                            constraint: Soft::Totalizer(tot, bound),
                        },
                    );
                } else {
                    satcoder::SatInstance::add_clause(&mut solver, vec![!core[0]]);
                }
            }
        }
    }
}

fn check_conflict(
    ((t1, v1), (t2, v2)): ((usize, usize), (usize, usize)),
    model: impl Fn(idl::DVar) -> i64,
    t_vars: &[Vec<idl::DVar>],
) -> Result<bool, SolverError> {
    let t_t1v1 = model(t_vars[t1][v1]);
    let t_t1v1next = model(t_vars[t1][v1 + 1]);
    let t_t2v2 = model(t_vars[t2][v2]);
    let t_t2v2next = model(t_vars[t2][v2 + 1]);

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
