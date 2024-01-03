use std::collections::{HashMap, VecDeque};

use log::info;

use crate::{
    maxsatsolver::MaxSatSolver,
    problem::{DelayCostType, Problem},
};

use super::{maxsatddd_ladder::SolveStats, SolverError};


pub fn solve_incremental<S: MaxSatSolver>(
    mk_solver: fn() -> S,
    env: &grb::Env,
    problem: &Problem,
    timeout: f64,
    delay_cost_type: DelayCostType,
    mut output_stats: impl FnMut(String, serde_json::Value),
) -> Result<(Vec<Vec<i32>>, SolveStats), SolverError> {

    todo!()
}


const M: f64 = 100_000.0;

pub fn solve<S: MaxSatSolver>(
    mk_solver: fn() -> S,
    env: &grb::Env,
    problem: &Problem,
    timeout: f64,
    delay_cost_type: DelayCostType,
    mut output_stats: impl FnMut(String, serde_json::Value),
) -> Result<(Vec<Vec<i32>>, SolveStats), SolverError> {
    // assert!(matches!(delay_cost_type, DelayCostType::FiniteSteps123));
    let start_time = std::time::Instant::now();
    let mut solver_time = std::time::Duration::ZERO;

    let _p_init = hprof::enter("maxsat ddd external solve init");
    let mut intervals = problem
        .trains
        .iter()
        .map(|t| {
            t.visits
                .iter()
                .map(|v| vec![(v.earliest, M as i32)])
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();

    let mut occupations = Vec::new();
    let mut resource_usage: HashMap<usize, Vec<(usize, usize)>> = HashMap::new();
    for (train_idx, train) in problem.trains.iter().enumerate() {
        for (visit_idx, visit) in train.visits.iter().enumerate() {
            occupations.push((train_idx, visit_idx));
            if problem
                .conflicts
                .contains(&(visit.resource_id, visit.resource_id))
            {
                resource_usage
                    .entry(visit.resource_id)
                    .or_default()
                    .push((train_idx, visit_idx));
            }
        }
    }

    let visit_conflicts = super::bigm::visit_conflicts(problem);
    drop(_p_init);

    let mut iteration = 0;
    loop {
        iteration += 1;
        let _p_wcnf = hprof::enter("build wcnf");
        let mut solver = crate::maxsatsolver::External::new();

        let t_vars = intervals
            .iter()
            .enumerate()
            .map(|(train_idx, t)| {
                t.iter()
                    .enumerate()
                    .map(|(visit_idx, v)| {
                        v.iter()
                            .map(|(time, time_out)| {
                                let cost = problem.trains[train_idx].visit_delay_cost(
                                    delay_cost_type,
                                    visit_idx,
                                    *time,
                                ) as u32;
                                let v = solver.new_var();
                                solver.add_clause(Some(cost), vec![-v]);
                                v
                            })
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();

        // CONSTRAINT 1: select one interval per time slot
        for (train_idx, train) in problem.trains.iter().enumerate() {
            for (visit_idx, _) in train.visits.iter().enumerate() {
                solver.exactly_one(&t_vars[train_idx][visit_idx]);
            }
        }

        #[allow(clippy::type_complexity)]
        let mut incompat: HashMap<(usize, usize, usize), Vec<(usize, usize, usize)>> =
            Default::default();
        let mut n_travel_time_constraints = 0;
        let mut n_conflict_constraints = 0;

        // travel time constraint
        for (train_idx, train) in problem.trains.iter().enumerate() {
            for visit_idx in 0..train.visits.len() - 1 {
                for (idx1, (v1start, _v1end)) in intervals[train_idx][visit_idx].iter().enumerate()
                {
                    let dt = problem.trains[train_idx].visits[visit_idx].travel_time;
                    let i1 = t_vars[train_idx][visit_idx][idx1];
                    let incompatible_intervals = intervals[train_idx][visit_idx + 1]
                        .iter()
                        .enumerate()
                        .take_while(|(_, (_v2start, v2end))| v1start + dt >= *v2end)
                        .map(|(idx2, _)| idx2)
                        .collect::<Vec<_>>();

                    for idx2 in incompatible_intervals.iter().copied() {
                        let i2 = t_vars[train_idx][visit_idx + 1][idx2];

                        incompat
                            .entry((train_idx, visit_idx, idx1))
                            .or_default()
                            .push((train_idx, visit_idx + 1, idx2));
                        incompat
                            .entry((train_idx, visit_idx + 1, idx2))
                            .or_default()
                            .push((train_idx, visit_idx, idx1));

                        solver.at_most_one(&[i1, i2]);
                        n_travel_time_constraints += 1;
                    }
                }
            }
        }

        // Conflict constraints
        for (_res, vs) in resource_usage.iter() {
            for (t1, v1) in vs.iter() {
                for (t2, v2) in vs.iter() {
                    if t1 == t2 {
                        assert!(v1 == v2);
                        continue;
                    }
                    if t1 > t2 {
                        continue;
                    }

                    let travel1 = problem.trains[*t1].visits[*v1].travel_time;
                    let travel2 = problem.trains[*t2].visits[*v2].travel_time;
                    for (t1_idx, (t1_start, t1_end)) in intervals[*t1][*v1].iter().enumerate() {
                        for (t2_idx, (t2_start, t2_end)) in intervals[*t2][*v2].iter().enumerate() {
                            let test1 = *t2_start + travel2 >= *t1_end;
                            let test2 = *t1_start + travel1 >= *t2_end;
                            let incompatible = test1 && test2;
                            if incompatible {
                                // println!("incomp t{}v{}x{}-{},dt{}  t{}v{}x{}-{},dt{}", t1, v1, t1_start, t1_end, travel1, t2, v2, t2_start, t2_end, travel2);
                                let i1 = t_vars[*t1][*v1][t1_idx];
                                let i2 = t_vars[*t2][*v2][t2_idx];

                                incompat
                                    .entry((*t1, *v1, t1_idx))
                                    .or_default()
                                    .push((*t2, *v2, t2_idx));
                                incompat
                                    .entry((*t2, *v2, t2_idx))
                                    .or_default()
                                    .push((*t1, *v1, t1_idx));

                                solver.at_most_one(&[i1, i2]);
                                n_conflict_constraints += 1;
                            }
                        }
                    }
                }
            }
        }

        drop(_p_wcnf);
        let _p = hprof::enter("solve wcnf");

        info!(
            "Solving maxsat ddd external iteration {} with {} travel time {} conflicts",
            iteration, n_travel_time_constraints, n_conflict_constraints
        );

        let bool_vec = solver
            .optimize(Some(timeout - start_time.elapsed().as_secs_f64()))
            .map_err(|e| match e {
                crate::maxsatsolver::MaxSatError::NoSolution => SolverError::NoSolution,
                crate::maxsatsolver::MaxSatError::Timeout => SolverError::Timeout,
            })?
            .1;

        let mut selected_interval = intervals
            .iter()
            .enumerate()
            .map(|(train_idx, ts)| {
                ts.iter()
                    .enumerate()
                    .map(|(visit_idx, is)| {
                        is.iter()
                            .enumerate()
                            .position(|(iidx, _)| {
                                let v = t_vars[train_idx][visit_idx][iidx];
                                assert!(v > 0);
                                bool_vec[v as usize - 1]
                            })
                            .unwrap()
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();

        let mut consecutive_unmodified_occupations = 0;
        let mut i = 0;
        let occs_len = occupations.len();
        while consecutive_unmodified_occupations < occs_len {
            let mut touched = false;
            let occ_idx = i % occupations.len();
            let (train_idx, visit_idx) = occupations[occ_idx];
            while selected_interval[train_idx][visit_idx] > 0 {
                let prev_interval = selected_interval[train_idx][visit_idx] - 1;
                let can_reduce = !incompat
                    .entry((train_idx, visit_idx, prev_interval))
                    .or_default()
                    .iter()
                    .copied()
                    .any(|(t2, v2, i2)| selected_interval[t2][v2] == i2);

                if can_reduce {
                    selected_interval[train_idx][visit_idx] = prev_interval;
                    touched = true;
                    consecutive_unmodified_occupations = 0;
                } else {
                    break;
                }
            }
            i += 1;
            if !touched {
                consecutive_unmodified_occupations += 1;
            }
        }

        let solution: Vec<Vec<i32>> = selected_interval
            .iter()
            .enumerate()
            .map(|(train_idx, vs)| {
                let mut train_times = vs
                    .iter()
                    .enumerate()
                    .map(|(visit_idx, interval_idx)| {
                        intervals[train_idx][visit_idx][*interval_idx].0
                    })
                    .collect::<Vec<_>>();
                let last = train_times.last().copied().unwrap()
                    + problem.trains[train_idx].visits.last().unwrap().travel_time;
                train_times.push(last);
                train_times
            })
            .collect::<Vec<_>>();

        let mut new_intervals = VecDeque::new();
        let mut n_new_intervals = 0;
        let mut n_conflicts = 0;

        // Check the travel times
        {
            let _p = hprof::enter("mip_ddd check travel");
            for (train_idx, train) in problem.trains.iter().enumerate() {
                for visit_idx in 0..train.visits.len() - 1 {
                    let dt = problem.trains[train_idx].visits[visit_idx].travel_time;
                    if solution[train_idx][visit_idx] + dt > solution[train_idx][visit_idx + 1] {
                        panic!(
                            "travel time refinement t{}v{}x{}",
                            train_idx,
                            visit_idx + 1,
                            solution[train_idx][visit_idx] + dt
                        );
                        // Because we always "propagate" the new shortest traveling time to subsequent
                        // train resource occupations, we should never have this conflict type.

                        // If we don't eagerly propagate like this, then we need this refinement:
                        // new_intervals.push_back((
                        //     train_idx,
                        //     visit_idx + 1,
                        //     solution[train_idx][visit_idx] + dt,
                        // ));
                    }
                }
            }
        }

        const ADD_ALL_CONFLICTS: bool = false;

        if ADD_ALL_CONFLICTS {
            // Check the conflicts
            {
                let _p = hprof::enter("mip_ddd check conflicts");
                for visit_pair @ ((t1, v1), (t2, v2)) in visit_conflicts.iter().copied() {
                    if !check_conflict(visit_pair, &solution)? {
                        // warn!("conflict refinement t{}v{}x{}--{} vee t{}v{}x{}--{}",
                        // t1, v1, solution[t1][v1], solution[t1][v1+1],
                        // t2, v2, solution[t2][v2], solution[t2][v2+1]);
                        // warn!("   t1-v1   intervals {:?}", intervals[t1][v1]);
                        // warn!("   t1-v1+1 intervals {:?}", intervals[t1][v1+1]);
                        // warn!("   t2-v2   intervals {:?}", intervals[t2][v2]);
                        // warn!("   t2-v2+1 intervals {:?}", intervals[t2][v2+1]);
                        new_intervals.push_back((t2, v2, solution[t1][v1 + 1]));
                        new_intervals.push_back((t1, v1, solution[t2][v2 + 1]));
                        n_conflicts += 1;
                    }
                }
            }
        } else {
            // Check the conflicts
            let violated_conflicts = {
                let _p = hprof::enter("check conflicts");

                let mut cs: HashMap<(usize, usize), Vec<(usize, usize)>> = HashMap::new();
                for visit_pair @ ((t1, v1), (t2, v2)) in visit_conflicts.iter().copied() {
                    if !check_conflict(visit_pair, &solution)? {
                        cs.entry((t1, t2)).or_default().push((v1, v2));
                    }
                }
                cs
            };

            for ((t1, t2), pairs) in violated_conflicts {
                let (v1, v2) = *pairs.iter().min_by_key(|(v1, v2)| v1 + v2).unwrap();
                new_intervals.push_back((t2, v2, solution[t1][v1 + 1]));
                new_intervals.push_back((t1, v1, solution[t2][v2 + 1]));
                n_conflicts += 1;
            }
        }

        if !new_intervals.is_empty() {
            while let Some((train_idx, visit_idx, time)) = new_intervals.pop_front() {
                let is = &mut intervals[train_idx][visit_idx];
                // println!(
                //     "INSERTING t{}v{}time{} is{:?} idx{:?}",
                //     train_idx,
                //     visit_idx,
                //     time,
                //     is,
                //     is.binary_search_by_key(&time, |(t, _)| *t)
                // );
                match is.binary_search_by_key(&time, |(t, _)| *t) {
                    Ok(_) => continue,
                    Err(idx) if idx == 0 => continue,
                    Err(idx) => {
                        assert!(idx > 0);
                        let old_end = is[idx - 1].1;
                        is[idx - 1].1 = time;
                        is.insert(idx, (time, old_end));
                        n_new_intervals += 1;

                        if visit_idx + 1 < problem.trains[train_idx].visits.len() {
                            let next_time =
                                time + problem.trains[train_idx].visits[visit_idx].travel_time;
                            new_intervals.push_back((train_idx, visit_idx + 1, next_time));
                        }
                    }
                };
            }
            info!(
                "Added {} new intervals to solve {} conflicts",
                n_new_intervals, n_conflicts
            );
        } else {
            // success
            let n_intervals = intervals
                .iter()
                .flat_map(|t| t.iter().map(|v| v.len()))
                .sum::<usize>();
            println!("Solved with {} intervals", n_intervals);

            output_stats("iteration".to_string(), iteration.into());
            output_stats(
                "intervals".to_string(),
                t_vars
                    .iter()
                    .map(|is| is.iter().map(|i| i.len()).sum::<usize>())
                    .sum::<usize>()
                    .into(),
            );
            output_stats(
                "travel_constraints".to_string(),
                n_travel_time_constraints.into(),
            );
            output_stats(
                "resource_constraints".to_string(),
                n_conflict_constraints.into(),
            );
            // output_stats("internal_cost".to_string(), cost.into());

            output_stats(
                "total_time".to_string(),
                start_time.elapsed().as_secs_f64().into(),
            );
            output_stats("solver_time".to_string(), solver_time.as_secs_f64().into());
            output_stats(
                "algorithm_time".to_string(),
                (start_time.elapsed().as_secs_f64() - solver_time.as_secs_f64()).into(),
            );

            return Ok((solution, SolveStats::default()));
        }
    }
}

fn check_conflict(
    ((t1, v1), (t2, v2)): ((usize, usize), (usize, usize)),
    solution: &[Vec<i32>],
) -> Result<bool, SolverError> {
    let t_t1v1 = solution[t1][v1];
    let t_t1v1next = solution[t1][v1 + 1];
    let t_t2v2 = solution[t2][v2];
    let t_t2v2next = solution[t2][v2 + 1];

    let separation = (t_t2v2 - t_t1v1next).max(t_t1v1 - t_t2v2next);

    // let t1_first = t_t1v1next <= t_t2v2;
    // let t2_first = t_t2v2next <= t_t1v1;
    let has_separation = separation >= 0;
    // println!("t{}v{} {}-{}  t{}v{} {}-{}  separation {} is_separated {}", t1, v1, t_t1v1, t_t1v1next, t2, v2, t_t2v2, t_t2v2next, separation, has_separation);
    // assert!((separation >= 1e-5) == has_separation);

    if has_separation {
        Ok(true)
    } else {
        // println!("conflict separation {:?}  {}", ((t1, v1), (t2, v2)), separation);

        Ok(false)
    }
}
