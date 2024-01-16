use std::collections::{HashMap, VecDeque};

use log::info;
use typed_index_collections::TiVec;

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
    propagate_traveltime_discretization :bool,
    mut output_stats: impl FnMut(String, serde_json::Value),
) -> Result<(Vec<Vec<i32>>, SolveStats), SolverError> {
    let _p = hprof::enter("ipamir incremental");
    let _p_init = hprof::enter("init");
    type Lit = isize;
    let start_time = std::time::Instant::now();

    let mut s = mk_solver();
    let mut resource_visits: Vec<Vec<(usize, usize)>> = Vec::new();
    let mut new_time_points = VecDeque::new();

    let mut n_new_intervals = 0;
    let mut n_traveltime = 0;
    let mut n_conflicts = 0;
    let mut n_timepoints = 0;

    let mut resource_usage: HashMap<usize, Vec<(usize, usize)>> = HashMap::new();
    for (train_idx, train) in problem.trains.iter().enumerate() {
        for (visit_idx, visit) in train.visits.iter().enumerate() {
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

    #[derive(Default)]
    struct CostVars {
        vars: Vec<Lit>,
    }

    impl CostVars {
        fn set_soft_lit(&mut self, solver: &mut impl MaxSatSolver, l: Lit, cost: u32) {
            if cost == 0 {
                return;
            }

            while self.vars.len() < cost as usize {
                let cost_var = solver.new_var();
                if self.vars.len() > 0 {
                    solver.add_clause(None, vec![-cost_var, self.vars[self.vars.len() - 1]]);
                }
                self.vars.push(cost_var);
                solver.add_clause(Some(1), vec![-cost_var]);
            }

            solver.add_clause(None, vec![l, self.vars[cost as usize - 1]]);
        }
    }

    struct Occ {
        time: Vec<TimePoint>,
        cost_vars: CostVars,
        next_lit: Lit,
    }

    struct TimePoint {
        start: i32,
        end: i32,
        lit: Lit,
        incompat: Vec<((usize, usize), usize)>,
    }

    let mut occupations: Vec<Vec<Occ>> = problem
        .trains
        .iter()
        .enumerate()
        .map(|(train_idx, train)| {
            train
                .visits
                .iter()
                .enumerate()
                .map(|(visit_idx, visit)| {
                    let v1 = s.new_var();
                    let v2 = s.new_var();
                    let cost = train.visit_delay_cost(delay_cost_type, visit_idx, visit.earliest);
                    let mut cost_vars: CostVars = Default::default();
                    cost_vars.set_soft_lit(&mut s, -v1, cost as u32);
                    s.add_clause(None, vec![v1, v2]);

                    while resource_visits.len() <= visit.resource_id {
                        resource_visits.push(Vec::new());
                    }

                    resource_visits[visit.resource_id].push((train_idx, visit_idx));

                    n_timepoints += 1;
                    Occ {
                        time: vec![TimePoint {
                            start: visit.earliest,
                            end: i32::MAX,
                            lit: v1,
                            incompat: Default::default(),
                        }],
                        cost_vars,
                        next_lit: v2,
                    }
                })
                .collect()
        })
        .collect();

    let visit_conflicts = super::bigm::visit_conflicts(problem);
    let visit_conflicts_map: Vec<Vec<Vec<(usize, usize)>>> = {
        let mut visit_conflicts_map: Vec<Vec<Vec<(usize, usize)>>> = (0..problem.trains.len())
            .map(|t| {
                (0..problem.trains[t].visits.len())
                    .map(|_| Default::default())
                    .collect()
            })
            .collect();

        for ((t1, v1), (t2, v2)) in visit_conflicts.iter() {
            visit_conflicts_map[*t1][*v1].push((*t2, *v2));
            visit_conflicts_map[*t2][*v2].push((*t1, *v1));
        }
        visit_conflicts_map
    };

    let occ_idxs: Vec<(usize, usize)> = problem
        .trains
        .iter()
        .enumerate()
        .flat_map(|(train_idx, train)| {
            (0..train.visits.len()).map(move |visit_idx| (train_idx, visit_idx))
        })
        .collect();

    type TPRef = ((usize, usize), usize);
    let mut potentially_new_incompatibilities: Vec<(TPRef, TPRef)> = Vec::new();
    let mut iteration = 0;
    drop(_p_init);
    loop {
        let _p_init = hprof::enter("iteration");
        iteration += 1;

        if start_time.elapsed().as_secs_f64() > timeout {
            return Err(SolverError::Timeout);
        }
        let _p_solve = hprof::enter("solve");
        info!("Iteration {} {}", iteration, s.status());

        for train_idx in 0..problem.trains.len() {
            let x = occupations[train_idx]
                .iter()
                .enumerate()
                .map(|(visit_idx, v)| {
                    v.time
                        .iter()
                        .map(|tp| (tp.start, tp.end))
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>();

            // println!("tr {} {:?}", train_idx, x);
        }

        let assumptions = occupations
            .iter()
            .flat_map(|t| t.iter().map(|o| -o.next_lit));

        log::info!(
            "solving it{} with {} timepoints {} conflicts",
            iteration,
            n_timepoints,
            n_conflicts
        );
        let (cost, solution) = s
            .optimize(
                Some(timeout - start_time.elapsed().as_secs_f64()),
                assumptions,
            )
            .map_err(|e| match e {
                crate::maxsatsolver::MaxSatError::NoSolution => SolverError::NoSolution,
                crate::maxsatsolver::MaxSatError::Timeout => SolverError::Timeout,
            })?;

        info!("Solved with cost {}", cost);

        drop(_p_solve);
        let _p_analyse = hprof::enter("analyse");

        let mut selected_interval: Vec<Vec<usize>> = occupations
            .iter()
            .map(|t_occ| {
                t_occ
                    .iter()
                    .map(|occ| {
                        assert!(
                            occ.time
                                .iter()
                                .filter(|tp| solution[tp.lit as usize - 1])
                                .count()
                                == 1
                        );

                        occ.time
                            .iter()
                            .position(|tp| solution[tp.lit as usize - 1])
                            .unwrap()
                    })
                    .collect()
            })
            .collect();

        // LOCALLY MINIMIZE
        // iteratively select an earlier interval if that preserves feasibility

        let mut consecutive_unmodified_occupations = 0;
        let mut i = 0;
        let occs_len = occ_idxs.len();
        while consecutive_unmodified_occupations < occs_len {
            let mut touched = false;
            let (train_idx, visit_idx) = occ_idxs[i % occupations.len()];
            while selected_interval[train_idx][visit_idx] > 0 {
                let prev_interval = selected_interval[train_idx][visit_idx] - 1;

                let blocked = occupations[train_idx][visit_idx].time[prev_interval]
                    .incompat
                    .iter()
                    .any(|((t2, v2), i2)| selected_interval[*t2][*v2] == *i2);

                if !blocked {
                    selected_interval[train_idx][visit_idx] -= 1;
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
            .map(|(train_idx, ts)| {
                let mut train_times: Vec<i32> = ts
                    .iter()
                    .enumerate()
                    .map(|(visit_idx, occ)| occupations[train_idx][visit_idx].time[*occ].start)
                    .collect();

                let last = train_times.last().copied().unwrap()
                    + problem.trains[train_idx].visits.last().unwrap().travel_time;

                // for visit_idx in 0..train_times.len() {
                //     if problem.trains[train_idx].visits[visit_idx].resource_id == 0 && visit_idx > 0
                //     {
                //         train_times[visit_idx] = train_times[visit_idx - 1]
                //             + problem.trains[train_idx].visits[visit_idx - 1].travel_time;
                //     }
                // }

                train_times.push(last);
                train_times
            })
            .collect();

        // CHECK TRAVEL TIME CONFLICTS

        for (train_idx, train) in problem.trains.iter().enumerate() {
            for visit_idx in 0..train.visits.len() - 1 {
                let dt = problem.trains[train_idx].visits[visit_idx].travel_time;
                if solution[train_idx][visit_idx] + dt > solution[train_idx][visit_idx + 1] {
                    if propagate_traveltime_discretization {
                        // Because we always "propagate" the new shortest traveling time to subsequent
                        // train resource occupations, we should never have this conflict type.

                        panic!(
                            "travel time refinement t{}v{}x{}",
                            train_idx,
                            visit_idx + 1,
                            solution[train_idx][visit_idx] + dt
                        );
                    } else {
                        // If we don't eagerly propagate like this, then we need this refinement:
                        // println!(
                        //     "travel time refinement t{}v{}-{} t0={} t1={} dt={}",
                        //     train_idx,
                        //     visit_idx,
                        //     visit_idx + 1,
                        //     solution[train_idx][visit_idx],
                        //     solution[train_idx][visit_idx + 1],
                        //     dt
                        // );
                        // println!(
                        //     "  {:?}",
                        //     occupations[train_idx][visit_idx]
                        //         .time
                        //         .iter()
                        //         .map(|t| (t.start, t.end))
                        //         .collect::<Vec<_>>()
                        // );
                        // println!(
                        //     "  {:?}",
                        //     occupations[train_idx][visit_idx + 1]
                        //         .time
                        //         .iter()
                        //         .map(|t| (t.start, t.end))
                        //         .collect::<Vec<_>>()
                        // );

                        let interval_exists = occupations[train_idx][visit_idx + 1]
                            .time
                            .iter()
                            .any(|tp| tp.start == solution[train_idx][visit_idx] + dt);

                        assert!(!interval_exists);

                        let xx = occupations[train_idx][visit_idx + 1].time[0].start
                            <= solution[train_idx][visit_idx] + dt;
                        if !xx {
                            // If we don't eagerly propagate like this, then we need this refinement:
                            println!(
                                "travel time refinement t{}v{}-{} t0={} t1={} dt={}",
                                train_idx,
                                visit_idx,
                                visit_idx + 1,
                                solution[train_idx][visit_idx],
                                solution[train_idx][visit_idx + 1],
                                dt
                            );
                            println!(
                                "  {:?}",
                                occupations[train_idx][visit_idx]
                                    .time
                                    .iter()
                                    .map(|t| (t.start, t.end))
                                    .collect::<Vec<_>>()
                            );
                            println!(
                                "  {:?}",
                                occupations[train_idx][visit_idx + 1]
                                    .time
                                    .iter()
                                    .map(|t| (t.start, t.end))
                                    .collect::<Vec<_>>()
                            );
                            panic!();
                        }

                        new_time_points.push_back((
                            train_idx,
                            visit_idx + 1,
                            solution[train_idx][visit_idx] + dt,
                            "tt",
                        ));
                        n_traveltime += 1;
                    }
                }
            }
        }

        // CHECK RESOURCE CONFLICTS

        // Conflict constraints
        for (_res, vs) in resource_usage.iter() {
            for (train_idx1, visit_idx1) in vs.iter() {
                for (train_idx2, visit_idx2) in vs.iter() {
                    if train_idx1 == train_idx2 {
                        assert!(visit_idx1 == visit_idx2);
                        continue;
                    }
                    if train_idx1 > train_idx2 {
                        continue;
                    }

                    // let rel = *train_idx1 == 2 && *visit_idx1 == 11 && *train_idx2 == 4;
                    // if rel {
                    //     println!(
                    //         "Checking {} {} {} {}",
                    //         train_idx1, visit_idx1, train_idx2, visit_idx2
                    //     );
                    // }

                    let travel1 = problem.trains[*train_idx1].visits[*visit_idx1].travel_time;
                    let travel2 = problem.trains[*train_idx2].visits[*visit_idx2].travel_time;

                    let tp1_idx = selected_interval[*train_idx1][*visit_idx1];
                    let tp2_idx = selected_interval[*train_idx2][*visit_idx2];
                    let tp1 = &occupations[*train_idx1][*visit_idx1].time[tp1_idx];
                    let tp2 = &occupations[*train_idx2][*visit_idx2].time[tp2_idx];

                    assert!(solution[*train_idx1][*visit_idx1] == tp1.start);
                    assert!(solution[*train_idx2][*visit_idx2] == tp2.start);

                    let separation =
                        (tp2.start - (tp1.start + travel1)).max(tp1.start - (tp2.start + travel2));
                    let has_separation = separation >= 0;
                    let incompatible = !has_separation;

                    // let test1 = tp2.start + travel2 >= tp1.end;
                    // let test2 = tp1.start + travel1 >= tp2.end;
                    // let incompatible = test1 && test2;
                    // if rel {
                    //     println!(
                    //         "Checking {} {} {} {} {} {} dt1={} dt2={} {}",
                    //         train_idx1,
                    //         visit_idx1,
                    //         train_idx2,
                    //         visit_idx2,
                    //         tp1.start,
                    //         tp2.start,
                    //         travel1,
                    //         travel2,
                    //         incompatible
                    //     );
                    // }

                    if incompatible {
                        new_time_points.push_back((
                            *train_idx2,
                            *visit_idx2,
                            tp1.start + travel1,
                            "res1",
                        ));
                        new_time_points.push_back((
                            *train_idx1,
                            *visit_idx1,
                            tp2.start + travel2,
                            "res2",
                        ));

                        // println!("incomp t{}v{}x{}-{},dt{}  t{}v{}x{}-{},dt{}", t1, v1, t1_start, t1_end, travel1, t2, v2, t2_start, t2_end, travel2);

                        assert!(
                            occupations[*train_idx1][*visit_idx1].time[0].start
                                < tp2.start + travel2
                        );
                        assert!(
                            occupations[*train_idx2][*visit_idx2].time[0].start
                                < tp1.start + travel1
                        );

                        // println!(
                        //     "SPLIT {} {} {}",
                        //     train_idx2,
                        //     visit_idx2,
                        //     tp1.start + travel1
                        // );
                        // println!(
                        //     "SPLIT {} {} {}",
                        //     train_idx1,
                        //     visit_idx1,
                        //     tp2.start + travel2
                        // );

                        // let i1 = interval_vars[*train_idx1][*visit_idx1][t1_idx];
                        // let i2 = interval_vars[*train_idx2][*visit_idx2][t2_idx];

                        // incompat
                        //     .entry((*train_idx1, *visit_idx1, t1_idx))
                        //     .or_default()
                        //     .push((*train_idx2, *visit_idx2, t2_idx));
                        // incompat
                        //     .entry((*train_idx2, *visit_idx2, t2_idx))
                        //     .or_default()
                        //     .push((*train_idx1, *visit_idx1, t1_idx));

                        // #[allow(clippy::useless_conversion)]
                        // model
                        //     .add_constr(
                        //         &format!(
                        //             "t{}v{}i{}--t{}v{}i{}",
                        //             train_idx1, visit_idx1, t1_idx, train_idx2, visit_idx2, t2_idx
                        //         ),
                        //         c!(i1 + i2 <= 1.0),
                        //     )
                        //     .unwrap();

                        n_conflicts += 1;
                    } else {
                        // println!("COMPATIBLE t{}v{}x{}-{},dt{}  t{}v{}x{}-{},dt{}", t1, v1, t1_start, t1_end, travel1, t2, v2, t2_start, t2_end, travel2);
                    }
                }
            }
        }

        // for visit_pair @ ((t1, v1), (t2, v2)) in visit_conflicts.iter().copied() {
        //     if !check_conflict(visit_pair, &solution)? {
        //         let travel1 = problem.trains[t1].visits[v1].travel_time;
        //         let travel2 = problem.trains[t2].visits[v2].travel_time;

        //         // assert!(solution[t1][v1+1] - solution[t1][v1] == travel1);
        //         // assert!(solution[t2][v2+1] - solution[t2][v2] == travel2);
        //         println!(
        //             "CONFLICT BETWEEN {}-{} dt={} {}-{} dt={}",
        //             t1, v1, travel1, t2, v2, travel2
        //         );
        //         println!(
        //             "  @{}-{} {:?}",
        //             solution[t1][v1],
        //             solution[t1][v1 + 1],
        //             occupations[t1][v1]
        //                 .time
        //                 .iter()
        //                 .map(|tp| (tp.start, tp.end))
        //                 .collect::<Vec<_>>()
        //         );
        //         println!(
        //             "  @{}-{} {:?}",
        //             solution[t2][v2],
        //             solution[t2][v2 + 1],
        //             occupations[t2][v2]
        //                 .time
        //                 .iter()
        //                 .map(|tp| (tp.start, tp.end))
        //                 .collect::<Vec<_>>()
        //         );

        //         assert!(!occupations[t1][v1]
        //             .time
        //             .iter()
        //             .any(|tp| tp.start == solution[t2][v2 + 1]));

        //         // TODO should we use minimum running time (`travel1`) instead of chosen running time (which has been relaxed)

        //         assert!(!occupations[t2][v2]
        //             .time
        //             .iter()
        //             .any(|tp| tp.start == solution[t1][v1 + 1]));

        //         new_time_points.push_back((t2, v2, solution[t1][v1 + 1]));
        //         new_time_points.push_back((t1, v1, solution[t2][v2 + 1]));
        //         n_conflicts += 1;
        //     }
        // }

        if new_time_points.is_empty() {
            println!("NO CONFLICTS");

            let solution: Vec<Vec<i32>> = selected_interval
                .iter()
                .enumerate()
                .map(|(train_idx, ts)| {
                    let mut train_times: Vec<i32> = ts
                        .iter()
                        .enumerate()
                        .map(|(visit_idx, occ)| occupations[train_idx][visit_idx].time[*occ].start)
                        .collect();

                    let last = train_times.last().copied().unwrap()
                        + problem.trains[train_idx].visits.last().unwrap().travel_time;

                    for visit_idx in 0..train_times.len() {
                        if problem.trains[train_idx].visits[visit_idx].resource_id == 0
                            && visit_idx > 0
                        {
                            train_times[visit_idx] = train_times[visit_idx - 1]
                                + problem.trains[train_idx].visits[visit_idx - 1].travel_time;
                        }
                    }

                    train_times.push(last);
                    train_times
                })
                .collect();

            return Ok((solution, Default::default()));
        }

        while let Some((train_idx, visit_idx, new_time, type_)) = new_time_points.pop_front() {
            if occupations[train_idx][visit_idx]
                .time
                .iter()
                .any(|tp| tp.start == new_time)
            {
                // panic!("skipping");
                continue;
            }

            // let relevant = train_idx == 4 && (visit_idx == 11 || visit_idx == 12);

            // if relevant {
            //     println!(" NEW TIME {} {} @{}", train_idx, visit_idx, new_time);
            // }
            // println!("SPLIT {} {} {}", train_idx, visit_idx, new_time);
            let (prev_idx, next_idx) = {
                let occ = &mut occupations[train_idx][visit_idx];
                // // if relevant {
                // println!(
                //     "   split at {} BEFORE {:?}",
                //     new_time,
                //     occ.time
                //         .iter()
                //         .map(|tp| (tp.start, tp.end))
                //         .collect::<Vec<_>>()
                // );
                // // }

                let mut prev_interval = (0, i32::MIN);
                for (i_idx, tp) in occ.time.iter().enumerate() {
                    if tp.start <= new_time && tp.start >= prev_interval.1 {
                        assert!(tp.start != new_time);
                        prev_interval = (i_idx, tp.start);
                    }
                }
                let prev_interval = prev_interval.0;
                let next_interval = occ.time.len();
                let end_time = occ.time[prev_interval].end;
                occ.time[prev_interval].end = new_time;

                let v1 = s.new_var();
                let cost = problem.trains[train_idx].visit_delay_cost(
                    delay_cost_type,
                    visit_idx,
                    new_time,
                );

                for t in occ.time.iter() {
                    s.at_most_one(&[v1, t.lit]);
                }

                let v2 = s.new_var();
                n_timepoints += 1;
                occ.time.push(TimePoint {
                    start: new_time,
                    end: end_time,
                    lit: v1,
                    incompat: Default::default(),
                });

                assert!(end_time >= new_time);
                let monotone = occ.time[prev_interval].end >= occ.time[prev_interval].start;
                if !monotone {
                    println!(
                        "   split {} at {} BEFORE {:?}",
                        type_,
                        new_time,
                        occ.time
                            .iter()
                            .map(|tp| (tp.start, tp.end))
                            .collect::<Vec<_>>()
                    );
                }
                assert!(monotone);

                s.add_clause(None, vec![-occ.next_lit, v1, v2]);
                occ.next_lit = v2;
                occ.cost_vars.set_soft_lit(&mut s, -v1, cost as u32);

                // if relevant {
                //     println!(
                //         "   AFTER {:?}",
                //         occ.time
                //             .iter()
                //             .map(|tp| (tp.start, tp.end))
                //             .collect::<Vec<_>>()
                //     );
                // }
                (prev_interval, next_interval)
            };

            assert!(potentially_new_incompatibilities.is_empty());

            // TRAVEL TIME CONFLICTS (WITH PREVIOUS VISIT)
            if visit_idx > 0 {
                for x in [prev_idx, next_idx] {
                    let next_tp = &occupations[train_idx][visit_idx].time[x];
                    let dt = problem.trains[train_idx].visits[visit_idx - 1].travel_time;
                    for (i2, prev_tp) in occupations[train_idx][visit_idx - 1]
                        .time
                        .iter()
                        .enumerate()
                    {
                        // if relevant {
                        //     println!("CHECK BEFORE {} {} {}", prev_tp.start, dt, next_tp.end);
                        // }
                        if prev_tp.start + dt >= next_tp.end {
                            // if relevant {
                            //     println!("  INCOPMAT");
                            // }
                            potentially_new_incompatibilities.push((
                                ((train_idx, visit_idx), x),
                                ((train_idx, visit_idx - 1), i2),
                            ));
                        }
                    }
                }
            }

            if visit_idx + 1 < problem.trains[train_idx].visits.len() {
                let dt = problem.trains[train_idx].visits[visit_idx].travel_time;

                if propagate_traveltime_discretization {
                    if new_time + dt >= occupations[train_idx][visit_idx + 1].time[0].start {
                        new_time_points.push_back((
                            train_idx,
                            visit_idx + 1,
                            new_time + dt,
                            "proptt",
                        ));
                    }
                }

                for x in [prev_idx, next_idx] {
                    let prev_tp = &occupations[train_idx][visit_idx].time[x];
                    for (i2, next_tp) in occupations[train_idx][visit_idx + 1]
                        .time
                        .iter()
                        .enumerate()
                    {
                        // if relevant {
                        //     println!(
                        //         "CHECK AFTER {}-{} {} {}-{}",
                        //         prev_tp.start, prev_tp.end, dt, next_tp.start, next_tp.end
                        //     );
                        // }
                        if prev_tp.start + dt >= next_tp.end {
                            // if relevant {
                            //     println!("  INCOPMAT");
                            // }

                            potentially_new_incompatibilities.push((
                                ((train_idx, visit_idx), x),
                                ((train_idx, visit_idx + 1), i2),
                            ));
                        }
                    }
                }
            }

            for (t2, v2) in visit_conflicts_map[train_idx][visit_idx].iter() {
                // if ((train_idx == 3 && visit_idx == 25) && (*t2 == 4 && *v2 == 11))
                //     || ((*t2 == 3 && *v2 == 25) && (train_idx == 4 && train_idx == 11))
                // {
                //     println!("checking t{}v{}", train_idx, visit_idx);
                // }

                assert!(train_idx != *t2);
                for occ1_idx in [prev_idx, next_idx] {
                    let occ1_start = occupations[train_idx][visit_idx].time[occ1_idx].start;
                    let occ1_end = occupations[train_idx][visit_idx].time[occ1_idx].end;
                    let occ1_travel = problem.trains[train_idx].visits[visit_idx].travel_time;

                    for occ2_idx in 0..occupations[*t2][*v2].time.len() {
                        let occ2_start = occupations[*t2][*v2].time[occ2_idx].start;
                        let occ2_end = occupations[*t2][*v2].time[occ2_idx].end;
                        let occ2_travel = problem.trains[*t2].visits[*v2].travel_time;

                        let test1 = occ2_start + occ2_travel >= occ1_end;
                        let test2 = occ1_start + occ1_travel >= occ2_end;
                        let incompatible = test1 && test2;

                        // if ((train_idx == 3 && visit_idx == 25) && (*t2 == 4 && *v2 == 11))
                        //     || ((*t2 == 3 && *v2 == 25) && (train_idx == 4 && train_idx == 11))
                        // {
                        //     println!(
                        //         "t{}v{} {}-{} dt={}",
                        //         train_idx, visit_idx, occ1_start, occ1_end, occ1_travel
                        //     );
                        //     println!(
                        //         " t{}v{} {}-{} dt={}",
                        //         t2, v2, occ2_start, occ2_end, occ2_travel
                        //     );
                        //     println!(" INCOMPAT = {}", incompatible);
                        // }

                        if incompatible {
                            potentially_new_incompatibilities
                                .push((((train_idx, visit_idx), occ1_idx), ((*t2, *v2), occ2_idx)));
                        }
                    }
                }
            }

            for (((t1, v1), o1), ((t2, v2), o2)) in potentially_new_incompatibilities.drain(..) {
                let already_incompat = occupations[t1][v1].time[o1]
                    .incompat
                    .iter()
                    .copied()
                    .any(|((tx, vx), ox)| tx == t2 && vx == v2 && ox == o2);

                if !already_incompat {
                    // println!(
                    //     "Adding incompatibility {}-{}-{}   {}-{}-{}",
                    //     t1, v1, o1, t2, v2, o2
                    // );
                    occupations[t1][v1].time[o1].incompat.push(((t2, v2), o2));
                    occupations[t2][v2].time[o2].incompat.push(((t1, v1), o1));
                    s.at_most_one(&[
                        occupations[t1][v1].time[o1].lit,
                        occupations[t2][v2].time[o2].lit,
                    ]);
                } else {
                    // println!(
                    //     "Already incompat {}-{}-{}   {}-{}-{}",
                    //     t1, v1, o1, t2, v2, o2
                    // );
                    assert!(occupations[t2][v2].time[o2]
                        .incompat
                        .iter()
                        .copied()
                        .any(|((tx, vx), ox)| tx == t1 && vx == v1 && ox == o1));
                }
            }
        }
    }
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
        let mut solver = mk_solver();

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
            "Solving maxsat ddd non-incremental {} iteration {} with {} travel time {} conflicts",
            solver.status(),
            iteration,
            n_travel_time_constraints,
            n_conflict_constraints
        );

        let bool_vec = solver
            .optimize(
                Some(timeout - start_time.elapsed().as_secs_f64()),
                std::iter::empty(),
            )
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
