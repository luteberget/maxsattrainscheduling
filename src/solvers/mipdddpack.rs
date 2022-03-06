//! MIP DDD with set packing constraints for intervals
//!
//!

use std::collections::{HashMap, VecDeque};

use grb::prelude::*;

use crate::problem::Problem;

use super::SolverError;
const M: f64 = 100_000.0;

pub fn solve(problem: &Problem) -> Result<Vec<Vec<i32>>, SolverError> {
    let _p_init = hprof::enter("mip solve init");
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

    let mut resource_usage : HashMap<usize, Vec<(usize,usize)>> = HashMap::new();
    for (train_idx, train) in problem.trains.iter().enumerate() {
        for (visit_idx, visit) in train.visits.iter().enumerate() {
            resource_usage.entry(visit.resource_id).or_default().push((train_idx, visit_idx));
        }
    }
    let visit_conflicts = super::bigm::visit_conflicts(problem);


    drop(_p_init);
    loop {
        let _p_enc = hprof::enter("build mip");
        let mut model = Model::new("model1").map_err(SolverError::GurobiError)?;

        let interval_vars = intervals.iter().enumerate().map(|(train_idx,t)| {
            t.iter().enumerate().map(|(visit_idx,v)| {

                let intervals = v.iter().map(|(time,time_out)| {
                    #[allow(clippy::unnecessary_cast)]
                    add_intvar!(model, 
                        name: &format!("t{}v{}-in{}_{}", train_idx, visit_idx, time, time_out),
                        bounds: 0..1.0_f64, 
                        obj: problem.trains[train_idx].visit_delay_cost(visit_idx, *time) as f64 + 0.01*(*time as f64))
                    .map_err(SolverError::GurobiError).unwrap()
                }).collect::<Vec<_>>();

                // exactly one of these must be chosen

                #[allow(clippy::useless_conversion)]
                model.add_constr(
                    &format!("t{}v{}ex", train_idx, visit_idx),
                c!( intervals.iter().grb_sum() == 1)).unwrap();

                intervals
            }).collect::<Vec<_>>()
        }).collect::<Vec<_>>();

        // travel time constraint
        for (train_idx, train) in problem.trains.iter().enumerate() {
            for visit_idx in 0..train.visits.len() - 1 {
                for (idx1, (time1,_end)) in intervals[train_idx][visit_idx].iter().enumerate() {
                    let dt = problem.trains[train_idx].visits[visit_idx].travel_time;
                    let source_interval = interval_vars[train_idx][visit_idx][idx1];
                    let incompatible_intervals = intervals[train_idx][visit_idx+1].iter().enumerate()
                    .take_while(|(_,(time2,_))| time1  + dt > *time2 ).map(|(idx2,_)| interval_vars[train_idx][visit_idx+1][idx2]);

                    #[allow(clippy::useless_conversion)]
                    model.add_constr(
                        &format!("t{}v{}i{}tt", train_idx, visit_idx, idx1),
                        c!( source_interval + incompatible_intervals.grb_sum() <= 1.0_f64 )).unwrap();
                }
            }
        }

        // Conflict constraints
        for (_res,vs) in resource_usage.iter() {
            for (t1,v1) in vs.iter() {
                for (t2,v2) in vs.iter() {
                    if t1 == t2 {
                        assert!(v1 == v2);
                        continue;
                    }
                    if t1 > t2 {
                        continue;
                    }

                    let travel1 = problem.trains[*t1].visits[*v1].travel_time;
                    let travel2 = problem.trains[*t2].visits[*v2].travel_time;
                    for (t1_idx,(t1_start,t1_end)) in intervals[*t1][*v1].iter().enumerate() {
                        for (t2_idx,(t2_start,t2_end)) in intervals[*t2][*v2].iter().enumerate() {
                            let test1 = *t1_end < *t2_start + travel2;
                            let test2 = *t2_end < *t1_start + travel1;
                            let incompatible = test1 && test2;
                            if incompatible {

                                let i1 = interval_vars[*t1][*v1][t1_idx];
                                let i2 = interval_vars[*t2][*v2][t2_idx];

                                // #[allow(clippy::useless_conversion)]
                                // model.add_constr(
                                //     &format!("t{}v{}i{} x t{}v{}i{}",
                                //             t1,v1,t1_idx,t2,v2,t2_idx),
                                //     c!( i1+i2 <= 1)).unwrap();
                            }
                        }
                    }
                }
            }
        }

        drop(_p_enc);

        {
        let _p = hprof::enter("mip solve");
        model.optimize().map_err(SolverError::GurobiError)?;
    }

        let _p = hprof::enter("mipddd refine");
        if model.status().map_err(SolverError::GurobiError)? != Status::Optimal {
            model.compute_iis().map_err(SolverError::GurobiError)?;
            model
                .write("infeasible.ilp")
                .map_err(SolverError::GurobiError)?;
            return Err(SolverError::NoSolution);
        }

        let cost = model
            .get_attr(attr::ObjVal)
            .map_err(SolverError::GurobiError)?;

        let solution = intervals.iter().enumerate().map(|(train_idx,t)| {
            let mut train_times = t.iter().enumerate().map(|(visit_idx,v)| {
                v.iter().enumerate().find_map(|(iidx, (t,_))| {
                    (model
                        .get_obj_attr(attr::X, &interval_vars[train_idx][visit_idx][iidx])
                        .map_err(SolverError::GurobiError).unwrap()
                        > 0.5).then(|| *t)
                }).unwrap()
            }).collect::<Vec<_>>();
            let last =                     train_times.last().copied().unwrap()
            + problem.trains[train_idx].visits.last().unwrap().travel_time;
            train_times.push(last);
            train_times
        }).collect::<Vec<_>>();

        // Check the conflicts
        let violated_conflicts = {
            let _p = hprof::enter("check conflicts");

            let mut cs = Vec::new();
            for visit_pair in visit_conflicts.iter().copied() {
                if !check_conflict(visit_pair, &solution)? {
                    cs.push(visit_pair);
                }
            }
            cs
        };

        if !violated_conflicts.is_empty() {
            let mut new_intervals = VecDeque::new();
            for ((t1,v1),(t2,v2)) in violated_conflicts {
                new_intervals.push_back((t2,v2,solution[t1][v1+1]));
                new_intervals.push_back((t1,v1,solution[t2][v2+1]));
            }

            while let Some((train_idx,visit_idx,time)) = new_intervals.pop_front() {
                let is = &mut intervals[train_idx][visit_idx];
                println!("INSERTING t{}v{}time{} is{:?} idx{:?}",
                    train_idx, visit_idx, time, is, is.binary_search_by_key(&time, |(t,_)| *t));
                match is.binary_search_by_key(&time, |(t,_)| *t) {
                    Ok(_) => continue,
                    Err(idx) if idx == 0 => continue,
                    Err(idx) => {
                        assert!(idx > 0);
                        let old_end = is[idx-1].1;
                        is[idx-1].1 = time;
                        is.insert(idx, (time, old_end));

                        if visit_idx + 1 < problem.trains[train_idx].visits.len() {
                            let next_time = time + problem.trains[train_idx].visits[visit_idx].travel_time;
                            new_intervals.push_back((train_idx, visit_idx+1, next_time));
                        }
                    }
                };
            }
        } else {
            // success
            let n_intervals = intervals.iter().flat_map(|t| t.iter().map(|v| v.len())).sum::<usize>();
            println!(
                "Solved with cost {} and {} intervals",
                cost,n_intervals
            );
            return Ok(solution);
        }
    }
}



fn check_conflict(
    ((t1, v1), (t2, v2)): ((usize, usize), (usize, usize)),
    solution :&[Vec<i32>],
) -> Result<bool, SolverError> {
    let t_t1v1 = solution[t1][v1];
    let t_t1v1next = solution[t1][v1+1];
    let t_t2v2 = solution[t2][v2];
    let t_t2v2next = solution[t2][v2+1];

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