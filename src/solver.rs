use std::collections::HashMap;

use crate::problem::Problem;
use rust_lapper::Lapper;
use satcoder::{
    prelude::SymbolicModel,
    solvers::minisat::{self, Bool},
    SatInstance, SatSolver, SatSolverWithCore,
};

pub fn solve(problem: &Problem) -> Result<Vec<Vec<i32>>, ()> {
    let mut solver = minisat::Solver::new();

    let mut train_occ = HashMap::new();

    let mut resource_occ: Vec<Lapper<u32, (usize,usize)>> = vec![Lapper::new(vec![]); problem.resources.len()];
    let mut touched_times = Vec::new();

    let mut conflicts: HashMap<usize, Vec<usize>> = HashMap::new();
    for (a, b) in problem.conflicts.iter() {
        conflicts.entry(*a).or_default().push(*b);
        conflicts.entry(*b).or_default().push(*a);
    }

    for (train_idx, train) in problem.trains.iter().enumerate() {
        for (visit_idx, (t, _resource)) in train.path.iter().enumerate() {
            train_occ.insert(
                (train_idx, visit_idx),
                Occ {
                    delays: vec![(true.into(), *t), (false.into(), i32::MAX)],
                    incumbent: 0,
                },
            );

            touched_times.push((train_idx, visit_idx));
        }
    }

    for (train_idx, train) in problem.trains.iter().enumerate() {
        for (visit_idx, (t_in, resource)) in train.path.iter().enumerate() {
            if visit_idx + 1 < train.path.len() {
                let travel_time = problem.resources[*resource].travel_time;
                let t_out = train.path[visit_idx + 1].0;
                // let insert_idx = resource_occ[*resource].partition_point(|(start, _, _, _)| start < t_in);
                // assert!(t_in + travel_time <= t_out);
                // resource_occ[*resource].insert(insert_idx, (*t_in, t_out, train_idx, visit_idx));
            }
        }
    }

    let mut new_soft_constraints = Vec::new();

    for (train_idx, train) in problem.trains.iter().enumerate() {
        for (visit_idx, (t_in, resource)) in train.path.iter().enumerate() {
            println!(
                "t{} v{} r{} {:?}",
                train_idx,
                visit_idx,
                resource,
                train_occ[&(train_idx, visit_idx)]
            );
        }
    }
    for (r, res) in resource_occ.iter().enumerate() {
        println!("r{} {:?}", r, res);
    }
    // println!("Initial train_occ {:#?}", train_occ);
    // println!("Initial resource_occ {:#?}", resource_occ);

    let mut iteration = 0;
    loop {
        println!("Iteration {} conflict detection starting...", iteration);
        let mut found_conflict = false;
        for (train_idx, visit_idx) in touched_times.drain(..) {
            let next_visit_idx = visit_idx + 1;

            // GATHER INFORMATION ABOUT TWO CONSECUTIVE TIME POINTS
            let this_occ = &train_occ[&(train_idx, visit_idx)];
            let t1_in = this_occ.incumbent_time();

            let this_resource_id = problem.trains[train_idx].path[visit_idx].1;
            let travel_time = problem.resources[this_resource_id].travel_time;

            let t1_out = train_occ
                .get(&(train_idx, next_visit_idx))
                .map(|o| o.incumbent_time())
                .unwrap_or_else(|| t1_in + travel_time);

            println!(
                "Checking t{}-v{}-r{} @ t={}--{}",
                train_idx, visit_idx, this_resource_id, t1_in, t1_out
            );

            if next_visit_idx < problem.trains[train_idx].path.len() {
                let next_occ = &train_occ[&(train_idx, next_visit_idx)];

                // TRAVEL TIME CONFLICT
                if t1_in + travel_time > t1_out {
                    found_conflict = true;
                    println!(
                        "  - TRAVEL time conflict train{} visit{} resource{} in{} travel{} out{}",
                        train_idx, visit_idx, this_resource_id, t1_in, travel_time, t1_out
                    );

                    // Insert the new time point.
                    let t1_in_var = this_occ.delays[this_occ.incumbent].0;
                    let new_t = this_occ.incumbent_time() + travel_time;
                    let new_timepoint_idx = next_occ.incumbent + 1;
                    let (t1_earliest_out_var, t1_is_new) = train_occ
                        .get_mut(&(train_idx, next_visit_idx))
                        .unwrap()
                        .time_point(&mut solver, new_timepoint_idx, new_t);

                    // T1_IN delay implies T1_EARLIEST_OUT delay.
                    SatInstance::add_clause(&mut solver, vec![!t1_in_var, t1_earliest_out_var]);

                    // The new timepoint might have a cost.
                    if t1_is_new {
                        let cost = problem.trains[train_idx].delay_cost(visit_idx, new_t);
                        if cost > 0 {
                            // Add soft.
                            new_soft_constraints.push((t1_earliest_out_var, cost));
                            // TODO
                        }
                    }
                }
            }
            // RESOURCE CONFLICT

            if let Some(conflicting_resources) = conflicts.get(&this_resource_id) {
                for other_resource in conflicting_resources.iter().copied() {
                    let this_resource_occ = &resource_occ[other_resource];

                    // // TODO correctly determine the relevant conflict index interval
                    // let conflict_start_idx =
                    //     this_resource_occ.partition_point(|(t, _, _, _)| *t < t1_in /*wrong partition */);
                    // let conflict_end_idx = this_resource_occ.partition_point(|(t, _, _, _)| *t < t1_out);

                    // let conflicting_occs = &this_resource_occ[conflict_start_idx..conflict_end_idx];
                    let conflicting_occs = &this_resource_occ[..];

                    for (t2_in, t2_out, other_train, other_visit) in conflicting_occs.iter().copied() {
                        // We have a train2 that is conflicting.
                        if other_train == train_idx {
                            continue; // Assume for now that the train doesn't conflict with itself.
                        }

                        if t1_out <= t2_in || t2_out <= t1_in {
                            continue;
                        }

                        found_conflict = true;

                        println!(
                            " - RESOURCE conflict between t{}-v{}-r{}-in{}-out{} t{}-v{}-r{}-in{}-out{}",
                            train_idx,
                            visit_idx,
                            problem.trains[train_idx].path[visit_idx].1,
                            t1_in,
                            t1_out,
                            other_train,
                            other_visit,
                            problem.trains[other_train].path[other_visit].1,
                            t2_in,
                            t2_out,
                        );

                        // We should not need to use these.
                        #[allow(unused)]
                        let t1_in = ();
                        #[allow(unused)]
                        let t2_in = ();

                        // The constraint is:
                        // We can delay T1_IN until T2_OUT?
                        let t1occ = &train_occ[&(train_idx, visit_idx)];
                        let new_timepoint_idx = t1occ.incumbent + 1;
                        let new_t = t2_out;
                        let (delay_t1, t1_is_new) = train_occ.get_mut(&(train_idx, visit_idx)).unwrap().time_point(
                            &mut solver,
                            new_timepoint_idx,
                            new_t,
                        );

                        // .. OR we can delay T2_IN until T1_OUT
                        let t2occ = &train_occ[&(other_train, other_visit)];
                        let new_timepoint_idx = t2occ.incumbent + 1;
                        let new_t = t1_out;
                        let (delay_t2, t2_is_new) = train_occ.get_mut(&(other_train, other_visit)).unwrap().time_point(
                            &mut solver,
                            new_timepoint_idx,
                            new_t,
                        );

                        let t1occ = &train_occ[&(train_idx, visit_idx)];
                        let t2occ = &train_occ[&(other_train, other_visit)];

                        const USE_CHOICE_VAR: bool = false;

                        if USE_CHOICE_VAR {
                            let choose = SatInstance::new_var(&mut solver);
                            SatInstance::add_clause(
                                &mut solver,
                                vec![!choose, !t1occ.delays[t1occ.incumbent].0, delay_t2],
                            );
                            SatInstance::add_clause(
                                &mut solver,
                                vec![choose, !t2occ.delays[t1occ.incumbent].0, delay_t1],
                            );
                        } else {
                            SatInstance::add_clause(
                                &mut solver,
                                vec![
                                    !t1occ.delays[t1occ.incumbent].0,
                                    !t2occ.delays[t1occ.incumbent].0,
                                    delay_t1,
                                    delay_t2,
                                ],
                            );
                        }
                    }
                }
            }
        }

        assert!(touched_times.is_empty());

        if !found_conflict {
            // Incumbent times are feasible and optimal.

            let mut trains = Vec::new();
            for (train_idx, train) in problem.trains.iter().enumerate() {
                let mut train_times = Vec::new();
                for (visit_idx, (t_in, resource)) in train.path.iter().enumerate() {
                    train_times.push(train_occ[&(train_idx, visit_idx)].incumbent_time());
                }

                let last_resource = problem.trains[train_idx].path[train_times.len() - 1].1;
                let last_t = train_times[train_times.len() - 1] + problem.resources[last_resource].travel_time;
                train_times.push(last_t);

                trains.push(train_times);
            }

            return Ok(trains);
        }

        let core = {
            let result = SatSolverWithCore::solve_with_assumptions(&mut solver, std::iter::empty());
            match result {
                satcoder::SatResultWithCore::Sat(model) => {
                    for train_idx in 0..problem.trains.len() {
                        for visit_idx in 0..problem.trains[train_idx].path.len() {
                            let this_occ = train_occ.get_mut(&(train_idx, visit_idx)).unwrap();
                            let old_time = this_occ.incumbent_time();
                            let mut touched = false;

                            while model.value(&this_occ.delays[this_occ.incumbent + 1].0) {
                                this_occ.incumbent += 1;
                                touched = true;
                            }
                            while !model.value(&this_occ.delays[this_occ.incumbent].0) {
                                this_occ.incumbent -= 1;
                                touched = true;
                            }

                            if touched {
                                let resource = problem.trains[train_idx].path[visit_idx].1;
                                let new_time = this_occ.incumbent_time();
                                println!(
                                    "Updated t{}-v{}-r{}  t={}-->{}",
                                    train_idx, visit_idx, resource, old_time, new_time
                                );

                                resource_occ[resource].retain(|(_, _, t, _)| *t != train_idx);
                                resource_occ[resource].push((*new_time, train_idx, visit_idx));

                                if visit_idx > 0 && touched_times.last() != Some(&(train_idx, visit_idx - 1)) {
                                    touched_times.push((train_idx, visit_idx - 1));
                                }
                                touched_times.push((train_idx, visit_idx));
                            }
                        }
                    }

                    None
                }
                satcoder::SatResultWithCore::Unsat(core) => {
                    if core.is_empty() {
                        return Err(()); // UNSAT
                    }
                    Some(core)
                }
            }
        };

        if let Some(core) = core {
            // Do weighted RC2
            todo!()
        }

        iteration += 1;
    }
}

#[derive(Debug)]
struct Occ {
    delays: Vec<(Bool, i32)>,
    incumbent: usize,
}

impl Occ {
    pub fn incumbent_time(&self) -> i32 {
        self.delays[self.incumbent].1
    }

    pub fn time_point(&mut self, solver: &mut impl SatInstance<minisat::Lit>, idx: usize, t: i32) -> (Bool, bool) {
        // The inserted time should be between the neighboring times.
        // assert!(idx == 0 || self.delays[idx - 1].1 < t);
        // assert!(idx == self.delays.len() || self.delays[idx + 1].1 > t);

        assert!(idx > 0); // cannot insert before the earliest time.
        assert!(idx < self.delays.len()); // cannot insert after infinity.

        println!("idx {} t {}   delays{:?}", idx, t, self.delays);

        assert!(self.delays[idx - 1].1 < t);
        assert!(self.delays[idx].1 >= t);

        if self.delays[idx].1 == t {
            return (self.delays[idx].0, false);
        }

        let var = solver.new_var();
        self.delays.insert(idx, (var, t));

        if idx > 0 {
            solver.add_clause(vec![!var, self.delays[idx - 1].0]);
        }

        if idx < self.delays.len() {
            solver.add_clause(vec![!self.delays[idx + 1].0, var]);
        }

        (var, true)
    }
}
