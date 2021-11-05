use std::collections::HashMap;

use crate::problem::Problem;
use satcoder::{
    prelude::SymbolicModel,
    solvers::minisat::{self, Bool},
    SatInstance, SatSolverWithCore,
};
use typed_index_collections::TiVec;

#[derive(Clone, Copy)]
struct VisitId(u32);

impl From<VisitId> for usize {
    fn from(v: VisitId) -> Self {
        v.0 as usize
    }
}

impl From<usize> for VisitId {
    fn from(x: usize) -> Self {
        VisitId(x as u32)
    }
}

#[derive(Clone, Copy, Debug)]
struct ResourceId(u32);

impl From<ResourceId> for usize {
    fn from(v: ResourceId) -> Self {
        v.0 as usize
    }
}

impl From<usize> for ResourceId {
    fn from(x: usize) -> Self {
        ResourceId(x as u32)
    }
}

pub fn solve(problem: &Problem) -> Result<Vec<Vec<i32>>, ()> {
    let mut solver = minisat::Solver::new();
    let mut visits: TiVec<VisitId, (usize, usize)> = TiVec::new();
    let mut resource_visits: Vec<Vec<VisitId>> = Vec::new();
    let mut occupations: TiVec<VisitId, Occ> = TiVec::new();
    let mut touched_intervals = Vec::new();
    let mut conflicts: HashMap<_, Vec<_>> = HashMap::new();

    for (a, b) in problem.conflicts.iter() {
        conflicts.entry(*a).or_default().push(*b);
        conflicts.entry(*b).or_default().push(*a);
    }

    for (train_idx, train) in problem.trains.iter().enumerate() {
        for (visit_idx, (t, resource)) in train.path.iter().enumerate() {
            let visit: VisitId = visits.len().into();

            occupations.push(Occ {
                delays: vec![(true.into(), *t), (false.into(), i32::MAX)],
                incumbent: 0,
            });

            while resource_visits.len() <= *resource {
                resource_visits.push(Vec::new());
            }

            resource_visits[*resource].push(visit);
            touched_intervals.push(visit);

            visits.push((train_idx, visit_idx));
        }
    }

    let mut iteration = 0;
    let mut new_soft_constraints = Vec::new();

    loop {
        println!("Iteration {} conflict detection starting...", iteration);
        let mut found_conflict = false;
        for visit in touched_intervals.drain(..) {
            let (train_idx, visit_idx) = visits[visit];
            let next_visit: Option<VisitId> = if visit_idx + 1 < problem.trains[train_idx].path.len() {
                Some((usize::from(visit) + 1).into())
            } else {
                None
            };

            // GATHER INFORMATION ABOUT TWO CONSECUTIVE TIME POINTS
            let t1_in = occupations[visit].incumbent_time();

            let this_resource_id = problem.trains[train_idx].path[visit_idx].1;
            let travel_time = problem.resources[this_resource_id].travel_time;

            if let Some(next_visit) = next_visit {
                let v1 = &occupations[visit];
                let v2 = &occupations[next_visit];
                let t1_out = v2.incumbent_time();

                // TRAVEL TIME CONFLICT
                if t1_in + travel_time > t1_out {
                    found_conflict = true;
                    println!(
                        "  - TRAVEL time conflict train{} visit{} resource{} in{} travel{} out{}",
                        train_idx, visit_idx, this_resource_id, t1_in, travel_time, t1_out
                    );

                    // Insert the new time point.
                    let t1_in_var = v1.delays[v1.incumbent].0;
                    let new_t = v1.incumbent_time() + travel_time;
                    let new_timepoint_idx = v2.incumbent + 1;
                    let (t1_earliest_out_var, t1_is_new) =
                        occupations[next_visit].time_point(&mut solver, new_timepoint_idx, new_t);

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
                    let t1_out = next_visit
                        .map(|v| occupations[v].incumbent_time())
                        .unwrap_or(t1_in + travel_time);

                    for other_visit in resource_visits[other_resource].iter().copied() {
                        if usize::from(visit) == usize::from(other_visit) {
                            continue;
                        }
                        let v1 = &occupations[visit];
                        let v2 = &occupations[other_visit];
                        let t2_in = v2.incumbent_time();
                        let (other_train_idx, other_visit_idx) = visits[other_visit];

                        // We have a train2 that is conflicting.
                        if other_train_idx == train_idx {
                            continue; // Assume for now that the train doesn't conflict with itself.
                        }

                        let other_next_visit: Option<VisitId> =
                            if other_visit_idx + 1 < problem.trains[other_train_idx].path.len() {
                                Some((usize::from(other_visit) + 1).into())
                            } else {
                                None
                            };

                        let t2_out = other_next_visit
                            .map(|v| occupations[v].incumbent_time())
                            .unwrap_or_else(|| {
                                let other_resource_id = problem.trains[other_train_idx].path[other_visit_idx].1;
                                let travel_time = problem.resources[other_resource_id].travel_time;
                                t2_in + travel_time
                            });

                        // They are not overlapping so not in conflict.
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
                            other_train_idx,
                            other_visit_idx,
                            problem.trains[other_train_idx].path[other_visit_idx].1,
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
                        let new_idx1 = v1.incumbent + 1;
                        let new_t1 = t2_out;

                        // .. OR we can delay T2_IN until T1_OUT
                        let new_idx2 = v2.incumbent + 1;
                        let new_t2 = t1_out;

                        
                        let (delay_t2, t2_is_new) = occupations[other_visit].time_point(&mut solver, new_idx2, new_t2);
                        let (delay_t1, t1_is_new) = occupations[visit].time_point(&mut solver, new_idx1, new_t1);

                        let v1 = &occupations[visit];
                        let v2 = &occupations[other_visit];

                        const USE_CHOICE_VAR: bool = false;

                        if USE_CHOICE_VAR {
                            let choose = SatInstance::new_var(&mut solver);
                            SatInstance::add_clause(&mut solver, vec![!choose, !v1.delays[v1.incumbent].0, delay_t2]);
                            SatInstance::add_clause(&mut solver, vec![choose, !v2.delays[v2.incumbent].0, delay_t1]);
                        } else {
                            SatInstance::add_clause(
                                &mut solver,
                                vec![
                                    !v1.delays[v1.incumbent].0,
                                    !v2.delays[v2.incumbent].0,
                                    delay_t1,
                                    delay_t2,
                                ],
                            );
                        }
                    }
                }
            }
        }

        if !found_conflict {
            // Incumbent times are feasible and optimal.

            let mut trains = Vec::new();
            let mut i = 0;
            for (train_idx, train) in problem.trains.iter().enumerate() {
                let mut train_times = Vec::new();
                for _ in train.path.iter().enumerate() {
                    train_times.push(occupations[VisitId(i)].incumbent_time());
                    i += 1;
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
                    for (visit, this_occ) in occupations.iter_mut_enumerated() {
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
                            let (train_idx, visit_idx) = visits[visit];
                            let resource = problem.trains[train_idx].path[visit_idx].1;
                            let new_time = this_occ.incumbent_time();
                            println!(
                                "Updated t{}-v{}-r{}  t={}-->{}",
                                train_idx, visit_idx, resource, old_time, new_time
                            );

                            if visit_idx > 0 {
                                touched_intervals.push((Into::<usize>::into(visit) - 1).into());
                            }
                            touched_intervals.push(visit);
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
