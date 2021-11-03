use std::collections::HashMap;

use crate::problem::Problem;
use satcoder::{
    prelude::SymbolicModel,
    solvers::minisat::{self, Bool},
    SatInstance, SatSolver, SatSolverWithCore,
};

pub fn solve(problem: &Problem) -> Result<(), ()> {
    let mut solver = minisat::Solver::new();

    let mut train_occ = HashMap::new();
    let mut resource_occ: Vec<Vec<(i32, i32, usize)>> = vec![Vec::new(); problem.resources.len()];
    let mut touched_times = Vec::new();

    let mut conflicts: HashMap<usize, Vec<usize>> = HashMap::new();
    for (a, b) in problem.conflicts.iter() {
        conflicts.entry(*a).or_default().push(*b);
        conflicts.entry(*b).or_default().push(*a);
    }

    for (train_idx, train) in problem.trains.iter().enumerate() {
        for (visit_idx, (t, resource)) in train.path.iter().enumerate() {
            train_occ.insert(
                (train_idx, visit_idx),
                Occ {
                    delays: vec![(true.into(), *t), (false.into(), i32::MAX)],
                    incumbent: 0,
                },
            );

            let travel_time = problem.resources[*resource].travel_time;
            let insert_idx = resource_occ[*resource].partition_point(|(start, _, _)| start < t);
            resource_occ[*resource].insert(insert_idx, (*t, t + travel_time /*wrong*/, train_idx));
            touched_times.push((train_idx, visit_idx));
        }
    }

    let mut new_soft_constraints = Vec::new();

    loop {
        let mut found_conflict = false;
        for (train_idx, visit_idx) in touched_times.drain(..) {
            let next_visit_idx = visit_idx + 1;

            if next_visit_idx < problem.trains[train_idx].path.len() {
                // GATHER INFORMATION ABOUT TWO CONSECUTIVE TIME POINTS
                let this_occ = &train_occ[&(train_idx, visit_idx)];
                let t1_in = this_occ.incumbent_time();

                let this_resource_id = problem.trains[train_idx].path[visit_idx].1;
                let travel_time = problem.resources[this_resource_id].travel_time;

                let next_occ = &train_occ[&(train_idx, next_visit_idx)];
                let t1_out = next_occ.incumbent_time();

                // TRAVEL TIME CONFLICT
                if t1_in + travel_time > t1_out {
                    found_conflict = true;

                    // Insert the new time point.
                    let t1_in_var = this_occ.delays[this_occ.incumbent].0;
                    let new_t = this_occ.incumbent_time() + travel_time;
                    let new_timepoint_idx = next_occ.incumbent + 1;
                    let t1_earliest_out_var = train_occ
                        .get_mut(&(train_idx, next_visit_idx))
                        .unwrap()
                        .add_time(&mut solver, new_timepoint_idx, new_t);

                    // T1_IN delay implies T1_EARLIEST_OUT delay.
                    SatInstance::add_clause(&mut solver, vec![!t1_in_var, t1_earliest_out_var]);

                    // The new timepoint might have a cost.
                    let cost = problem.trains[train_idx].delay_cost(visit_idx, new_t);
                    if cost > 0 {
                        // Add soft.
                        new_soft_constraints.push((t1_earliest_out_var, cost)); // TODO
                    }
                }

                // RESOURCE CONFLICT

                if let Some(conflicting_resources) = conflicts.get(&this_resource_id) {
                    for other_resource in conflicting_resources.iter().copied() {
                        let this_resource_occ = &resource_occ[other_resource];
                        let conflict_start_idx = this_resource_occ
                            .partition_point(|(t, _, _)| *t < t1_in /*wrong partition */);
                        let conflict_end_idx =
                            this_resource_occ.partition_point(|(t, _, _)| *t < t1_out);
                        let conflicting_occs =
                            &this_resource_occ[conflict_start_idx..conflict_end_idx];

                        for (t2in, t2out, train2) in conflicting_occs.iter().copied() {
                            // We have a train2 that is conflicting.
                            if train2 == train_idx {
                                continue; // Assume for now that the train doesn't conflict with itself.
                            }
                            let this_occ = &train_occ[&(train_idx, visit_idx)];

                            // The constraint is:
                            // We can delay T1_IN until T2_OUT?

                            let new_timepoint_idx = this_occ.incumbent + 1;
                            let new_t = t2out;
                            let delay_t1 = train_occ
                                .get_mut(&(train_idx, visit_idx))
                                .unwrap()
                                .add_time(&mut solver, new_timepoint_idx, new_t);


                            // .. OR we can delay T2_IN until T1_OUT
                            let new_timepoint_idx = this_occ.incumbent + 1;
                            let new_t = t2out;
                            let delay_t1 = train_occ
                                .get_mut(&(train2, visit_idx))
                                .unwrap()
                                .add_time(&mut solver, new_timepoint_idx, new_t);
                        }
                    }
                }
            }
        }

        touched_times.clear();

        if !found_conflict {
            // Incumbent times are feasible and optimal.
            return Ok(());
        }

        let core = {
            let result = SatSolverWithCore::solve_with_assumptions(&mut solver, std::iter::empty());
            match result {
                satcoder::SatResultWithCore::Sat(model) => {
                    for train_idx in 0..problem.trains.len() {
                        for visit_idx in 0..problem.trains[train_idx].path.len() {
                            if train_occ
                                .get_mut(&(train_idx, visit_idx))
                                .unwrap()
                                .update_chosen_delay(model.as_ref())
                            {
                                if visit_idx > 0
                                    && touched_times.last() != Some(&(train_idx, visit_idx - 1))
                                {
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
        }
    }
}

struct Occ {
    delays: Vec<(Bool, i32)>,
    incumbent: usize,
}

impl Occ {
    pub fn update_chosen_delay(
        &mut self,
        model: &dyn satcoder::SatModel<Lit = minisat::Lit>,
    ) -> bool {
        let mut touched = false;

        while !model.value(&self.delays[self.incumbent].0) {
            self.incumbent -= 1;
            touched = true;
        }
        while model.value(&self.delays[self.incumbent].0) {
            self.incumbent += 1;
            touched = true;
        }

        touched
    }

    pub fn incumbent_time(&self) -> i32 {
        self.delays[self.incumbent].1
    }

    pub fn add_time(
        &mut self,
        solver: &mut impl SatInstance<minisat::Lit>,
        idx: usize,
        t: i32,
    ) -> Bool {
        // The inserted time should be between the neighboring times.
        assert!(idx == 0 || self.delays[idx - 1].1 < t);
        assert!(idx == self.delays.len() || self.delays[idx + 1].1 > t);

        let var = solver.new_var();
        self.delays.insert(idx, (var, t));

        if idx > 0 {
            solver.add_clause(vec![!var, self.delays[idx - 1].0]);
        }

        if idx < self.delays.len() {
            solver.add_clause(vec![!self.delays[idx + 1].0, var]);
        }

        var
    }
}
