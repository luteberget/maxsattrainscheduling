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
            resource_occ[*resource].insert(insert_idx, (*t, t + travel_time, train_idx));
            touched_times.push((train_idx, visit_idx));
        }
    }

    let mut soft = Vec::new();

    loop {
        let mut found_conflict = false;
        for (train_idx, visit_idx) in touched_times.drain(..) {
            // Check if we could reach here from the previous visit
            if visit_idx > 0 {
                let prev_occ = &train_occ[&(train_idx, visit_idx - 1)];
                let this_occ = &train_occ[&(train_idx, visit_idx)];

                let prev_resource_id = problem.trains[train_idx].path[visit_idx].1;
                let travel_time = problem.resources[prev_resource_id].travel_time;

                if prev_occ.incumbent_time() + travel_time > this_occ.incumbent_time() {

                    found_conflict= true;

                    // Insert the new time point.
                    let idx = prev_occ.incumbent;
                    let t = this_occ.incumbent_time();
                    let var = train_occ
                        .get_mut(&(train_idx, visit_idx))
                        .unwrap()
                        .add_time(&mut solver, idx, t);

                    let cost = problem.trains[train_idx].delay_cost(visit_idx, t);
                    if cost > 0 {
                        // Add soft.
                        soft.push((var, cost)); // TODO
                    }
                }
            }
        }

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
                            if let Some(new_time) = train_occ
                                .get_mut(&(train_idx, visit_idx))
                                .unwrap()
                                .update_chosen_delay(model.as_ref())
                            {
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
    pub fn update_chosen_delay(&mut self, model: &dyn satcoder::SatModel<Lit = minisat::Lit>) -> Option<i32> {
        let mut touched = false;

        while !model.value(&self.delays[self.incumbent].0) {
            self.incumbent -= 1;
            touched = true;
        }
        while model.value(&self.delays[self.incumbent].0) {
            self.incumbent += 1;
            touched = true;
        }

        touched.then(|| self.incumbent_time())
    }

    pub fn incumbent_time(&self) -> i32 {
        self.delays[self.incumbent].1
    }

    pub fn add_time(&mut self, solver: &mut impl SatInstance<minisat::Lit>, idx: usize, t: i32) -> Bool {
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
