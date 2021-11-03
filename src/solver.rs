use std::collections::HashMap;

use crate::problem::Problem;
use satcoder::{
    prelude::SymbolicModel,
    solvers::minisat::{self, Bool},
    SatInstance, SatSolver, SatSolverWithCore,
};

pub fn solve(problem: &Problem) -> Result<(), ()> {
    let mut solver = minisat::Solver::new();

    struct Occ {
        vars: Vec<(Bool, i32)>,
        incumbent_chosen_delay: isize,
    }

    impl Occ {
        pub fn update_chosen_delay(
            &mut self,
            model: &dyn satcoder::SatModel<Lit = minisat::Lit>,
        ) -> Option<i32> {
            let mut touched = false;
            while !self.eval_time_idx(model, self.incumbent_chosen_delay) {
                self.incumbent_chosen_delay -= 1;
                touched = true;
            }
            while self.eval_time_idx(model, self.incumbent_chosen_delay + 1) {
                self.incumbent_chosen_delay += 1;
                touched = true;
            }

            touched.then(|| {
                if self.incumbent_chosen_delay == -1 {
                    0
                } else {
                    self.vars[self.incumbent_chosen_delay as usize].1
                }
            })
        }

        fn eval_time_idx(
            &mut self,
            model: &dyn satcoder::SatModel<Lit = minisat::Lit>,
            idx: isize,
        ) -> bool {
            if idx < 0 {
                return true;
            }
            let idx = idx as usize;

            if idx >= self.vars.len() {
                return false;
            }

            model.value(&self.vars[idx].0)
        }
    }

    let mut occ = HashMap::new();
    // let mut delay_lits :HashMap<minisat::Bool, (usize,usize,usize)> = HashMap::new();
    for (train_idx, train) in problem.trains.iter().enumerate() {
        for (visit_idx, (_, _)) in train.path.iter().enumerate() {
            occ.insert(
                (train_idx, visit_idx),
                Occ {
                    vars: Vec::new(),
                    incumbent_chosen_delay: -1,
                },
            );
        }
    }

    let mut soft = Vec::new();
    soft.push(());

    loop {
        let result = SatSolverWithCore::solve_with_assumptions(&mut solver, std::iter::empty());
        match result {
            satcoder::SatResultWithCore::Sat(model) => {
                for ((train, resource), occs) in occ.iter_mut() {
                    if let Some(new_time) = occs.update_chosen_delay(model.as_ref()) {}
                }
            }
            satcoder::SatResultWithCore::Unsat(core) => {}
        };
    }
}
