#![allow(dead_code)]
use grb::add_binvar;

use super::SolverError;
use crate::{problem::Problem, solvers::bigm::visit_conflicts};

struct TimingVariable {
    intervals: Vec<(i32, grb::Var)>,
}

impl TimingVariable {
    pub fn new(model: &mut grb::Model, earliest: i32) -> Self {
        TimingVariable {
            intervals: vec![(earliest, add_binvar!(model).unwrap())],
        }
    }

    pub fn get_interval(&mut self, model: &mut grb::Model, t1: i32, t2: i32) -> grb::Var {
        
    }

    fn get_time_var(&mut self, model: &mut grb::Model, time: i32) -> grb::Var {
        match self.intervals.binary_search_by_key(&time, |(t, _)| *t) {
            Ok(idx) => self.intervals[idx].1,
            Err(idx) => {
                let new_var = add_binvar!(model).unwrap();
                self.intervals.insert(idx, (time, new_var));
                new_var
            }
        }
    }
}

pub fn solve(problem: &Problem) -> Result<Vec<Vec<i32>>, SolverError> {
    let _p = hprof::enter("mip-ddd solver");
    use grb::prelude::*;
    let mut model = Model::new("model1").map_err(SolverError::GurobiError)?;

    // The initial timing variables for all trains/vars
    let mut t_vars = problem
        .trains
        .iter()
        .map(|t| {
            t.visits
                .iter()
                .map(|v| TimingVariable::new(&mut model, v.earliest))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();

    // Incompatibles
    let mut incompatibles: Vec<()> = Vec::new();

    // TODO objective

    let mut solution = t_vars
        .iter()
        .map(|t| t.iter().map(|t| t.intervals[0].0).collect::<Vec<_>>())
        .collect::<Vec<_>>();

    let all_conflicts = visit_conflicts(problem);

    loop {
        model.optimize().unwrap();

        // Update solution
        for (t_idx, t) in t_vars.iter().enumerate() {
            for (v_idx, v) in t.iter().enumerate() {
                debug_assert!(check_partitioning_constraint(&model, v));

                let value = v
                    .intervals
                    .iter()
                    .find_map(|(val, var)| {
                        (model.get_obj_attr(attr::X, var).unwrap() > 0.5).then(|| *val)
                    })
                    .unwrap();

                solution[t_idx][v_idx] = value;
            }
        }

        //
        // Verify solution (or refine discretization)
        //

        // Check travel time constraints
        for t in 0..problem.trains.len() {
            for v in 0..problem.trains[t].visits.len() - 1 {
                let earliest_finish = solution[t][v] + problem.trains[t].visits[v].travel_time;

                if solution[t][v + 1] < earliest_finish {}
            }
        }

        // Check conflict constraints
        for ((t1, v1), (t2, v2)) in all_conflicts.iter().copied() {
            let (t1in, t1out) = (solution[t1][v1], solution[t1][v1 + 1]);
            let (t2in, t2out) = (solution[t2][v2], solution[t2][v2 + 1]);
            let ok = t1out <= t2in || t2out <= t1in;
            if !ok {
                // Delay T1
                let new_t1v1_time = t_vars[t1][v1].get_time(&mut model, t2out);
                let new_t2v2_time = t_vars[t2][v2].get_time(&mut model, t1out);
            }
        }
    }
}

fn check_partitioning_constraint(model: &grb::Model, v: &TimingVariable) -> bool {
    use grb::prelude::*;
    if !v
        .intervals
        .iter()
        .map(|(_val, var)| model.get_obj_attr(attr::X, var).unwrap())
        .all(|bin_var_value| bin_var_value >= 0. && !(0.01..=0.99).contains(&bin_var_value))
    {
        return false;
    }
    if v.intervals
        .iter()
        .filter(|(_val, var)| model.get_obj_attr(attr::X, var).unwrap() > 0.99)
        .count()
        != 1
    {
        return false;
    }

    true
}
