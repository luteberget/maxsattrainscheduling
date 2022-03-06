//! MIP DDD with set packing constraints for intervals
//!
//!

use grb::prelude::*;

use crate::problem::Problem;

use super::SolverError;
const M: f64 = 100_000.0;

pub fn solve(problem: &Problem) -> Result<Vec<Vec<i32>>, SolverError> {
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

    loop {
        let mut model = Model::new("model1").map_err(SolverError::GurobiError)?;

        let interval_vars = intervals.iter().enumerate().map(|(train_idx,t)| {
            t.iter().enumerate().map(|(visit_idx,v)| {

                let intervals = v.iter().map(|(time,_)| {
                    #[allow(clippy::unnecessary_cast)]
                    add_intvar!(model, 
                        bounds: 0..1.0_f64, 
                        obj: problem.trains[train_idx].visit_delay_cost(visit_idx, *time) as f64)
                    .map_err(SolverError::GurobiError).unwrap()
                }).collect::<Vec<_>>();

                // exactly one of these must be chosen

                #[allow(clippy::useless_conversion)]
                model.add_constr("", c!( intervals.iter().grb_sum() == 1)).unwrap();

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
                    .take_while(|(_,(time2,_))| time1  + dt < *time2 ).map(|(idx2,_)| interval_vars[train_idx][visit_idx+1][idx2]);

                    #[allow(clippy::useless_conversion)]
                    model.add_constr("", c!( source_interval + incompatible_intervals.grb_sum() <= 1)).unwrap();
                }
            }
        }
    }
}
