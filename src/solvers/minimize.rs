use crate::{
    problem::{DelayCostType, Problem},
    solvers::SolverError,
};

pub type VisitPair = ((usize, usize), (usize, usize));
pub fn minimize_solution(
    env: &grb::Env,
    problem: &Problem,
    // delay_cost_type: DelayCostType,
    priorities: Vec<VisitPair>,
    // train_names: &[String],
    // resource_names: &[String],
) -> Result<Vec<Vec<i32>>, SolverError> {
    let _p = hprof::enter("minimize-solution");
    use grb::prelude::*;

    let _p1 = hprof::enter("build");

    let mut model = Model::with_env("model2", env).map_err(SolverError::GurobiError)?;

    // model
    //     .set_param(param::IntFeasTol, 1e-8)
    //     .map_err(SolverError::GurobiError)?;

    // timing variables
    let t_vars = problem
        .trains
        .iter()
        .enumerate()
        .map(|(train_idx, train)| {
            train
                .visits
                .iter()
                .enumerate()
                .map(|(visit_idx, visit)| {
                    add_ctsvar!(model,
                // name : &format!("tn{}_v{}_tk{}", train_names[train_idx], visit_idx, resource_names[visit.resource_id]), 
                bounds: visit.earliest..,
                obj: 1.0
            )
                    .map_err(SolverError::GurobiError)
                })
                .collect::<Result<Vec<_>, _>>()
        })
        .collect::<Result<Vec<_>, _>>()?;

    // Travel time constraints
    for train_idx in 0..problem.trains.len() {
        for visit_idx in 0..problem.trains[train_idx].visits.len() - 1 {
            let visits = &problem.trains[train_idx].visits;
            #[allow(clippy::useless_conversion)]
            model
                .add_constr("",
                    // &format!(
                    //     "tn{}_v{}_tk{}_travel",
                    //     train_names[train_idx],
                    //     visit_idx,
                    //     resource_names[problem.trains[train_idx].visits[visit_idx].resource_id]
                    // ),
                    c!(
                        (t_vars[train_idx][visit_idx + 1]) - (t_vars[train_idx][visit_idx])
                            >= visits[visit_idx].travel_time
                    ),
                )
                .map_err(SolverError::GurobiError)?;
        }
    }

    // Priorities
    for ((t1, v1), (t2, v2)) in priorities {
        #[allow(clippy::useless_conversion)]
        model
            .add_constr("",
                // &format!(
                //     "tn{}_v{}_tk{}_pri_tn{}_v{}_tk{}",
                //     train_names[t1],
                //     v1,
                //     resource_names[problem.trains[t1].visits[v1].resource_id],
                //     train_names[t2],
                //     v2,
                //     resource_names[problem.trains[t2].visits[v2].resource_id]
                // ),
                c!(t_vars[t1][v1 + 1] <= t_vars[t2][v2]),
            )
            .map_err(SolverError::GurobiError)?;
    }

    drop(_p1);
    {
        let _p = hprof::enter("solve");
        model.optimize().map_err(SolverError::GurobiError)?;
        assert!(model.status().map_err(SolverError::GurobiError)? == Status::Optimal);
    }

    let _p2 = hprof::enter("extract");

    let mut solution = Vec::new();
    for (train_idx, train_ts) in t_vars.iter().enumerate() {
        let mut train_solution = Vec::new();
        for visit_start_t_var in train_ts {
            let t = model
                .get_obj_attr(attr::X, visit_start_t_var)
                .map_err(SolverError::GurobiError)?
                .round() as i32;
            train_solution.push(t);
        }

        train_solution.push(
            train_solution.last().copied().unwrap()
                + problem.trains[train_idx].visits.last().unwrap().travel_time,
        );

        solution.push(train_solution);
    }

    Ok(solution)
}
