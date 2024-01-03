use crate::{problem::{Problem, DelayCostType}, maxsatsolver::MaxSatSolver};

use super::{maxsatddd_ladder::SolveStats, SolverError};


pub fn solve(
    s :impl MaxSatSolver,
    env: &grb::Env,
    problem: &Problem,
    timeout: f64,
    delay_cost_type: DelayCostType,
    output_stats: impl FnMut(String, serde_json::Value),
) -> Result<(Vec<Vec<i32>>, SolveStats), SolverError> {
    todo!()
}