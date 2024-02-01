pub mod maxsatddd_ladder;
pub mod bigm;
pub mod binarizedbigm;
pub mod greedy;
pub mod idl;
pub mod mipdddpack;
pub mod costtree;
mod minimize;
pub mod maxsat_ti;
pub mod maxsat_ddd;
pub mod heuristic;
pub mod maxsatddd_ladder_abstract;
pub mod milp_ti;
// pub mod cutting;


#[derive(Debug)]
pub enum SolverError {
    NoSolution,
    GurobiError(grb::Error),
    Timeout,
}