pub mod maxsatddd;
pub mod bigm;
pub mod binarizedbigm;
pub mod greedy;
pub mod idl;
pub mod mipdddpack;
pub mod costtree;
mod minimize;
pub mod maxsatddd_full;
pub mod heuristic;
// pub mod cutting;


#[derive(Debug)]
pub enum SolverError {
    NoSolution,
    GurobiError(grb::Error),
    Timeout,
}