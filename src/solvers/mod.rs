pub mod maxsatddd;
pub mod bigm;
pub mod mipddd;
pub mod greedy;
pub mod idl;
pub mod mipdddpack;


#[derive(Debug)]
pub enum SolverError {
    NoSolution,
    GurobiError(grb::Error),
    Timeout,
}