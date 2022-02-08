pub mod maxsatddd;
pub mod bigm;
mod ddd;


#[derive(Debug)]
pub enum SolverError {
    NoSolution,
    GurobiError(grb::Error)
}