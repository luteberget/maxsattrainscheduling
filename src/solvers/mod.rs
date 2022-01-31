pub mod maxsatddd;
pub mod bigm;


#[derive(Debug)]
pub enum SolverError {
    NoSolution,
    GurobiError(grb::Error)
}