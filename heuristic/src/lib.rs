use interval::TimeInterval;
use problem::{TimeValue, BlockRef, ResourceRef};

pub mod problem;
pub mod solvers;
pub mod interval;
pub mod occupation;
pub mod branching;
pub mod trainset;


#[derive(Debug)]
pub enum TrainSolverStatus {
    Failed,
    Optimal,
    Working,
}

pub trait TrainSolver {
    fn current_solution(&self) -> (i32, Vec<TimeValue>);
    fn current_time(&self) -> TimeValue;
    fn status(&self) -> TrainSolverStatus;
    fn step(&mut self, use_resource: &mut impl FnMut(bool, BlockRef, ResourceRef, TimeInterval));
    fn set_occupied(
        &mut self,
        add: bool,
        resource: ResourceRef,
        enter_after: TimeValue,
        exit_before: TimeValue,
        use_resource: &mut impl FnMut(bool, BlockRef, ResourceRef, TimeInterval),
    );
    fn new(id: usize, train: crate::problem::Train) -> Self;
}
