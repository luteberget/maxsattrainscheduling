use std::collections::{HashMap, BTreeMap};

use interval::TimeInterval;
use occupation::ResourceConflicts;
use problem::{TimeValue, BlockRef, ResourceRef, TrainRef};
use solvers::train_queue::QueueTrainSolver;
use trainset::TrainSet;

pub mod problem;
pub mod solvers;
pub mod interval;
pub mod occupation;
pub mod branching;
pub mod trainset;
pub mod node_eval;


#[derive(Debug)]
pub enum TrainSolverStatus {
    Failed,
    Optimal,
    Working,
}

pub trait TrainSolver {
    fn current_solution(&self) -> (i32, Vec<TimeValue>);
    fn current_time(&self) -> TimeValue;
    fn order_time(&self) -> TimeValue;
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


pub trait ConflictSolver {
    fn trainset(&self) -> &TrainSet<QueueTrainSolver>;
    fn conflicts(&self) -> &ResourceConflicts;
    fn visit_weights(&self) -> Option<&BTreeMap<(TrainRef,BlockRef), f32>> { None }
    fn small_step(&mut self) -> Option<(i32,Vec<Vec<i32>>)>;
    fn big_step(&mut self) -> Option<(i32,Vec<Vec<i32>>)>;
}