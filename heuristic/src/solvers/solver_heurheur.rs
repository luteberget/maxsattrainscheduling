use crate::{
    branching::Branching, node_eval::NodeEval, occupation::ResourceConflicts, problem::Problem,
    trainset::TrainSet,
};

use super::train_queue::QueueTrainSolver;

pub struct HeurHeur {
    pub input: Vec<crate::problem::Train>,
    pub trainset: TrainSet<QueueTrainSolver>,

    pub conflict_space: Branching<()>,
    pub conflicts: ResourceConflicts,
}

impl HeurHeur {
    pub fn new(problem: Problem) -> Self {
        let n_resources = problem.n_resources;
        let input = problem.trains.clone();
        let trainset = TrainSet::new(problem);
        Self {
            conflict_space: Branching::new(trainset.trains.len()),
            input,
            trainset,
            conflicts: crate::occupation::ResourceConflicts::empty(n_resources),
        }
    }

    pub fn solve_step(&mut self) -> Option<Option<(i32, Vec<Vec<i32>>)>> {
        if self.conflicts.has_conflict() {
            let (conflict_resource, (occ_a, occ_b)) = self
                .conflicts
                .conflicting_resource_set
                .iter()
                .map(|r| {
                    (
                        *r,
                        self.conflicts.resources[*r as usize]
                            .get_conflict()
                            .unwrap(),
                    )
                })
                .min_by_key(|(_, c)| c.0.interval.time_start.min(c.1.interval.time_start))
                .unwrap();

            let (node_a, node_b) = self
                .conflict_space
                .branch((conflict_resource, occ_a, occ_b), |_| ((), ()));

            

            None
        } else if self.trainset.is_dirty() {
            // Solve trains until conflict or done.

            None
        } else {
            // Done.
            Some(Some(self.trainset.current_solution()))
        }
    }

    pub fn solve(&mut self) -> Option<(i32, Vec<Vec<i32>>)> {
        loop {
            if let Some(x) = self.solve_step() {
                return x;
            }
        }
    }
}
