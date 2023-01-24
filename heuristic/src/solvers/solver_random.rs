use std::rc::Rc;

use crate::{
    branching::{Branching, ConflictSolverNode},
    occupation::ResourceConflicts,
    problem::Problem,
    trainset::TrainSet,
    ConflictSolver,
};

use super::train_queue::QueueTrainSolver;

pub struct RandomHeuristic {
    pub input: Vec<crate::problem::Train>,
    pub trainset: TrainSet<QueueTrainSolver>,

    pub conflict_space: Branching<()>,
    pub conflicts: ResourceConflicts,
    queue: Vec<Rc<ConflictSolverNode<()>>>,
}

impl RandomHeuristic {
    pub fn new(problem: Problem) -> Self {
        let n_resources = problem.n_resources;
        let input = problem.trains.clone();
        let trainset = TrainSet::new(problem);
        Self {
            conflict_space: Branching::new(trainset.trains.len()),
            input,
            trainset,
            conflicts: crate::occupation::ResourceConflicts::empty(n_resources),
            queue: Vec::new(),
        }
    }

    pub fn solve_step(&mut self) -> Option<Option<(i32, Vec<Vec<i32>>)>> {
        if let Some(conflict) = self.conflicts.first_conflict() {
            let (node_a, node_b) = self.conflict_space.branch(conflict, |_| ((), ()));

            let node = match (node_a, node_b) {
                (None, None) => {
                    println!("Random heuristic failed, backtracking to random discarded node.");
                    self.queue
                        .remove(rand::random::<usize>() % self.queue.len())
                }
                (None, Some(b)) => b,
                (Some(a), None) => a,
                (Some(a), Some(b)) => {
                    if rand::random() {
                        self.queue.push(b);
                        a
                    } else {
                        self.queue.push(a);
                        b
                    }
                }
            };

            self.conflict_space.set_node(Some(node), &mut |a, c| {
                self.trainset
                    .add_remove_constraint(a, c, &mut self.conflicts)
            });

            None
        } else if self.trainset.is_dirty() {
            // Solve trains until conflict or done.
            println!("{:?}", self.trainset.dirty_trains);
            self.trainset.solve_first_train(&mut self.conflicts);
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

impl ConflictSolver for RandomHeuristic {
    fn trainset(&self) -> &TrainSet<QueueTrainSolver> {
        &self.trainset
    }

    fn conflicts(&self) -> &ResourceConflicts {
        &self.conflicts
    }

    fn small_step(&mut self) -> Option<(i32, Vec<Vec<i32>>)> {
        self.solve_step().flatten()
    }

    fn big_step(&mut self) -> Option<(i32, Vec<Vec<i32>>)> {
        self.solve()
    }
}