use crate::{
    node_eval::NodeEval, occupation::ResourceConflicts, problem::*, trainset::TrainSet,
    ConflictSolver, TrainSolver,
};
use log::{debug, info, warn};
use std::{
    collections::{BinaryHeap, HashSet},
    rc::Rc,
};

use crate::branching::{Branching, ConflictSolverNode};

use super::train_queue::QueueTrainSolver;

#[derive(Debug)]
pub enum ConflictSolverStatus {
    Exhausted,
    SelectNode,
    Conflict,
    SolveTrains,
}

impl PartialEq for ConflictSolverNode<NodeEval> {
    fn eq(&self, other: &Self) -> bool {
        self.eval.node_score == other.eval.node_score
    }
}

impl PartialOrd for ConflictSolverNode<NodeEval> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.eval.node_score.partial_cmp(&other.eval.node_score)
    }
}

impl Eq for ConflictSolverNode<NodeEval> {}

impl Ord for ConflictSolverNode<NodeEval> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.partial_cmp(other).unwrap()
    }
}

pub struct BnBConflictSolver<Train> {
    pub input: Vec<crate::problem::Train>,
    pub trainset: TrainSet<Train>,

    pub conflict_space: Branching<NodeEval>,
    pub queued_nodes: BinaryHeap<Rc<ConflictSolverNode<NodeEval>>>,
    pub conflicts: ResourceConflicts,

    pub ub: i32,
    pub priorities: HashSet<(TrainRef, TrainRef, u32)>,
}

impl ConflictSolver for BnBConflictSolver<QueueTrainSolver> {
    fn trainset(&self) -> &TrainSet<QueueTrainSolver> {
        &self.trainset
    }

    fn conflicts(&self) -> &ResourceConflicts {
        &self.conflicts
    }

    fn small_step(&mut self) -> Option<(i32, Vec<Vec<i32>>)> {
        self.solve_partial()
    }

    fn big_step(&mut self) -> Option<(i32, Vec<Vec<i32>>)> {
        self.solve_partial_alltrains()
    }
}

impl<Train: TrainSolver> BnBConflictSolver<Train> {
    pub fn status(&self) -> ConflictSolverStatus {
        match (
            !self.trainset.is_dirty(),
            !self.conflicts.has_conflict(),
            self.queued_nodes.is_empty(),
        ) {
            (_, _, true) if self.trainset.lb >= self.ub => ConflictSolverStatus::Exhausted,
            (_, false, _) => ConflictSolverStatus::Conflict,
            (false, _, _) => ConflictSolverStatus::SolveTrains,
            (_, _, false) => ConflictSolverStatus::SelectNode,
            (true, true, true) => ConflictSolverStatus::Exhausted,
        }
    }

    pub fn solve_partial(&mut self) -> Option<(i32, Vec<Vec<i32>>)> {
        if matches!(self.status(), ConflictSolverStatus::Exhausted) {
            return None;
        }

        self.step();

        if !self.trainset.is_dirty() && !self.conflicts.has_conflict() {
            return Some(self.trainset.current_solution());
        }
        return None;
    }

    pub fn solve_partial_alltrains(&mut self) -> Option<(i32, Vec<Vec<i32>>)> {
        if self.trainset.is_dirty() {
            self.trainset.solve_all_trains(&mut self.conflicts);
            return None;
        }
        if !self.trainset.is_dirty() && !self.conflicts.has_conflict() {
            return Some(self.trainset.current_solution());
        }

        if matches!(self.status(), ConflictSolverStatus::Exhausted) {
            return None;
        }

        self.step();
        if self.trainset.is_dirty() {
            self.trainset.solve_all_trains(&mut self.conflicts);
        }

        if !self.trainset.is_dirty() && !self.conflicts.has_conflict() {
            return Some(self.trainset.current_solution());
        }
        return None;
    }

    pub fn solve_next_stopcb(
        &mut self,
        mut stop: impl FnMut() -> bool,
    ) -> Option<(i32, Vec<Vec<i32>>)> {
        loop {
            if stop() || matches!(self.status(), ConflictSolverStatus::Exhausted) {
                return None;
            }

            self.step();

            if !self.trainset.is_dirty() && !self.conflicts.has_conflict() {
                return Some(self.trainset.current_solution());
            }
        }
    }

    pub fn solve_next(&mut self) -> Option<(i32, Vec<Vec<i32>>)> {
        loop {
            if matches!(self.status(), ConflictSolverStatus::Exhausted) {
                return None;
            }

            self.step();

            if !self.trainset.is_dirty() && !self.conflicts.has_conflict() {
                return Some(self.trainset.current_solution());
            }
        }
    }

    pub fn step(&mut self) {
        let _p = hprof::enter("conflict step");
        // TODO We are solving conflicts when they appear, before trains are
        // finished solving. Is there a realistic pathological case for this?

        loop {
            if self.trainset.lb >= self.ub {
                // println!("Node fathomed. {} >= {}", self.lb, self.ub);
                self.switch_to_any_node();
                break;
            } else if !self.conflicts.conflicting_resource_set.is_empty() {
                self.branch();
                break;
            } else if self.trainset.is_dirty()
            // let Some(&dirty_train) = self.trainset.dirty_trains.last()
            {
                match self.trainset.solve_step(&mut self.conflicts) {
                    Some(false) => {
                        self.switch_to_any_node();
                        break;
                    }
                    Some(true) => {
                        break;
                    }
                    None => {}
                }
            } else {
                // New solution
                let (cost, _sol) = self.trainset.current_solution();
                if cost < self.ub {
                    self.ub = cost;
                    self.priorities = self.current_priorities();
                    // println!(" setting priorities {:?}", self.priorities);
                }

                if !self.queued_nodes.is_empty() {
                    debug!("Switching node");
                    self.switch_to_any_node();
                    break;
                } else {
                    print!("Search exhausted.");
                    break;
                }
            }

            if !self.conflicts.conflicting_resource_set.is_empty() {
                break;
            }
        }
    }

    fn current_priorities(&self) -> HashSet<(TrainRef, TrainRef, ResourceRef)> {
        let mut set = HashSet::new();
        let mut node = self.conflict_space.current_node.as_ref();
        loop {
            if let Some(n) = node {
                set.insert((
                    n.constraint.train,
                    n.constraint.other_train,
                    n.constraint.resource,
                ));

                node = n.parent.as_ref();
            } else {
                break;
            }
        }

        set
    }

    /// Select a conflict and create constraints for it.
    fn branch(&mut self) {
        assert!(!self.conflicts.conflicting_resource_set.is_empty());

        let conflict = self.conflicts.first_conflict().unwrap();

        let (node_a, node_b) = self.conflict_space.branch(conflict, |c| {
            info!(
                "Evaluating:\n - a {:?} {:?}\n - b {:?} {:?}",
                c.occ_a, c.c_a, c.occ_b, c.c_b
            );
            let eval = crate::node_eval::node_evaluation(
                &self.trainset.original_trains,
                &self.trainset.slacks,
                c.occ_a,
                c.c_a,
                c.occ_b,
                c.c_b,
                None,
                false,
            );
            (eval, eval)
        });

        let mut chosen_node = None;
        if let Some(node_a) = node_a {
            if node_a.eval.a_best {
                self.queued_nodes.push(node_a);
            } else {
                chosen_node = Some(node_a);
            }
        }

        if let Some(node_b) = node_b {
            if !node_b.eval.a_best {
                self.queued_nodes.push(node_b);
            } else {
                chosen_node = Some(node_b);
            }
        }

        if let Some(chosen_node) = chosen_node {
            self.set_node(chosen_node);
        } else {
            self.switch_to_any_node();
        }
    }

    fn switch_to_any_node(&mut self) {
        warn!("non-DFS node");
        let new_node = match self.queued_nodes.pop() {
            Some(n) => n,
            None => {
                println!("No more nodes. Status: {:?}", self.status());
                // assert!(matches!(self.status(), ConflictSolverStatus::Exhausted));
                return;
            }
        };

        self.set_node(new_node);
    }

    fn set_node(&mut self, node: Rc<ConflictSolverNode<NodeEval>>) {
        self.conflict_space.set_node(Some(node), &mut |add, constraint| {
            self.trainset
                .add_remove_constraint(add, constraint, &mut self.conflicts)
        });
    }

    pub fn new(problem: Problem) -> Self {
        problem.verify();
        let n_resources = problem.n_resources;
        let input = problem.trains.clone();
        let trainset = TrainSet::new(problem);
        Self {
            conflict_space: Branching::new(trainset.trains.len()),
            input,

            trainset,
            queued_nodes: Default::default(),
            ub: i32::MAX,
            priorities: Default::default(),
            conflicts: crate::occupation::ResourceConflicts::empty(n_resources),
        }
    }
}
