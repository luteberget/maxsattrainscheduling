use crate::{
    occupation::ResourceConflicts,
    problem::*,
    train::{TrainSolver, TrainSolverStatus},
};
use log::debug;
use std::rc::Rc;

#[derive(Debug)]
pub enum ConflictSolverStatus {
    Exhausted,
    SelectNode,
    Conflict,
    SolveTrains,
}

#[derive(Debug)]
pub struct ConflictConstraint {
    pub train: TrainRef,
    pub resource: ResourceRef,
    pub enter_after: TimeValue,
    pub leave_before: TimeValue,
}

pub struct ConflictSolverNode {
    pub constraint: ConflictConstraint,
    pub depth: u32,
    pub parent: Option<Rc<ConflictSolverNode>>,
}

impl std::fmt::Debug for ConflictSolverNode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ConflictSolverNode")
            .field("constraint", &self.constraint)
            .field("depth", &self.depth)
            .field("has_parent", &self.parent.is_some())
            .finish()
    }
}

pub struct ConflictSolver {
    pub trains: Vec<TrainSolver>,
    pub conflicts: ResourceConflicts,
    pub queued_nodes: Vec<Rc<ConflictSolverNode>>,
    pub current_node: Option<Rc<ConflictSolverNode>>, // The root node is "None".
    pub dirty_trains: Vec<u32>,
    pub total_nodes: usize,
    // pub dirty_train_idxs: Vec<i32>,
}

impl ConflictSolver {
    pub fn status(&self) -> ConflictSolverStatus {
        match (
            self.dirty_trains.is_empty(),
            self.conflicts.conflicting_resource_set.is_empty(),
            self.queued_nodes.is_empty(),
        ) {
            (_, false, _) => ConflictSolverStatus::Conflict,
            (false, _, _) => ConflictSolverStatus::SolveTrains,
            (_, _, false) => ConflictSolverStatus::SelectNode,
            (true, true, true) => ConflictSolverStatus::Exhausted,
        }
    }

    pub fn step(&mut self) {
        // TODO We are solving conflicts when they appear, before trains are
        // finished solving. Is there a realistic pathological case for this?

        if !self.conflicts.conflicting_resource_set.is_empty() {
            self.branch();
        } else if let Some(dirty_train) = self.dirty_trains.last() {
            let dirty_train_idx = *dirty_train as usize;
            match self.trains[dirty_train_idx].status() {
                TrainSolverStatus::Failed => {
                    debug!("Train {} failed.", dirty_train_idx);
                    self.switch_node();
                }
                TrainSolverStatus::Optimal => {
                    debug!("Train {} optimal.", dirty_train_idx);
                    self.dirty_trains.pop();
                }
                TrainSolverStatus::Working => {
                    self.trains[dirty_train_idx].step(|add, track, interval| {
                        self.conflicts.add_or_remove(
                            add,
                            dirty_train_idx as TrainRef,
                            track,
                            interval,
                        )
                    });

                    Self::train_queue_bubble_leftward(&mut self.dirty_trains, &self.trains);
                }
            }
        } else if !self.queued_nodes.is_empty() {
            debug!("Switching node");
            self.switch_node();
        } else {
            debug!("Search exhausted.")
        }
    }

    fn train_queue_bubble_leftward(dirty_trains: &mut [u32], trains: &[TrainSolver]) {
        // Bubble the last train in the dirty train queue to the correct ordering.
        let mut idx = dirty_trains.len();
        while idx >= 2
            && trains[dirty_trains[idx - 2] as usize].current_time()
                < trains[dirty_trains[idx - 1] as usize].current_time()
        {
            dirty_trains.swap(idx - 2, idx - 1);
            idx -= 1;
        }
    }

    /// Select a conflict and create constraints for it.
    fn branch(&mut self) {
        assert!(!self.conflicts.conflicting_resource_set.is_empty());
        let conflict_resource = self.conflicts.conflicting_resource_set[0];
        // println!("CONFLICTS {:#?}", self.conflicts);
        let (occ_a, occ_b) = self.conflicts.resources[conflict_resource as usize]
            .get_conflict()
            .unwrap();

        let constraint_a = ConflictConstraint {
            train: occ_a.train,
            resource: conflict_resource,
            enter_after: occ_b.interval.time_end,
            leave_before: occ_a.interval.time_end,
        };

        let constraint_b = ConflictConstraint {
            train: occ_b.train,
            resource: conflict_resource,
            enter_after: occ_a.interval.time_end,
            leave_before: occ_b.interval.time_end,
        };

        debug!(
            "Branch:\n - a: {:?}\n - b: {:?}",
            constraint_a, constraint_b
        );

        let node_a = ConflictSolverNode {
            constraint: constraint_a,
            depth: self.current_node.as_ref().map_or(0, |n| n.depth) + 1,
            parent: self.current_node.clone(),
        };

        let node_b = ConflictSolverNode {
            constraint: constraint_b,
            depth: self.current_node.as_ref().map_or(0, |n| n.depth) + 1,
            parent: self.current_node.clone(),
        };

        self.queued_nodes.push(Rc::new(node_a));
        self.queued_nodes.push(Rc::new(node_b));

        self.total_nodes += 2;

        // TODO special-case DFS without putting the node on the queue.

        self.switch_node();
    }

    fn select_node(&mut self) -> Option<Rc<ConflictSolverNode>> {
        self.queued_nodes.pop()
    }

    fn switch_node(&mut self) {
        let new_node = self.select_node().unwrap();
        debug!("conflict search switching to node {:?}", new_node);
        let mut backward = self.current_node.as_ref();
        let mut forward = Some(&new_node);

        loop {
            // trace!(
            //     "Comparing backward depth1 {} to forawrd depth2 {}",
            //     depth1,
            //     depth2
            // );

            let same_node = match (backward, forward) {
                (Some(rc1), Some(rc2)) => Rc::ptr_eq(rc1, rc2),
                (None, None) => true,
                _ => false,
            };
            if same_node {
                break;
            } else {
                let depth1 = backward.as_ref().map_or(0, |n| n.depth);
                let depth2 = forward.as_ref().map_or(0, |n| n.depth);
                if depth1 < depth2 {
                    let node = forward.unwrap();

                    // Add constraint
                    let train = node.constraint.train;
                    self.trains[train as usize].set_occupied(
                        node.constraint.resource,
                        node.constraint.enter_after,
                        node.constraint.leave_before,
                        |a, t, i| self.conflicts.add_or_remove(a, train, t, i),
                    );

                    Self::train_queue_rewinded_train(train, &mut self.dirty_trains, &self.trains);
                    forward = node.parent.as_ref();
                } else {
                    let node = backward.unwrap();

                    // Remove constraint
                    let train = node.constraint.train;

                    self.trains[train as usize].remove_occupied(
                        node.constraint.resource,
                        node.constraint.enter_after,
                        node.constraint.leave_before,
                        |a, t, i| self.conflicts.add_or_remove(a, train, t, i),
                    );

                    Self::train_queue_rewinded_train(train, &mut self.dirty_trains, &self.trains);
                    backward = node.parent.as_ref();
                }
            }
        }

        self.current_node = Some(new_node);
    }

    fn train_queue_rewinded_train(
        train: TrainRef,
        dirty_trains: &mut Vec<u32>,
        trains: &[TrainSolver],
    ) {
        if let Some((mut idx, _)) = dirty_trains
            .iter()
            .enumerate()
            .rev() /* Search from the back, because a conflicting train is probably recently used. */
            .find(|(_, t)| **t == train as u32)
        {
            // Bubble to the right, in this case, because the train has become earlier.
            while idx + 1 < dirty_trains.len()
                && trains[dirty_trains[idx] as usize].current_time()
                    < trains[dirty_trains[idx+1] as usize].current_time()
            {
                dirty_trains.swap(idx, idx +1);
                idx += 1;
            }
        } else {
            dirty_trains.push(train as u32);
            Self::train_queue_bubble_leftward(dirty_trains, trains);
        }
    }

    pub fn new(problem: Problem) -> Self {
        problem.verify();

        let trains: Vec<TrainSolver> = problem.trains.into_iter().map(TrainSolver::new).collect();
        let conflicts = crate::occupation::ResourceConflicts::empty(problem.n_resources);

        let mut dirty_trains: Vec<u32> = (0..(trains.len() as u32)).collect();
        dirty_trains.sort_by_key(|t| -(trains[*t as usize].current_time() as i32));

        Self {
            trains,
            conflicts,
            current_node: None,
            dirty_trains,
            queued_nodes: vec![],
            total_nodes: 0,
        }
    }
}
