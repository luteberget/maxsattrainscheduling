use crate::{
    interval::TimeInterval,
    problem::*,
    train::{TrainSolver, TrainSolverStatus}, occupation::ResourceConflicts,
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
    pub track: TrackRef,
    pub interval: TimeInterval,
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
    pub problem: Rc<Problem>,
    pub trains: Vec<TrainSolver>,
    pub conflicts: ResourceConflicts,
    pub queued_nodes: Vec<Rc<ConflictSolverNode>>,
    pub current_node: Option<Rc<ConflictSolverNode>>, // The root node is "None".
    pub dirty_trains: Vec<u32>,
    pub dirty_train_idxs: Vec<i32>,
}

impl ConflictSolver {
    pub fn status(&self) -> ConflictSolverStatus {
        match (
            self.dirty_trains.is_empty(),
            self.conflicts.conflicting_resource_set.is_empty(),
            self.queued_nodes.is_empty(),
        ) {
            (false, _, _) => ConflictSolverStatus::SolveTrains,
            (_, false, _) => ConflictSolverStatus::Conflict,
            (_, _, false) => ConflictSolverStatus::SelectNode,
            (true, true, true) => ConflictSolverStatus::Exhausted,
        }
    }

    pub fn step(&mut self) {
        // TODO possibly resolve conflicts before finishing solving all trains.

        if let Some(dirty_train) = self.dirty_trains.last() {
            let dirty_train_idx = *dirty_train as usize;
            match self.trains[dirty_train_idx].status() {
                TrainSolverStatus::Failed => {
                    debug!("Train {} failed.", dirty_train_idx);
                    self.switch_node();
                }
                TrainSolverStatus::Optimal => {
                    debug!("Train {} optimal.", dirty_train_idx);
                    Self::remove_dirty_train(
                        &mut self.dirty_trains,
                        &mut self.dirty_train_idxs,
                        dirty_train_idx,
                    );
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
                }
            }
        } else if !self.conflicts.conflicting_resource_set.is_empty() {
            self.branch();
        } else if !self.queued_nodes.is_empty() {
            debug!("Switching node");
            self.switch_node();
        } else {
            debug!("Search exhausted.")
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

        // If both trains are running at their minimum travel time, then
        // we can create valid conflicting intervals.

        // TODO: remove assumptions that the trains are as early as possible to every event.

        let block_a = TimeInterval {
            time_start: occ_a.interval.time_start,
            time_end: occ_b.interval.time_end,
        };

        let block_b = TimeInterval {
            time_start: occ_b.interval.time_start,
            time_end: occ_a.interval.time_end,
        };

        assert!(block_a.overlap(&block_b));

        let constraint_a = ConflictConstraint {
            train: occ_a.train,
            track: conflict_resource as TrackRef,
            interval: block_a,
        };

        let constraint_b = ConflictConstraint {
            train: occ_b.train,
            track: conflict_resource as TrackRef,
            interval: block_b,
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
                    self.trains[train as usize].add_constraint(
                        node.constraint.track,
                        node.constraint.interval,
                        |a, t, i| self.conflicts.add_or_remove(a, train, t, i),
                    );
                    Self::add_dirty_train(
                        &mut self.dirty_trains,
                        &mut self.dirty_train_idxs,
                        train as usize,
                    );

                    forward = node.parent.as_ref();
                } else {
                    let node = backward.unwrap();

                    // Remove constraint
                    let train = node.constraint.train;

                    self.trains[train as usize].remove_constraint(
                        node.constraint.track,
                        node.constraint.interval,
                        |a, t, i| self.conflicts.add_or_remove(a, train, t, i),
                    );
                    Self::add_dirty_train(
                        &mut self.dirty_trains,
                        &mut self.dirty_train_idxs,
                        train as usize,
                    );

                    backward = node.parent.as_ref();
                }
            }
        }

        self.current_node = Some(new_node);
    }

    fn remove_dirty_train(
        dirty_trains: &mut Vec<u32>,
        dirty_train_idxs: &mut [i32],
        train_idx: usize,
    ) {
        assert!(dirty_train_idxs[train_idx] >= 0);

        let dirty_idx = dirty_train_idxs[train_idx] as usize;
        dirty_trains.swap_remove(dirty_idx);
        if dirty_idx < dirty_trains.len() {
            let other_train = dirty_trains[dirty_idx] as usize;
            dirty_train_idxs[other_train] = dirty_idx as i32;
        }
        dirty_train_idxs[train_idx] = -1;
    }

    fn add_dirty_train(
        dirty_trains: &mut Vec<u32>,
        dirty_train_idxs: &mut [i32],
        train_idx: usize,
    ) {
        if dirty_train_idxs[train_idx] == -1 {
            dirty_train_idxs[train_idx] = dirty_trains.len() as i32;
            dirty_trains.push(train_idx as u32);
        }
    }

    pub fn new(problem: Rc<Problem>) -> Self {
        let trains: Vec<TrainSolver> = problem
            .trains
            .iter()
            .enumerate()
            .map(|(train_idx, _t)| {
                TrainSolver::new(
                    problem.clone(),
                    train_idx,
                )
            })
            .collect();
        let conflicts = crate::occupation::ResourceConflicts::empty(problem.tracks.len());
        let dirty_trains = (0..(trains.len() as u32)).collect();
        let dirty_train_idxs = (0..(trains.len() as i32)).collect();
        Self {
            problem,
            trains,
            conflicts,
            current_node: None,
            dirty_trains,
            dirty_train_idxs,
            queued_nodes: vec![],
        }
    }
}
