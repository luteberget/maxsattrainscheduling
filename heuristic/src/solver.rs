use crate::{
    occupation::ResourceConflicts,
    problem::*,
    train::{TrainSolver, TrainSolverStatus},
};
use log::{debug, warn};
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
    pub other_train: TrainRef,
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
    pub prev_conflict: Vec<TinyVec<[(ResourceRef, u32); 16]>>,
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

    pub fn solve_any(mut self) -> Vec<Vec<i32>> {
        while !self.dirty_trains.is_empty() || !self.conflicts.conflicting_resource_set.is_empty() {
            self.step();
        }
        self.current_solution()
    }

    pub fn current_solution(&self) -> Vec<Vec<TimeValue>> {
        self.trains.iter().map(|t| t.current_solution()).collect()
    }

    pub fn step(&mut self) {
        let _p = hprof::enter("conflict step");
        // TODO We are solving conflicts when they appear, before trains are
        // finished solving. Is there a realistic pathological case for this?

        loop {
            if !self.conflicts.conflicting_resource_set.is_empty() {
                self.branch();
                break;
            } else if let Some(dirty_train) = self.dirty_trains.last() {
                let dirty_train_idx = *dirty_train as usize;
                match self.trains[dirty_train_idx].status() {
                    TrainSolverStatus::Failed => {
                        debug!("Train {} failed.", dirty_train_idx);
                        self.switch_node();
                        break;
                    }
                    TrainSolverStatus::Optimal => {
                        debug!("Train {} optimal.", dirty_train_idx);
                        self.dirty_trains.pop();
                        break;
                    }
                    TrainSolverStatus::Working => {
                        self.trains[dirty_train_idx].step(&mut |add, track, interval| {
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
                break;
            } else {
                debug!("Search exhausted.");
                break;
            }

            if !self.conflicts.conflicting_resource_set.is_empty() {
                break;
            }
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

        assert!(occ_a.train != occ_b.train);

        let constraint_a = ConflictConstraint {
            train: occ_a.train,
            other_train: occ_b.train,
            resource: conflict_resource,
            enter_after: occ_b.interval.time_end,
            leave_before: occ_a.interval.time_end,
        };

        let constraint_b = ConflictConstraint {
            train: occ_b.train,
            other_train: occ_a.train,
            resource: conflict_resource,
            enter_after: occ_a.interval.time_end,
            leave_before: occ_b.interval.time_end,
        };

        debug!(
            "Branch:\n - a: {:?}\n - b: {:?}",
            constraint_a, constraint_b
        );


        if self.is_reasonable_constraint(&constraint_a, self.current_node.as_ref()) {
            let node_a = ConflictSolverNode {
                constraint: constraint_a,
                depth: self.current_node.as_ref().map_or(0, |n| n.depth) + 1,
                parent: self.current_node.clone(),
            };
            self.queued_nodes.push(Rc::new(node_a));
        } else {
            warn!("Skipping loop-like constraint {:?}", constraint_a);
        }

        if self.is_reasonable_constraint(&constraint_b, self.current_node.as_ref()) {
            let node_b = ConflictSolverNode {
                constraint: constraint_b,
                depth: self.current_node.as_ref().map_or(0, |n| n.depth) + 1,
                parent: self.current_node.clone(),
            };
            self.queued_nodes.push(Rc::new(node_b));
        } else {
            warn!("Skipping loop-like constraint {:?}", constraint_b);
        }


        // TODO Here, we will want to know whether these two trains ever conflicted before.

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
                        &mut |a, res, i| self.conflicts.add_or_remove(a, train, res, i),
                    );

                    {
                        let mut found = false;
                        for idx in (0..(self.prev_conflict[train as usize]).len()).rev() {
                            let elem = &mut self.prev_conflict[train as usize][idx];
                            if elem.0 == node.constraint.resource {
                                assert!(elem.1 > 0);
                                found = true;
                                elem.1 += 1;
                            }
                        }
                        if !found {
                            self.prev_conflict[train as usize].push((node.constraint.resource, 1));
                        }
                    }

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
                        &mut |a, t, i| self.conflicts.add_or_remove(a, train, t, i),
                    );

                    {
                        let mut found = false;
                        for idx in (0..(self.prev_conflict[train as usize]).len()).rev() {
                            let elem = &mut self.prev_conflict[train as usize][idx];
                            if elem.0 == node.constraint.resource {
                                assert!(elem.1 > 0);
                                found = true;
                                elem.1 -= 1;
                                if elem.1 == 0 {
                                    self.prev_conflict[train as usize].remove(idx);
                                }
                            }
                        }
                        assert!(found);
                    }

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
        dirty_trains.sort_by_key(|t| -(trains[*t as usize].current_time() as i64));

        let prev_conflict = trains.iter().map(|_| Default::default()).collect();

        Self {
            trains,
            conflicts,
            current_node: None,
            dirty_trains,
            queued_nodes: vec![],
            total_nodes: 0,
            prev_conflict,
        }
    }

    pub(crate) fn is_reasonable_constraint(
        &self,
        constr: &ConflictConstraint,
        current_node: Option<&Rc<ConflictSolverNode>>,
    ) -> bool {
        // If this train has less than 3 conflict on the same resource, it is reasonable.
        let res = constr.resource;
        let n_conflicts = *self.prev_conflict[constr.train as usize]
            .iter()
            .find_map(|(r, n)| (*r == res).then(|| n))
            .unwrap_or(&0);

        if n_conflicts < 3 {
            return true;
        }

        // This train has multiple conflicts on the same resource.
        // We do a (heuristic) cycle check.

        let current_node = current_node.unwrap();
        let mut curr_node = current_node;
        let mut curr_train = constr.train;
        let mut count = 0;

        const MAX_CYCLE_CHECK_LENGTH: usize = 10;
        const MAX_CYCLE_LENGTH: usize = 2;

        for _ in 0..MAX_CYCLE_CHECK_LENGTH {
            if curr_node.constraint.other_train == curr_train {
                curr_train = curr_node.constraint.other_train;
                if curr_train == constr.train {
                    count += 1;
                    if count >= MAX_CYCLE_LENGTH {
                        return false;
                    }
                }
            }

            if let Some(x) = curr_node.parent.as_ref() {
                curr_node = x;
            } else {
                break;
            }
        }

        true
    }
}
