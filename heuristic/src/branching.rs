use std::{collections::BinaryHeap, rc::Rc};

use log::debug;
use tinyvec::TinyVec;

use crate::{
    occupation::{ResourceConflicts, ResourceOccupation},
    problem::{ResourceRef, TimeValue, TrainRef},
};

#[derive(Debug)]
pub struct ConflictConstraint {
    pub train: TrainRef,
    pub other_train: TrainRef,
    pub resource: ResourceRef,
    pub enter_after: TimeValue,
    pub exit_before: TimeValue,
}

pub struct ConflictSolverNode<T> {
    pub constraint: ConflictConstraint,
    pub depth: u32,
    pub parent: Option<Rc<ConflictSolverNode<T>>>,
    pub eval: T,
}

impl<T> std::fmt::Debug for ConflictSolverNode<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ConflictSolverNode")
            .field("constraint", &self.constraint)
            .field("depth", &self.depth)
            .field("has_parent", &self.parent.is_some())
            .finish()
    }
}

pub struct Branching<T> {
    pub train_prev_conflict: Vec<TinyVec<[(ResourceRef, u32); 16]>>,
    pub current_node: Option<Rc<ConflictSolverNode<T>>>, // The root node is "None".
    // pub conflicts :ResourceConflicts,
    pub n_nodes_explored: usize,
    pub n_nodes_generated: usize,
}

pub struct ConflictDescription<'a> {
    pub c_a: &'a ConflictConstraint,
    pub c_b: &'a ConflictConstraint,
    pub occ_a :&'a ResourceOccupation,
    pub occ_b :&'a ResourceOccupation,
}

impl<T> Branching<T> {
    pub fn new(n_trains: usize) -> Self {
        Self {
            train_prev_conflict: (0..n_trains).map(|_| Default::default()).collect(),
            // conflicts: crate::occupation::ResourceConflicts::empty(n_resources),
            current_node: None,
            n_nodes_explored: 0,
            n_nodes_generated: 0,
        }
    }

    // pub fn has_conflict(&self) -> bool {
    //     !self.conflicts.conflicting_resource_set.is_empty()
    // }

    pub fn branch(
        &mut self,
        conflict :(ResourceRef, &ResourceOccupation, &ResourceOccupation),
        mut eval_f :impl FnMut(ConflictDescription) -> (T,T),
    ) -> (
        Option<Rc<ConflictSolverNode<T>>>,
        Option<Rc<ConflictSolverNode<T>>>,
    ) {
        // let conflict_resource = conflicts.conflicting_resource_set[0];
        // println!("CONFLICTS {:#?}", self.conflicts);
        let (conflict_resource, occ_a, occ_b) = conflict;
        // conflicts.resources[conflict_resource as usize]
        //     .get_conflict()
        //     .unwrap();

        assert!(occ_a.train != occ_b.train);

        let constraint_a = ConflictConstraint {
            train: occ_a.train,
            other_train: occ_b.train,
            resource: conflict_resource,
            enter_after: occ_b.interval.time_end,
            exit_before: occ_a.interval.time_end,
        };

        let constraint_b = ConflictConstraint {
            train: occ_b.train,
            other_train: occ_a.train,
            resource: conflict_resource,
            enter_after: occ_a.interval.time_end,
            exit_before: occ_b.interval.time_end,
        };

        let (eval_a, eval_b) = eval_f(ConflictDescription {
            c_a: &constraint_a,
            c_b: &constraint_b,
            occ_a :&occ_a,
            occ_b :&occ_b,
        });

        let mut node_a = None;
        let mut node_b = None;

        if self.is_reasonable_constraint(&constraint_a, self.current_node.as_ref()) {
            node_a = Some(Rc::new(ConflictSolverNode {
                constraint: constraint_a,
                depth: self.current_node.as_ref().map_or(0, |n| n.depth) + 1,
                parent: self.current_node.clone(),
                eval: eval_a,
            }));
            self.n_nodes_generated += 1;
        }

        if self.is_reasonable_constraint(&constraint_b, self.current_node.as_ref()) {
            node_b = Some(Rc::new(ConflictSolverNode {
                constraint: constraint_b,
                depth: self.current_node.as_ref().map_or(0, |n| n.depth) + 1,
                parent: self.current_node.clone(),
                eval: eval_b,
            }));
            self.n_nodes_generated += 1;
        }

        (node_a, node_b)
    }

    pub fn set_node(&mut self, new_node: Rc<ConflictSolverNode<T>>, f :&mut impl FnMut(bool, &ConflictConstraint)) {
        self.n_nodes_explored += 1;
        debug!("conflict search switching to node {:?}", new_node);
        let mut backward = self.current_node.as_ref();
        let mut forward = Some(&new_node);

        loop {

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
                    f(true, &node.constraint);
                    
                    // Record previous conflict for loop detection.
                    {
                        let train = node.constraint.train;
                        let mut found = false;
                        for idx in (0..(self.train_prev_conflict[train as usize]).len()).rev() {
                            let elem = &mut self.train_prev_conflict[train as usize][idx];
                            if elem.0 == node.constraint.resource {
                                assert!(elem.1 > 0);
                                found = true;
                                elem.1 += 1;
                            }
                        }
                        if !found {
                            self.train_prev_conflict[train as usize].push((node.constraint.resource, 1));
                        }
                    }

                    forward = node.parent.as_ref();
                } else {
                    let node = backward.unwrap();

                    
                    // Remove constraint
                    f(false, &node.constraint);
                    
                    {
                        let train = node.constraint.train;
                        let mut found = false;
                        for idx in (0..(self.train_prev_conflict[train as usize]).len()).rev() {
                            let elem = &mut self.train_prev_conflict[train as usize][idx];
                            if elem.0 == node.constraint.resource {
                                assert!(elem.1 > 0);
                                found = true;
                                elem.1 -= 1;
                                if elem.1 == 0 {
                                    self.train_prev_conflict[train as usize].remove(idx);
                                }
                            }
                        }
                        assert!(found);
                    }

                    backward = node.parent.as_ref();
                }
            }
        }

        self.current_node = Some(new_node);
    }

    pub fn is_reasonable_constraint(
        &self,
        constr: &ConflictConstraint,
        current_node: Option<&Rc<ConflictSolverNode<T>>>,
    ) -> bool {
        // If this train has less than 3 conflict on the same resource, it is reasonable.
        let res = constr.resource;
        let n_conflicts = *self.train_prev_conflict[constr.train as usize]
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
                        // println!("CYCLE {:?}", current_node);
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
