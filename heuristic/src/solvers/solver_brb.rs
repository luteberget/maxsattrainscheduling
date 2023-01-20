use crate::{
    interval::TimeInterval,
    node_eval::NodeEval,
    occupation::{ResourceConflicts, ResourceOccupation},
    problem::*,
    trainset::TrainSet,
    TrainSolver, TrainSolverStatus,
};
use log::{debug, warn, info};
use std::{
    collections::{BinaryHeap, HashSet},
    rc::Rc,
};

use crate::branching::{Branching, ConflictConstraint, ConflictSolverNode};

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

pub struct ConflictSolver<Train> {
    pub input: Vec<crate::problem::Train>,
    pub trainset: TrainSet<Train>,

    pub conflict_space: Branching<NodeEval>,
    pub queued_nodes: BinaryHeap<Rc<ConflictSolverNode<NodeEval>>>,
    pub conflicts: ResourceConflicts,

    pub ub: i32,
    pub priorities: HashSet<(TrainRef, TrainRef, u32)>,
}

impl<Train: TrainSolver> ConflictSolver<Train> {
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
            return Some(self.current_solution());
        }
        return None;
    }

    pub fn solve_partial_alltrains(&mut self) -> Option<(i32, Vec<Vec<i32>>)> {
        if self.trainset.is_dirty() {
            self.solve_all_trains();
            return None;
        }
        if !self.trainset.is_dirty() && !self.conflicts.has_conflict() {
            return Some(self.current_solution());
        }

        if matches!(self.status(), ConflictSolverStatus::Exhausted) {
            return None;
        }

        self.step();
        if self.trainset.is_dirty() {
            self.solve_all_trains();
        }

        if !self.trainset.is_dirty() && !self.conflicts.has_conflict() {
            return Some(self.current_solution());
        }
        return None;
    }

    pub fn solve_next_stopcb(&mut self, mut stop :impl FnMut() -> bool) -> Option<(i32, Vec<Vec<i32>>)> {
        loop {
            if stop() || matches!(self.status(), ConflictSolverStatus::Exhausted) {
                return None;
            }

            self.step();

            if !self.trainset.is_dirty() && !self.conflicts.has_conflict() {
                return Some(self.current_solution());
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
                return Some(self.current_solution());
            }
        }
    }

    pub fn current_solution(&self) -> (i32, Vec<Vec<TimeValue>>) {
        let mut total_cost = 0;
        let mut out_vec = Vec::new();
        for (cost, times) in self.trainset.trains.iter().map(|t| t.current_solution()) {
            total_cost += cost;
            out_vec.push(times);
        }

        (total_cost, out_vec)
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
            } else if let Some(&dirty_train) = self.trainset.dirty_trains.last() {
                match self.solve_train(&dirty_train) {
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
                let (cost, sol) = self.current_solution();
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

    fn solve_all_trains(&mut self) {
        while let Some(&dirty_train) = self.trainset.dirty_trains.last() {
            let x = self.solve_train(&dirty_train);
            if x == Some(false) {
                println!("Train failed {}", dirty_train);
                break;
            }
        }
    }

    fn solve_train(&mut self, dirty_train: &u32) -> Option<bool> {
        let dirty_train_idx = *dirty_train as usize;
        match self.trainset.trains[dirty_train_idx].status() {
            TrainSolverStatus::Failed => {
                assert!(self.trainset.train_lbs[dirty_train_idx] == 0);
                debug!("Train {} failed.", dirty_train_idx);
                Some(false)
            }
            TrainSolverStatus::Optimal => {
                let prev_cost = self.trainset.train_lbs[dirty_train_idx];
                let train_cost = self.trainset.trains[dirty_train_idx].current_solution().0;
                assert!(train_cost >= self.trainset.train_const_lbs[dirty_train_idx]);
                let new_cost = train_cost - self.trainset.train_const_lbs[dirty_train_idx];
                self.trainset.train_lbs[dirty_train_idx] = new_cost;
                self.trainset.lb += new_cost - prev_cost;

                debug!(
                    "Train {} optimal. train_LB={} problem_LB={}",
                    dirty_train_idx, new_cost, self.trainset.lb
                );
                self.trainset.dirty_trains.pop();
                Some(true)
            }
            TrainSolverStatus::Working => {
                assert!(self.trainset.train_lbs[dirty_train_idx] == 0);

                self.trainset.trains[dirty_train_idx].step(
                    &mut |add, block, resource, interval| {
                        self.conflicts.add_or_remove(
                            add,
                            dirty_train_idx as TrainRef,
                            block,
                            resource,
                            interval,
                        )
                    },
                );

                Self::train_queue_bubble_leftward(
                    &mut self.trainset.dirty_trains,
                    &self.trainset.trains,
                );
                None
            }
        }
    }

    fn train_queue_bubble_leftward(dirty_trains: &mut [u32], trains: &[Train]) {
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

        let (conflict_resource, conflict_occs) = self
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

        let (node_a, node_b) = self.conflict_space.branch(
            (conflict_resource, conflict_occs.0, conflict_occs.1),
            |c| {
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
                );
                (eval, eval)
            },
        );

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
        self.conflict_space.set_node(node, &mut |add, constraint| {
            if add {
                let train = constraint.train;
                self.trainset.trains[train as usize].set_occupied(
                    true,
                    constraint.resource,
                    constraint.enter_after,
                    constraint.exit_before,
                    &mut |a, block, res, i| self.conflicts.add_or_remove(a, train, block, res, i),
                );

                self.trainset.lb -= self.trainset.train_lbs[train as usize];
                self.trainset.train_lbs[train as usize] = 0;
                Self::train_queue_rewinded_train(
                    train,
                    &mut self.trainset.dirty_trains,
                    &self.trainset.trains,
                );
            } else {
                let train = constraint.train;

                // Remove constraint
                self.trainset.trains[train as usize].set_occupied(
                    false,
                    constraint.resource,
                    constraint.enter_after,
                    constraint.exit_before,
                    &mut |a, b, r, i| self.conflicts.add_or_remove(a, train, b, r, i),
                );

                self.trainset.lb -= self.trainset.train_lbs[train as usize];
                self.trainset.train_lbs[train as usize] = 0;

                Self::train_queue_rewinded_train(
                    train,
                    &mut self.trainset.dirty_trains,
                    &self.trainset.trains,
                );
            }
        });
    }

    fn train_queue_rewinded_train(train: TrainRef, dirty_trains: &mut Vec<u32>, trains: &[Train]) {
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

        let original_trains = problem.trains.clone();
        let input = problem.trains.clone();
        let slacks = problem
            .trains
            .iter()
            .map(|train| {
                let mut slacks: Vec<i32> = train
                    .blocks
                    .iter()
                    .map(|b| b.delayed_after.unwrap_or(999999i32))
                    .collect();
                for (block_idx, block) in train.blocks.iter().enumerate().rev() {
                    if slacks[block_idx] == 999999 {
                        slacks[block_idx] = block.minimum_travel_time
                            + block
                                .nexts
                                .iter()
                                .map(|n| slacks[*n as usize])
                                .min()
                                .unwrap_or(999999i32);
                    }
                }
                slacks
            })
            .collect();

        let train_const_lbs: Vec<i32> = problem
            .trains
            .iter()
            .enumerate()
            .map(|(i, t)| {
                let mut t = Train::new(i, t.clone());
                while !matches!(t.status(), TrainSolverStatus::Optimal) {
                    t.step(&mut |_, _, _, _| {});
                }
                t.current_solution().0
            })
            .collect();

        println!("Train LBs {:?}", train_const_lbs);

        let trains: Vec<Train> = problem
            .trains
            .into_iter()
            .enumerate()
            .map(|(i, t)| Train::new(i, t))
            .collect();

        let mut dirty_trains: Vec<u32> = (0..(trains.len() as u32)).collect();
        dirty_trains.sort_by_key(|t| -(trains[*t as usize].current_time() as i64));

        let train_lbs = trains.iter().map(|_| 0).collect();

        let trainset = TrainSet {
            original_trains,
            trains,
            slacks,
            lb: train_const_lbs.iter().sum(),
            train_lbs,
            train_const_lbs,
            dirty_trains,
        };

        Self {
            conflict_space: Branching::new(trainset.trains.len()),
            input,

            trainset,
            queued_nodes: Default::default(),
            ub: i32::MAX,
            priorities: Default::default(),
            conflicts: crate::occupation::ResourceConflicts::empty(problem.n_resources),
        }
    }
}
