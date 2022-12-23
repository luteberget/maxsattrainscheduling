use crate::{
    interval::TimeInterval,
    occupation::{ResourceConflicts, ResourceOccupation},
    problem::*,
};
use log::{debug, warn};
use std::{
    collections::{BinaryHeap, HashSet},
    rc::Rc,
};

#[derive(Debug)]
pub enum ConflictSolverStatus {
    Exhausted,
    SelectNode,
    Conflict,
    SolveTrains,
}

#[derive(Debug)]
pub enum TrainSolverStatus {
    Failed,
    Optimal,
    Working,
}

#[derive(Debug)]
pub struct ConflictConstraint {
    pub train: TrainRef,
    pub other_train: TrainRef,
    pub resource: ResourceRef,
    pub enter_after: TimeValue,
    pub exit_before: TimeValue,
}

#[derive(Copy, Clone)]
pub struct NodeEval {
    a_best: bool,
    node_score: f64,
}

pub struct ConflictSolverNode {
    pub constraint: ConflictConstraint,
    pub depth: u32,
    pub parent: Option<Rc<ConflictSolverNode>>,
    pub eval: NodeEval,
}

impl PartialEq for ConflictSolverNode {
    fn eq(&self, other: &Self) -> bool {
        self.eval.node_score == other.eval.node_score
    }
}

impl PartialOrd for ConflictSolverNode {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.eval.node_score.partial_cmp(&other.eval.node_score)
    }
}

impl Eq for ConflictSolverNode {}

impl Ord for ConflictSolverNode {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.partial_cmp(other).unwrap()
    }
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

pub struct ConflictSolver<Train> {
    pub input: Vec<crate::problem::Train>,
    pub trains: Vec<Train>,
    pub slacks: Vec<Vec<i32>>,
    pub train_lbs: Vec<i32>,
    pub train_const_lbs: Vec<i32>,
    pub lb: i32,
    pub conflicts: ResourceConflicts,
    pub queued_nodes: BinaryHeap<Rc<ConflictSolverNode>>,
    pub prev_conflict: Vec<TinyVec<[(ResourceRef, u32); 16]>>,
    pub current_node: Option<Rc<ConflictSolverNode>>, // The root node is "None".
    pub dirty_trains: Vec<u32>,
    pub total_nodes: usize,
    // pub dirty_train_idxs: Vec<i32>,
    pub n_nodes_created: usize,
    pub n_nodes_explored: usize,
    pub ub: i32,
    pub priorities: HashSet<(TrainRef, TrainRef, u32)>,
}

impl<Train: TrainSolver> ConflictSolver<Train> {
    pub fn status(&self) -> ConflictSolverStatus {
        match (
            self.dirty_trains.is_empty(),
            self.conflicts.conflicting_resource_set.is_empty(),
            self.queued_nodes.is_empty(),
        ) {
            (_, _, true) if self.lb >= self.ub => ConflictSolverStatus::Exhausted,
            (_, false, _) => ConflictSolverStatus::Conflict,
            (false, _, _) => ConflictSolverStatus::SolveTrains,
            (_, _, false) => ConflictSolverStatus::SelectNode,
            (true, true, true) => ConflictSolverStatus::Exhausted,
        }
    }

    pub fn solve_next(&mut self) -> Option<(i32, Vec<Vec<i32>>)> {
        loop {
            match self.status() {
                ConflictSolverStatus::Exhausted => return None,
                _ => self.step(),
            }

            if self.dirty_trains.is_empty() && self.conflicts.conflicting_resource_set.is_empty() {
                return Some(self.current_solution());
            }
        }
    }

    pub fn current_solution(&self) -> (i32, Vec<Vec<TimeValue>>) {
        let mut total_cost = 0;
        let mut out_vec = Vec::new();
        for (cost, times) in self.trains.iter().map(|t| t.current_solution()) {
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
            if self.lb >= self.ub {
                // println!("Node fathomed. {} >= {}", self.lb, self.ub);
                self.switch_to_any_node();
                break;
            } else if !self.conflicts.conflicting_resource_set.is_empty() {
                self.branch();
                break;
            } else if let Some(dirty_train) = self.dirty_trains.last() {
                let dirty_train_idx = *dirty_train as usize;
                match self.trains[dirty_train_idx].status() {
                    TrainSolverStatus::Failed => {
                        assert!(self.train_lbs[dirty_train_idx] == 0);
                        debug!("Train {} failed.", dirty_train_idx);
                        self.switch_to_any_node();
                        break;
                    }
                    TrainSolverStatus::Optimal => {
                        let prev_cost = self.train_lbs[dirty_train_idx];
                        let train_cost = self.trains[dirty_train_idx].current_solution().0;
                        assert!(train_cost >= self.train_const_lbs[dirty_train_idx]);
                        let new_cost = train_cost - self.train_const_lbs[dirty_train_idx];
                        self.train_lbs[dirty_train_idx] = new_cost;
                        self.lb += new_cost - prev_cost;

                        debug!(
                            "Train {} optimal. train_LB={} problem_LB={}",
                            dirty_train_idx, new_cost, self.lb
                        );
                        self.dirty_trains.pop();
                        break;
                    }
                    TrainSolverStatus::Working => {
                        assert!(self.train_lbs[dirty_train_idx] == 0);

                        self.trains[dirty_train_idx].step(&mut |add, block, resource, interval| {
                            self.conflicts.add_or_remove(
                                add,
                                dirty_train_idx as TrainRef,
                                block,
                                resource,
                                interval,
                            )
                        });

                        Self::train_queue_bubble_leftward(&mut self.dirty_trains, &self.trains);
                    }
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
        let mut node = self.current_node.as_ref();
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

    fn node_evaluation(
        &self,
        occ_a: &ResourceOccupation,
        c_a: &ConflictConstraint,
        occ_b: &ResourceOccupation,
        c_b: &ConflictConstraint,
    ) -> NodeEval {
        // Things we might want to know to decide priority of node.

        // How much do the trains overlap.
        let interval_inner = occ_a.interval.intersect(&occ_b.interval);

        // How much time from the first entry to the last exit.
        let interval_outer = occ_a.interval.envelope(&occ_b.interval);

        // How much time do the trains spend in this resource.
        let total_time = occ_a.interval.length() + occ_b.interval.length();

        // Is one interval contained in the other?
        let contained = (occ_a.interval.time_start < occ_b.interval.time_start)
            != (occ_a.interval.time_end < occ_b.interval.time_end);

        let delay_a_to = occ_b.interval.time_end;
        let delay_b_to = occ_a.interval.time_end;

        let slack_a =
            self.slacks[occ_a.train as usize][occ_a.block as usize] - occ_a.interval.time_start;
        let new_slack_a =
            self.slacks[occ_a.train as usize][occ_a.block as usize] - occ_b.interval.time_end;
        let slack_b =
            self.slacks[occ_b.train as usize][occ_b.block as usize] - occ_b.interval.time_start;
        let new_slack_b =
            self.slacks[occ_b.train as usize][occ_b.block as usize] - occ_a.interval.time_end;

        match (
            slack_a >= 0,
            new_slack_a >= 0,
            slack_b >= 0,
            new_slack_b >= 0,
        ) {
            (_, false, _, true) => {
                return NodeEval {
                    a_best: false,
                    node_score: 0.1*interval_outer.length() as f64,
                }
            }
            (_, true, _, false) => {
                return NodeEval {
                    a_best: true,
                    node_score: 0.1*interval_outer.length() as f64,
                }
            },
            _ => {
                return NodeEval {
                    a_best: new_slack_a >= new_slack_b,
                    node_score: interval_outer.length() as f64,
                }
            }
        }

        let mut sum_score = 0.0;
        let mut sum_weight = 0.0;
        let mut sum_product = 0.0;

        // assert!(occ_a.interval.time_start <= occ_b.interval.time_start);

        // Criterion 1:
        // First come, first served
        let (h1_score, h1_weight) = if occ_a.interval.time_start <= occ_b.interval.time_start {
            (
                0.0,
                1.0 - (occ_a.interval.time_end - occ_b.interval.time_start) as f64
                    / interval_outer.length() as f64,
            )
        } else {
            (
                1.0,
                1.0 - (occ_b.interval.time_end - occ_a.interval.time_start) as f64
                    / interval_outer.length() as f64,
            )
        };
        sum_product += h1_score * h1_weight;
        sum_score += h1_score;
        sum_weight += h1_weight;

        // Criterion 2:
        // First leave, first served
        let (h2_score, h2_weight) = if occ_a.interval.time_end < occ_b.interval.time_end {
            (
                0.0,
                1.0 - (occ_a.interval.time_end - occ_b.interval.time_start) as f64
                    / interval_outer.length() as f64,
            )
        } else {
            (
                1.0,
                1.0 - (occ_b.interval.time_end - occ_a.interval.time_start) as f64
                    / interval_outer.length() as f64,
            )
        };
        sum_product += h2_score * h2_weight;
        sum_score += h2_score;
        sum_weight += h2_weight;

        // Criterion 3:
        // Fastest first
        let (h3_score, h3_weight) = if occ_a.interval.length() < occ_b.interval.length() {
            let a_faster =
                (occ_a.interval.length() as f64 / occ_b.interval.length() as f64).powf(1.0 / 3.0);
            let b_pushed = (occ_a.interval.time_end - occ_b.interval.time_start) as f64
                / interval_outer.length() as f64;
            (0.0, 1.0 - a_faster * b_pushed)
        } else {
            let b_faster =
                (occ_b.interval.length() as f64 / occ_a.interval.length() as f64).powf(1.0 / 3.0);
            let a_pushed = (occ_b.interval.time_end - occ_a.interval.time_start) as f64
                / interval_outer.length() as f64;
            (0.0, 1.0 - b_faster * a_pushed)
        };
        sum_product += h3_score * h3_weight;
        sum_score += h3_score;
        sum_weight += h3_weight;

        // Criterion 4:
        // Timetable priority
        // TODO

        // Criterion 5:
        // Best remaining slack

        // Criterion 6:
        // Best remaining transferred slack

        // Measure significance
        // Not normalized.
        let significance = interval_outer.length() as f64;

        // Measure decision controversy
        let choice = sum_product / sum_weight < 0.5;

        // let choice = occ_a.interval.length() < occ_b.interval.length();

        let importance = [
            (h1_score, h1_weight),
            (h2_score, h2_weight),
            (h3_score, h3_weight),
        ]
        .into_iter()
        .filter_map(|(s, w)| ((s > 0.5) != choice).then(|| w * (s - 0.5).abs()))
        .sum::<f64>();

        NodeEval {
            a_best: choice,
            node_score: importance,
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
            exit_before: occ_a.interval.time_end,
        };

        let constraint_b = ConflictConstraint {
            train: occ_b.train,
            other_train: occ_a.train,
            resource: conflict_resource,
            enter_after: occ_a.interval.time_end,
            exit_before: occ_b.interval.time_end,
        };

        debug!(
            "Branch:\n - a: {:?}\n - b: {:?}",
            constraint_a, constraint_b
        );

        let node_eval = self.node_evaluation(&occ_a, &constraint_a, &occ_b, &constraint_b);

        // assert!(node_eval.a_best);

        let prioritized_a =
            self.priorities
                .contains(&(constraint_a.train, constraint_a.other_train, constraint_a.resource));

        let prioritized_b =
            self.priorities
                .contains(&(constraint_b.train, constraint_b.other_train, constraint_b.resource));

        if prioritized_a && prioritized_b {
            println!("BOTH {:?} {:?}", constraint_a, constraint_b);
        }

        let prioritized = prioritized_a ^ prioritized_b;

        let mut chosen_node = None;

        if self.is_reasonable_constraint(&constraint_a, self.current_node.as_ref()) {
            let node_a = ConflictSolverNode {
                constraint: constraint_a,
                depth: self.current_node.as_ref().map_or(0, |n| n.depth) + 1,
                parent: self.current_node.clone(),
                eval: node_eval,
            };

            if prioritized {
                if prioritized_a {
                    chosen_node = Some(Rc::new(node_a));
                } else {
                    self.queued_nodes.push(Rc::new(node_a));
                }
            } else {
                if node_eval.a_best {
                    // Since A is best, we choose to impose the constrain on the other train instead.
                    self.queued_nodes.push(Rc::new(node_a));
                } else {
                    chosen_node = Some(Rc::new(node_a));
                }
            }
            self.n_nodes_created += 1;
        } else {
            warn!("Skipping loop-like constraint {:?}", constraint_a);
        }

        if self.is_reasonable_constraint(&constraint_b, self.current_node.as_ref()) {
            let node_b = ConflictSolverNode {
                constraint: constraint_b,
                depth: self.current_node.as_ref().map_or(0, |n| n.depth) + 1,
                parent: self.current_node.clone(),
                eval: node_eval,
            };

            if prioritized {
                if prioritized_b {
                    chosen_node = Some(Rc::new(node_b));
                } else {
                    self.queued_nodes.push(Rc::new(node_b));
                }
            } else {
                if !node_eval.a_best {
                    self.queued_nodes.push(Rc::new(node_b));
                } else {
                    chosen_node = Some(Rc::new(node_b));
                }
            }
            self.n_nodes_created += 1;
        } else {
            warn!("Skipping loop-like constraint {:?}", constraint_b);
        }

        self.total_nodes += 2;
        // TODO special-case DFS without putting the node on the queue.
        if let Some(n) = chosen_node {
            self.set_node(n);
        } else {
            self.switch_to_any_node();
        }
    }

    fn switch_to_any_node(&mut self) {
        let new_node = match self.queued_nodes.pop() {
            Some(n) => n,
            None => {
                println!("No more nodes. Status: {:?}", self.status());
                assert!(matches!(self.status(), ConflictSolverStatus::Exhausted));
                return;
            }
        };

        self.set_node(new_node);
    }

    fn set_node(&mut self, new_node: Rc<ConflictSolverNode>) {
        self.n_nodes_explored += 1;
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
                        true,
                        node.constraint.resource,
                        node.constraint.enter_after,
                        node.constraint.exit_before,
                        &mut |a, block, res, i| {
                            self.conflicts.add_or_remove(a, train, block, res, i)
                        },
                    );

                    self.lb -= self.train_lbs[train as usize];
                    self.train_lbs[train as usize] = 0;

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

                    debug!("train {} remove", train);
                    self.trains[train as usize].set_occupied(
                        false,
                        node.constraint.resource,
                        node.constraint.enter_after,
                        node.constraint.exit_before,
                        &mut |a, b, r, i| self.conflicts.add_or_remove(a, train, b, r, i),
                    );

                    self.lb -= self.train_lbs[train as usize];
                    self.train_lbs[train as usize] = 0;

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
        let conflicts = crate::occupation::ResourceConflicts::empty(problem.n_resources);

        let mut dirty_trains: Vec<u32> = (0..(trains.len() as u32)).collect();
        dirty_trains.sort_by_key(|t| -(trains[*t as usize].current_time() as i64));

        let prev_conflict = trains.iter().map(|_| Default::default()).collect();
        let train_lbs = trains.iter().map(|_| 0).collect();

        // let priorities: Vec<(u32, u32, bool)> = vec![
        //     (3, 6, 1 == 1),
        //     (5, 6, 0 == 1),
        //     (5, 12, 0 == 1),
        //     (6, 12, 0 == 1),
        //     (4, 13, 1 == 1),
        //     (0, 7, 0 == 1),
        //     (1, 2, 1 == 1),
        //     (1, 4, 1 == 1),
        //     (2, 12, 0 == 1),
        //     (3, 12, 0 == 1),
        //     (11, 12, 0 == 1),
        //     (9, 10, 0 == 1),
        //     (2, 13, 1 == 1),
        //     (3, 4, 1 == 1),
        //     (3, 12, 0 == 1),
        //     (4, 13, 0 == 1),
        //     (0, 7, 0 == 1),
        //     (0, 12, 0 == 1),
        //     (4, 12, 0 == 1),
        //     (3, 4, 1 == 1),
        //     (10, 12, 0 == 1),
        //     (6, 12, 0 == 1),
        //     (7, 12, 0 == 1),
        //     (4, 12, 0 == 1),
        //     (11, 12, 0 == 1),
        //     (3, 12, 1 == 1),
        //     (5, 7, 1 == 1),
        //     (1, 6, 1 == 1),
        //     (0, 7, 0 == 1),
        //     (11, 12, 0 == 1),
        //     (6, 12, 1 == 1),
        //     (0, 7, 0 == 1),
        //     (11, 12, 0 == 1),
        //     (0, 5, 0 == 1),
        //     (3, 4, 0 == 1),
        // ];
        // assert!(priorities.iter().all(|(x, y, _)| x < y));

        // let symm_priorities = priorities.iter().map(|(x, y, b)| (*y, *x, !*b));

        // let priorities: HashSet<(u32, u32, bool)> =
        //     priorities.iter().copied().chain(symm_priorities).collect();

        Self {
            input,
            trains,
            slacks,
            lb: train_const_lbs.iter().sum(),
            train_lbs,
            train_const_lbs,
            conflicts,
            current_node: None,
            dirty_trains,
            queued_nodes: Default::default(),
            total_nodes: 0,
            prev_conflict,

            n_nodes_created: 0,
            n_nodes_explored: 0,
            ub: i32::MAX,
            priorities: Default::default(),
        }
    }

    pub fn is_reasonable_constraint(
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
