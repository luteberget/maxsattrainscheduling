use std::{
    collections::{BTreeMap, HashMap},
    rc::Rc,
};

use itertools::Itertools;
use log::warn;

use crate::{
    branching::{Branching, ConflictSolverNode},
    node_eval::NodeEval,
    occupation::ResourceConflicts,
    problem::{BlockRef, Problem, TrainRef},
    trainset::TrainSet,
    ConflictSolver,
};

use super::train_queue::QueueTrainSolver;

pub struct HeurHeur2 {
    pub input: Vec<crate::problem::Train>,
    pub trainset: TrainSet<QueueTrainSolver>,

    pub conflict_space: Branching<NodeEval>,
    pub conflicts: ResourceConflicts,

    best_cost: i32,
    visit_weights: Option<BTreeMap<(TrainRef, BlockRef), f32>>,

    queue_main: Vec<Rc<ConflictSolverNode<NodeEval>>>,
    queue_dive: Vec<Rc<ConflictSolverNode<NodeEval>>>,
}

impl HeurHeur2 {
    pub fn new(problem: Problem) -> Self {
        let n_resources = problem.n_resources;
        let input = problem.trains.clone();
        let trainset = TrainSet::new(problem);
        let mut h = Self {
            conflict_space: Branching::new(trainset.trains.len()),
            input,
            trainset,
            conflicts: crate::occupation::ResourceConflicts::empty(n_resources),
            queue_main: Default::default(),
            queue_dive: Default::default(),
            best_cost: i32::MAX,
            visit_weights: None,
        };

        h.dive(None);
        h
    }

    fn solution_event(&mut self) -> i32 {
        let (cost, _) = self.trainset.current_solution();

        if cost < self.best_cost {
            self.best_cost = cost;
            self.visit_weights = Some(self.create_visit_weights());
        }
        cost
    }

    fn create_visit_weights(&self) -> BTreeMap<(TrainRef, BlockRef), f32> {
        // Extract the conflicts in order.
        let mut conflicts = Vec::new();
        let mut current_node = &self.conflict_space.current_node;
        while let Some(node) = current_node {
            conflicts.push(&node.constraint);
            current_node = &node.parent;
        }

        let mut trains: Vec<Vec<(BlockRef, f32)>> = (0..self.trainset.trains.len())
            .map(|_| Vec::new())
            .collect();

        for c in conflicts {
            for (t, b) in [(c.train, c.block), (c.other_train, c.other_block)] {
                let prev_train_weight = trains[t as usize].last().map(|(_, w)| *w).unwrap_or(0.);
                trains[t as usize].push((b, 0.75 * prev_train_weight + 1.));
            }
        }

        let mut map = BTreeMap::new();
        for (train_idx, train) in trains.into_iter().enumerate() {
            for (block, weight) in train {
                map.insert((train_idx as TrainRef, block), weight);
            }
        }

        println!("New weights");
        for ((x, y), z) in map.iter() {
            println!(" {} {} {}", x, y, z);
        }

        map
    }

    fn dive(&mut self, from_node: Option<&Rc<ConflictSolverNode<NodeEval>>>) -> i32 {
        let prev_node = self.conflict_space.current_node.clone();
        self.set_node(from_node.cloned());
        self.queue_dive.clear();
        loop {
            if let Some(conflict) = self.conflicts.first_conflict() {
                let (node_a, node_b) = self.conflict_space.branch(conflict, |c| {
                    let eval = crate::node_eval::node_evaluation(
                        &self.trainset.original_trains,
                        &self.trainset.slacks,
                        c.occ_a,
                        c.c_a,
                        c.occ_b,
                        c.c_b,
                        self.visit_weights.as_ref(),
                        false,
                    );
                    (eval, eval)
                });

                let mut chosen_node = None;
                if let Some(node_a) = node_a {
                    if node_a.eval.a_best {
                        self.queue_dive.push(node_a);
                    } else {
                        chosen_node = Some(node_a);
                    }
                }

                if let Some(node_b) = node_b {
                    if !node_b.eval.a_best {
                        self.queue_dive.push(node_b);
                    } else {
                        chosen_node = Some(node_b);
                    }
                }

                if let Some(chosen_node) = chosen_node {
                    self.set_node(Some(chosen_node));
                } else {
                    let node = Some(self.queue_dive.pop().unwrap());
                    self.set_node(node);
                }
            } else if self.trainset.is_dirty() {
                self.trainset.solve_all_trains(&mut self.conflicts);
            } else {
                let cost = self.solution_event();
                self.set_node(prev_node);
                return cost;
            }
        }
    }

    pub fn solve_step(&mut self) -> Option<Option<(i32, Vec<Vec<i32>>)>> {
        if let Some(conflict) = self.conflicts.first_conflict() {
            let (node_a, node_b) = self.conflict_space.branch(conflict, |c| {
                let eval = crate::node_eval::node_evaluation(
                    &self.trainset.original_trains,
                    &self.trainset.slacks,
                    c.occ_a,
                    c.c_a,
                    c.occ_b,
                    c.c_b,
                    self.visit_weights.as_ref(),
                    true,
                );
                (eval, eval)
            });
            // Evaluate each node using a dive.

            let score_a = node_a
                .as_ref()
                .map(|n| self.dive(Some(n)))
                .unwrap_or(i32::MAX);
            let score_b = node_b
                .as_ref()
                .map(|n| self.dive(Some(n)))
                .unwrap_or(i32::MAX);

            println!("Heuristic dive calculated");
            println!("  node_a:{}", score_a);
            println!("  node_b:{}", score_b);

            let mut chosen_node = None;

            if let Some(node_a) = node_a {
                if score_a < score_b {
                    chosen_node = Some(node_a);
                } else {
                    self.queue_main.push(node_a);
                }
            }
            if let Some(node_b) = node_b {
                if score_b < score_a {
                    chosen_node = Some(node_b);
                } else {
                    self.queue_main.push(node_b);
                }
            }

            if let Some(node) = chosen_node {
                self.set_node(Some(node));
            } else {
                self.switch_to_any_node();
            }

            self.trainset.solve_all_trains(&mut self.conflicts);

            None
        } else if self.trainset.is_dirty() {
            // Solve trains until conflict or done.

            self.trainset.solve_all_trains(&mut self.conflicts);

            None
        } else {
            // Done.
            Some(Some(self.trainset.current_solution()))
        }
    }

    fn set_node(&mut self, node: Option<Rc<ConflictSolverNode<NodeEval>>>) {
        self.conflict_space.set_node(node, &mut |add, constraint| {
            self.trainset
                .add_remove_constraint(add, constraint, &mut self.conflicts)
        });
    }

    fn switch_to_any_node(&mut self) {
        println!("switch to any");
        warn!("non-DFS node");
        if self.queue_main.is_empty() {
            println!("No more nodes");
        } else {
            let new_node = self
                .queue_main
                .remove(rand::random::<usize>() % self.queue_main.len());
            self.set_node(Some(new_node));
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

impl ConflictSolver for HeurHeur2 {
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

    fn visit_weights(&self) -> Option<&BTreeMap<(TrainRef, BlockRef), f32>> {
        self.visit_weights.as_ref()
    }
}
