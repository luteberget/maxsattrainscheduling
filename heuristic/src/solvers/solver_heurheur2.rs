use std::{
    collections::{BTreeMap, HashMap},
    rc::Rc,
};

use itertools::Itertools;
use log::warn;
use ordered_float::OrderedFloat;

use crate::{
    branching::{Branching, ConflictSolverNode},
    node_eval::NodeEval,
    occupation::ResourceConflicts,
    problem::{BlockRef, Problem, TimeValue, TrainRef},
    trainset::TrainSet,
    ConflictSolver, TrainSolver,
};

use super::train_queue::QueueTrainSolver;

pub struct HeurHeur2 {
    pub input: Vec<crate::problem::Train>,
    pub trainset: TrainSet<QueueTrainSolver>,

    pub conflict_space: Branching<NodeEval>,
    pub conflicts: ResourceConflicts,

    best_cost: i32,
    visit_weights: Option<BTreeMap<(TrainRef, BlockRef), f32>>,

    pub queue_main: Vec<(f64, Rc<ConflictSolverNode<NodeEval>>)>,
    pub queue_dive: Vec<Rc<ConflictSolverNode<NodeEval>>>,
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
        println!("ROOT heuristic cost {}", h.best_cost);
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

    pub fn solve_next_stopcb(
        &mut self,
        mut stop: impl FnMut() -> bool,
    ) -> Option<(i32, Vec<Vec<i32>>)> {
        loop {
            let done = !self.conflicts.has_conflict()
                && !self.trainset.is_dirty()
                && self.queue_main.is_empty();
            if stop() || done {
                return None;
            }

            if let Some((cost, sol)) = self.big_step() {
                return Some((cost, sol));
            }
        }
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

        // println!("New weights");
        // for ((x, y), z) in map.iter() {
        //     println!(" {} {} {}", x, y, z);
        // }

        map
    }

    fn dive(&mut self, from_node: Option<&Rc<ConflictSolverNode<NodeEval>>>) -> (i32, Vec<i32>) {
        let prev_node = self.conflict_space.current_node.clone();
        self.set_node(from_node.cloned());
        self.queue_dive.clear();
        let mut heur_nodes = 0;
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
                    if self.queue_dive.is_empty() || heur_nodes > 10000 {
                        return (i32::MAX, vec![]);
                    }
                    let n = self.queue_dive.pop().unwrap();
                    self.set_node(Some(n));
                }
                heur_nodes += 1;
            } else if self.trainset.is_dirty() {
                self.trainset.solve_all_trains(&mut self.conflicts);
            } else {
                let cost = self.solution_event();
                let cost_vector = self
                    .trainset
                    .trains
                    .iter()
                    .map(|t| t.current_solution().0)
                    .collect();

                self.set_node(prev_node);
                return (cost, cost_vector);
            }
        }
    }

    pub fn solve_step(&mut self) -> Option<Option<(i32, Vec<Vec<i32>>)>> {
        if let Some(conflict) = self.conflicts.first_conflict() {
            // println!(
            //     "Train A: {}  Train B: {}",
            //     conflict.occ_a.train, conflict.occ_b.train
            // );

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
            // Evaluate each node using a dive.

            let (score_a, costs_a) = node_a
                .as_ref()
                .map(|n| self.dive(Some(n)))
                .unwrap_or((i32::MAX, vec![]));
            let (score_b, costs_b) = node_b
                .as_ref()
                .map(|n| self.dive(Some(n)))
                .unwrap_or((i32::MAX, vec![]));

            let cost_diff = costs_a
                .iter()
                .zip(costs_b.iter())
                .map(|(a, b)| (a - b).abs())
                .sum::<TimeValue>();
            let criticality = (cost_diff as f64) / ((score_a - score_b).abs() as f64);

            println!("Heuristic dive calculated");
            println!("  node_a:{}   costs {:?}", score_a, costs_a);
            println!("  node_b:{}   costs {:?}", score_b, costs_b);
            println!(
                "  total cost diff {}, train cost diff {}",
                (score_a - score_b).abs(),
                cost_diff
            );
            println!(" criticality? {}", criticality);

            let mut chosen_node = None;

            if let Some(node_a) = node_a {
                if score_a < score_b {
                    chosen_node = Some(node_a);
                } else {
                    println!("pushed a {}", score_a as f64- cost_diff as f64);
                    self.queue_main.push((score_a as f64- cost_diff as f64, node_a));
                }
            }
            if let Some(node_b) = node_b {
                if score_b < score_a {
                    chosen_node = Some(node_b);
                } else {
                    println!("pushed b {}", score_b as f64- cost_diff as f64);
                    self.queue_main.push((score_b as f64 - cost_diff as f64, node_b));
                }
            }

            if let Some(node) = chosen_node {
                self.set_node(Some(node));
            } else {
                if self.queue_main.is_empty() {
                    return Some(None);
                }
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
            let sol = self.trainset.current_solution();
            self.switch_to_any_node();
            Some(Some(sol))
        }
    }

    fn set_node(&mut self, node: Option<Rc<ConflictSolverNode<NodeEval>>>) {
        self.conflict_space.set_node(node, &mut |add, constraint| {
            self.trainset
                .add_remove_constraint(add, constraint, &mut self.conflicts)
        });
    }

    fn switch_to_any_node(&mut self) {
        // println!("switch to any");
        warn!("non-DFS node");
        if self.queue_main.is_empty() {
            // println!("No more nodes");
        } else {
            self.queue_main.sort_by_key(|(crit, _)| OrderedFloat(*crit));
            let (crit, new_node) = self.queue_main.pop().unwrap();
            // println!("Popping node with criticality {}", crit);
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
