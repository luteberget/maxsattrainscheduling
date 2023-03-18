#![allow(unused)]

use rand::prelude::*;
use std::{collections::HashMap, rc::Rc};

use super::train_queue::QueueTrainSolver;
use crate::{
    branching::{Branching, ConflictSolverNode},
    node_eval::NodeEval,
    occupation::ResourceConflicts,
    problem::{self, TrainRef},
    trainset::TrainSet,
};

struct Node {
    constraint: Option<Rc<ConflictSolverNode<NodeEval>>>,
    children: Option<(usize, usize)>,
    score: NodeScore,
}

impl Node {
    pub fn new(constraint: Option<Rc<ConflictSolverNode<NodeEval>>>) -> Self {
        Node {
            constraint,
            children: None,
            score: NodeScore::Approximate {
                visit_count: 0,
                total_cost: 0,
            },
        }
    }
}

enum NodeScore {
    Closed(i32),
    Approximate { visit_count: usize, total_cost: i64 },
}

#[derive(Debug)]
pub enum MCTSError {
    NoSolutionFound,
}

pub type Solution = (i32, Vec<Vec<i32>>);

#[derive(PartialEq, PartialOrd, Ord, Eq, Hash)]
struct MoveCode {
    first_train: TrainRef,
    last_train: TrainRef,
}

#[derive(PartialEq, PartialOrd, Ord, Eq, Hash)]
struct SimpleMoveCode {
    train: TrainRef,
    important: bool,
}

pub fn solve(
    problem: problem::Problem,
    mut terminate: impl FnMut() -> bool,
) -> Result<Solution, MCTSError> {
    let mut nodes: Vec<Node> = Vec::new();

    let mut orig_trains = problem.trains.clone();
    let mut conflicts = ResourceConflicts::empty(problem.n_resources);
    let mut trainset: TrainSet<QueueTrainSolver> = TrainSet::new(problem);
    let mut branching: Branching<NodeEval> = Branching::new(orig_trains.len());
    let mut best: Option<(i32, Vec<Vec<i32>>)> = None;
    let mut rng = thread_rng();
    let mut n_rollouts = 0usize;

    let mut simple_policy: HashMap<SimpleMoveCode, (i64, usize)> = HashMap::new();
    let mut policy: HashMap<MoveCode, (i64, usize)> = HashMap::new();

    // Add root node
    nodes.push(Node::new(None));

    'mcts: loop {
        if terminate() || matches!(nodes[0].score, NodeScore::Closed(_)) {
            println!(
                "MCTS UCT finised with {} nodes and {} rollouts",
                nodes.len(),
                n_rollouts
            );
            return best.ok_or(MCTSError::NoSolutionFound);
        }

        // Dive through the tree by max(uct).
        // TODO wasn't this supposed to be sampling-based?
        let mut path: Vec<usize> = vec![];
        let mut node_idx = 0;
        'selection: loop {
            path.push(node_idx);
            if let Some((left_idx, right_idx)) = nodes[node_idx].children {
                node_idx = match (&nodes[left_idx].score, &nodes[right_idx].score) {
                    (
                        &NodeScore::Approximate {
                            visit_count: left_count,
                            total_cost: left_cost,
                        },
                        &NodeScore::Approximate {
                            visit_count: right_count,
                            total_cost: right_cost,
                        },
                    ) => {
                        let left_constraint =
                            &nodes[left_idx].constraint.as_ref().unwrap().constraint;
                        let right_constraint =
                            &nodes[right_idx].constraint.as_ref().unwrap().constraint;

                        // assert!(left_constraint.train == right_constraint.other_train);
                        // assert!(right_constraint.train == left_constraint.other_train);

                        let (mean_left, mean_right) = policy_value(
                            &policy,
                            &simple_policy,
                            Some(left_constraint),
                            Some(right_constraint),
                        );

                        let total_count = left_count as f64 + right_count as f64;
                        let log = (total_count).log10();
                        let left_better = uct(
                            left_cost as f64,
                            left_count as f64,
                            log,
                            total_count,
                            mean_left,
                        ) >= uct(
                            right_cost as f64,
                            right_count as f64,
                            log,
                            total_count,
                            mean_right,
                        );

                        // print!("test  1/0={} 0/0={} " , 1.0f64/0.0f64, 0.0f64/0.0f64);
                        if left_better {
                            // println!(
                            //     "  Left best {}/{}={} over {}/{}={} (log {})",
                            //     left_cost,
                            //     left_count,
                            //     (left_cost as f64) / (left_count as f64),
                            //     right_cost,
                            //     right_count,
                            //     (right_cost as f64) / (right_cost as f64),
                            //     log
                            // );
                            left_idx
                        } else {
                            // println!(
                            //     " Right best {}/{}={} over {}/{}={} (log {})",
                            //     right_cost,
                            //     right_count,
                            //     (right_cost as f64) / (right_count as f64),
                            //     left_cost,
                            //     left_count,
                            //     (left_cost as f64) / (left_count as f64),
                            //     log
                            // );
                            right_idx
                        }
                    }
                    (NodeScore::Closed(_), NodeScore::Approximate { .. }) => right_idx,
                    (NodeScore::Approximate { .. }, NodeScore::Closed(_)) => left_idx,
                    (NodeScore::Closed(_), NodeScore::Closed(_)) => panic!(),
                }
            } else {
                break 'selection;
            }
        }

        // Activate the constraints at this node.
        branching.set_node(nodes[node_idx].constraint.clone(), &mut |a, c| {
            trainset.add_remove_constraint(a, c, &mut conflicts)
        });

        // Expand the node
        trainset.solve_all_trains(&mut conflicts);

        let mut is_terminal = true;
        let mut is_stuck = false;

        if let Some(conflict) = conflicts.first_conflict() {
            // Create the branching constraints
            let (node_a, node_b) = branching.branch(conflict, |c| {
                let eval = crate::node_eval::node_evaluation(
                    &trainset.original_trains,
                    &trainset.slacks,
                    c.occ_a,
                    c.c_a,
                    c.occ_b,
                    c.c_b,
                    None,
                    false,
                );
                (eval, eval)
            });

            match (node_a, node_b) {
                (None, None) => {
                    // couldn't branch even though there is a conflict. Deadlock?
                    is_stuck = true;
                    is_terminal = true;
                }
                (Some(node_a), Some(node_b)) => {
                    // Normal case. We make two new children.
                    let left_idx = nodes.len();
                    nodes.push(Node::new(Some(node_a)));
                    let right_idx = nodes.len();
                    nodes.push(Node::new(Some(node_b)));
                    nodes[node_idx].children = Some((left_idx, right_idx));
                    is_terminal = false;
                }
                (None, Some(single_node)) | (Some(single_node), None) => {
                    // Only one of the choices for the conflict are reasonable
                    // Just replace this node with the new constraint.
                    nodes[node_idx].constraint = Some(single_node);
                    is_terminal = false;
                }
            }
        }

        // Rollout phase

        const N_ROLLOUTS: usize = 10;
        for _ in 0..N_ROLLOUTS {
            if is_terminal {
                break;
            }

            n_rollouts += 1;
            // Activate the constraints at this node.
            branching.set_node(nodes[node_idx].constraint.clone(), &mut |a, c| {
                trainset.add_remove_constraint(a, c, &mut conflicts)
            });

            // Dive with random heuristic
            let mut queue_dive = Vec::new();
            'rollout: loop {
                trainset.solve_all_trains(&mut conflicts);
                if let Some(conflict) = conflicts.first_conflict() {
                    let (node_a, node_b) = branching.branch(conflict, |c| {
                        let eval = crate::node_eval::node_evaluation(
                            &trainset.original_trains,
                            &trainset.slacks,
                            c.occ_a,
                            c.c_a,
                            c.occ_b,
                            c.c_b,
                            None,
                            false,
                        );
                        (eval, eval)
                    });

                    let greedy_choice = node_a.as_ref().map(|n| n.eval.a_score).unwrap_or(0.5);

                    let (mean_left, mean_right) = policy_value(
                        &policy,
                        &simple_policy,
                        node_a.as_ref().map(|n| &n.constraint),
                        node_b.as_ref().map(|n| &n.constraint),
                    );

                    let policy_choice = match (mean_left.is_infinite(), mean_right.is_infinite()) {
                        (true, true) => 0.5,
                        (true, false) => 0.0,
                        (false, true) => 1.0,
                        (false, false) => {
                            if mean_left < mean_right {
                                0.05
                            } else {
                                0.95
                            }
                        }
                    };

                    let greedy_weight = (1.0 - 2.0 * greedy_choice).abs();

                    let p = (greedy_weight * greedy_choice + (1.0 - greedy_weight) * policy_choice)
                        .clamp(0.05, 0.95);
                    let choice = rng.gen_bool(p);

                    let mut chosen_node = None;

                    if let Some(node_a) = node_a {
                        if !choice {
                            chosen_node = Some(node_a);
                        } else {
                            queue_dive.push(node_a);
                        }
                    }

                    if let Some(node_b) = node_b {
                        if choice {
                            chosen_node = Some(node_b);
                        } else {
                            queue_dive.push(node_b);
                        }
                    }

                    let new_node = if let Some(new_node) = chosen_node {
                        new_node
                    } else if let Some(queued_node) = queue_dive.pop() {
                        queued_node
                    } else {
                        // Deadlocked (even with backtracking)
                        is_terminal = true;
                        is_stuck = true;
                        break 'rollout;
                    };
                    branching.set_node(Some(new_node), &mut |a, c| {
                        trainset.add_remove_constraint(a, c, &mut conflicts)
                    });
                } else {
                    break 'rollout;
                }
            }

            // Check terminal again
            if is_terminal {
                break;
            }

            // We should now be at a solution.
            assert!(conflicts.first_conflict().is_none());
            let (cost, sol) = trainset.current_solution();
            if best.is_none() || best.as_ref().unwrap().0 > cost {
                println!("new best in rollout {}", cost);
                best = Some((cost, sol));
            }

            // Update tree statistics
            for node_idx in path.iter().rev() {
                match &mut nodes[*node_idx].score {
                    NodeScore::Approximate {
                        visit_count,
                        total_cost,
                    } => {
                        *visit_count += 1;
                        *total_cost += cost as i64;
                    }
                    NodeScore::Closed(_) => panic!(),
                }
            }

            // Update policy
            // These are calculated from the nodes of the branching tree, not the MC tree.
            let mut curr = branching.current_node.as_ref();
            while let Some(conflict_node) = curr {
                let move_code = MoveCode {
                    first_train: conflict_node.constraint.train,
                    last_train: conflict_node.constraint.other_train,
                };

                let policy_entry = policy.entry(move_code).or_insert((0, 0));
                policy_entry.0 += cost as i64;
                policy_entry.1 += 1;

                let simple_policy_entry = simple_policy
                    .entry(SimpleMoveCode {
                        train: conflict_node.constraint.other_train,
                        important: true,
                    })
                    .or_insert((0, 0));

                simple_policy_entry.0 += cost as i64;
                simple_policy_entry.1 += 1;

                let simple_policy_entry = simple_policy
                    .entry(SimpleMoveCode {
                        train: conflict_node.constraint.train,
                        important: false,
                    })
                    .or_insert((0, 0));

                simple_policy_entry.0 += cost as i64;
                simple_policy_entry.1 += 1;

                curr = conflict_node.parent.as_ref();
            }
        }

        if is_terminal {
            // This is a terminal node.

            if is_stuck {
                nodes[node_idx].score = NodeScore::Closed(i32::MAX);
            } else {
                let (cost, sol) = trainset.current_solution();
                if best.is_none() || best.as_ref().unwrap().0 > cost {
                    best = Some((cost, sol));
                    println!("new best in terminal {}", cost);
                }
                nodes[node_idx].score = NodeScore::Closed(cost);
            }

            // Close nodes back up the tree
            // TODO maybe we actually need dynamic garbage collection because of closing nodes.
            path.pop();
            'closeup: while let Some(ancestor_idx) = path.pop() {
                let (l, r) = nodes[ancestor_idx].children.unwrap();
                match (&nodes[l].score, &nodes[r].score) {
                    (&NodeScore::Closed(sl), &NodeScore::Closed(sr)) => {
                        nodes[ancestor_idx].score = NodeScore::Closed(sl.min(sr));
                    }
                    _ => break 'closeup,
                };
            }
        }
    }
}

fn policy_value(
    policy: &HashMap<MoveCode, (i64, usize)>,
    simple_policy: &HashMap<SimpleMoveCode, (i64, usize)>,
    left_constraint: Option<&crate::branching::ConflictConstraint>,
    right_constraint: Option<&crate::branching::ConflictConstraint>,
) -> (f64, f64) {
    let left_policy = left_constraint
        .and_then(|c| {
            policy.get(&MoveCode {
                first_train: c.train,
                last_train: c.other_train,
            })
        })
        .unwrap_or(&(0, 0));
    let right_policy = right_constraint
        .and_then(|c| {
            policy.get(&MoveCode {
                first_train: c.train,
                last_train: c.other_train,
            })
        })
        .unwrap_or(&(0, 0));

    let mean_left = if left_policy.1 == 0 {
        f64::INFINITY
    } else {
        left_policy.0 as f64 / left_policy.1 as f64
    };
    let mean_right = if right_policy.1 == 0 {
        f64::INFINITY
    } else {
        right_policy.0 as f64 / right_policy.1 as f64
    };

    let simple_left_policy_important = left_constraint
        .and_then(|c| {
            simple_policy.get(&SimpleMoveCode {
                train: c.other_train,
                important: true,
            })
        })
        .unwrap_or(&(0, 0));
    let simple_left_policy_not_important = left_constraint
        .and_then(|c| {
            simple_policy.get(&SimpleMoveCode {
                train: c.train,
                important: false,
            })
        })
        .unwrap_or(&(0, 0));
    let simple_right_policy_important = right_constraint
        .and_then(|c| {
            simple_policy.get(&SimpleMoveCode {
                train: c.other_train,
                important: true,
            })
        })
        .unwrap_or(&(0, 0));
    let simple_right_policy_not_important = right_constraint
        .and_then(|c| {
            simple_policy.get(&SimpleMoveCode {
                train: c.train,
                important: false,
            })
        })
        .unwrap_or(&(0, 0));

    let simple_mean_left =
        if simple_left_policy_important.1 + simple_right_policy_not_important.1 == 0 {
            f64::INFINITY
        } else {
            simple_left_policy_important.0 as f64
                + simple_right_policy_not_important.0 as f64
                    / (simple_left_policy_important.1 as f64
                        + simple_right_policy_not_important.1 as f64)
        };

    let simple_mean_right =
        if simple_right_policy_important.1 + simple_left_policy_not_important.1 == 0 {
            f64::INFINITY
        } else {
            simple_right_policy_important.0 as f64
                + simple_left_policy_not_important.0 as f64
                    / (simple_right_policy_important.1 as f64
                        + simple_left_policy_not_important.1 as f64)
        };

    const K: f64 = 100.0;
    let beta_count = left_policy.1 as f64 + right_policy.1 as f64;
    let beta = (K / (3.0 * beta_count + K)).sqrt();

    let beta = 1.0 - ((left_policy.1 as f64 + right_policy.1 as f64) / 20.0).clamp(0., 1.);

    // const K: f64 = 5.0;
    // let beta = (K / (3.0 * (left_policy.1 as f64 + right_policy.1 as f64) + K)).sqrt();

    (
        (1.0 - beta) * mean_left + (beta) * simple_mean_left,
        (1.0 - beta) * mean_right + (beta) * simple_mean_right,
    )
}

fn uct(cost: f64, count: f64, log: f64, total_count: f64, movecode_mean: f64) -> f64 {
    if count == 0.0 {
        f64::INFINITY
    } else {
        const K: f64 = 250.0;
        let beta = (K / (3.0 * total_count + K)).sqrt();
        // let beta = 0.0;
        (1.0 - beta) * (-cost / count)
            + beta * (-movecode_mean)
            + 2.0f64.sqrt() * 60000.0 * (log / count).sqrt()
    }
}
