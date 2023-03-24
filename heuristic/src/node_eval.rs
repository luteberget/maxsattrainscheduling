use std::collections::BTreeMap;

use log::error;

use crate::{
    branching::ConflictConstraint,
    occupation::ResourceOccupation,
    problem::{BlockRef, TrainRef},
};

#[derive(Debug, Copy, Clone, Default)]
pub struct NodeEval {
    pub a_best: bool,
    pub node_score: f64,
}

#[derive(Debug)]
pub enum LocalHeuristicType {
    FirstEntering,
    FirstLeaving,
    FastestFirst,
    TimetableFirst,
    CriticalDependencies,
}

pub fn node_evaluation(
    trains: &Vec<crate::problem::Train>,
    slacks: &Vec<Vec<i32>>,
    occ_a: &ResourceOccupation,
    c_a: &ConflictConstraint,
    occ_b: &ResourceOccupation,
    c_b: &ConflictConstraint,
    weights: Option<&BTreeMap<(TrainRef, BlockRef), f32>>,
    print: bool,
) -> NodeEval {
    let mut priorities: Vec<(LocalHeuristicType, f64, f64)> = Vec::new();
    priorities.push((
        LocalHeuristicType::FirstEntering,
        first_entering(trains, slacks, occ_a, c_a, occ_b, c_b),
        1.0,
    ));
    priorities.push((
        LocalHeuristicType::FirstLeaving,
        first_leaving(trains, slacks, occ_a, c_a, occ_b, c_b),
        1.0,
    ));

    // Fastest first doesn't seem like a good heuristic.
    priorities.push((
        LocalHeuristicType::FastestFirst,
        fastest_first(trains, slacks, occ_a, c_a, occ_b, c_b),
        3.0,
    ));

    // priorities.push((
    //     LocalHeuristicType::TimetableFirst,
    //     timetable_first(trains, slacks, occ_a, c_a, occ_b, c_b),
    //     1.0,
    // ));

    // priorities.push((
    //     LocalHeuristicType::CriticalDependencies,
    //     critical_dependencies(trains, slacks, occ_a, c_a, occ_b, c_b, weights),
    //     3.0,
    // ));

    if print {
        println!("NODE EVAL");
        for (pri_type, pri_value, factor) in &priorities {
            assert!(*pri_value >= 0.0 - 1.0e-5 && *pri_value <= 1.0 + 1.0e-5);
            println!(" -  {:?} * {}", (pri_type, pri_value), factor);
        }
    }

    let a_best = priorities.iter().map(|(_, v, f)| (*f) * (*v)).sum::<f64>()
        / priorities.iter().map(|(_, _, f)| (*f)).sum::<f64>()
        < 0.5;
    let significance = if a_best {
        // how much is B delayed
        c_b.enter_after - occ_b.interval.time_start
    } else {
        c_a.enter_after - occ_a.interval.time_start
    };
    assert!(significance > 0);
    let significance = significance as f64;

    let mut uncertainty = 0.0;
    for i in 0..priorities.len() {
        for j in (i + 1)..priorities.len() {
            let a = priorities[i].1;
            let b = priorities[j].1;
            if a > 0.5 && b < 0.5 || a < 0.5 && b > 0.5 {
                uncertainty += (a - b) * (a - b);
            }
        }
    }

    // dbg!(
    NodeEval {
        a_best,
        node_score: uncertainty * significance,
    }
    // )
}

fn first_entering(
    _trains: &Vec<crate::problem::Train>,

    _slacks: &Vec<Vec<i32>>,
    occ_a: &ResourceOccupation,
    _c_a: &ConflictConstraint,
    occ_b: &ResourceOccupation,
    _c_b: &ConflictConstraint,
) -> f64 {
    if occ_a.interval.time_start < occ_b.interval.time_start {
        0.5 - 0.5 * (occ_b.interval.time_start - occ_a.interval.time_start) as f64
            / occ_a.interval.length() as f64
    } else {
        0.5 + 0.5 * (occ_a.interval.time_start - occ_b.interval.time_start) as f64
            / occ_b.interval.length() as f64
    }
}

fn first_leaving(
    _trains: &Vec<crate::problem::Train>,

    _slacks: &Vec<Vec<i32>>,
    occ_a: &ResourceOccupation,
    _c_a: &ConflictConstraint,
    occ_b: &ResourceOccupation,
    _c_b: &ConflictConstraint,
) -> f64 {
    if occ_a.interval.time_end < occ_b.interval.time_end {
        0.5 - 0.5 * (occ_b.interval.time_end - occ_a.interval.time_end) as f64
            / occ_b.interval.length() as f64
    } else {
        0.5 + 0.5 * (occ_a.interval.time_end - occ_b.interval.time_end) as f64
            / occ_a.interval.length() as f64
    }
}

fn fastest_first(
    _trains: &Vec<crate::problem::Train>,

    _slacks: &Vec<Vec<i32>>,
    occ_a: &ResourceOccupation,
    _c_a: &ConflictConstraint,
    occ_b: &ResourceOccupation,
    _c_b: &ConflictConstraint,
) -> f64 {

    let factor = occ_a.interval.length() as f64/ occ_b.interval.length() as f64;
    if factor > 0.2 && factor < 5.0 {
        return 0.5;
    }

    if occ_a.interval.length() < occ_b.interval.length() {
        0.5 - 0.5 * (1.0 - occ_a.interval.length() as f64 / occ_b.interval.length() as f64)
    } else {
        0.5 + 0.5 * (1.0 - occ_b.interval.length() as f64 / occ_a.interval.length() as f64)
    }
}

fn timetable_first(
    trains: &Vec<crate::problem::Train>,

    _slacks: &Vec<Vec<i32>>,
    occ_a: &ResourceOccupation,
    _c_a: &ConflictConstraint,
    occ_b: &ResourceOccupation,
    _c_b: &ConflictConstraint,
) -> f64 {
    let tt_a = trains[occ_a.train as usize].blocks[occ_a.block as usize].earliest_start;
    let tt_b = trains[occ_b.train as usize].blocks[occ_b.block as usize].earliest_start;
    let x = (tt_a - tt_b) as f64;
    let slope: f64 = 240.0;

    1.0 / (1.0 + f64::exp(-x / slope))
}

fn critical_dependencies(
    trains: &Vec<crate::problem::Train>,

    _slacks: &Vec<Vec<i32>>,
    occ_a: &ResourceOccupation,
    _c_a: &ConflictConstraint,
    occ_b: &ResourceOccupation,
    _c_b: &ConflictConstraint,
    weights: Option<&BTreeMap<(TrainRef, BlockRef), f32>>,
) -> f64 {
    let mut same_dir = false;
    'res: for b1 in trains[occ_a.train as usize]
        .blocks
        .iter()
        .skip(occ_a.block as usize + 1)
        .take(3)
    {
        for r1 in b1.resource_usage.iter() {
            for b2 in trains[occ_b.train as usize]
                .blocks
                .iter()
                .skip(occ_b.block as usize + 1)
                .take(3)
            {
                for r2 in b2.resource_usage.iter() {
                    if r1.resource == r2.resource {
                        same_dir = true;
                        break 'res;
                    }
                }
            }
        }
    }

    if same_dir {
        return 0.5;
    }

    let w_a = weights
        .and_then(|r| {
            r.range((occ_a.train, occ_a.block + 1)..(occ_a.train, BlockRef::MAX))
                .nth(0)
                .map(|(_, w)| *w)
        })
        .unwrap_or(0.);
    let w_b = weights
        .and_then(|r| {
            r.range((occ_b.train, occ_b.block + 1)..(occ_b.train, BlockRef::MAX))
                .nth(0)
                .map(|(_, w)| *w)
        })
        .unwrap_or(0.);

    //     let x = (w_a - w_b) as f64;
    // let slope: f64 = 1.0;

    // 1.0 / (1.0 + f64::exp(-x / slope))

    if w_a == 0. && w_b > 0. {
        1.
    } else if w_b == 0. && w_a > 0. {
        0.
    } else {
        0.5
    }
}

pub fn old_node_evaluation(
    slacks: &Vec<Vec<i32>>,
    occ_a: &ResourceOccupation,
    _c_a: &ConflictConstraint,
    occ_b: &ResourceOccupation,
    _c_b: &ConflictConstraint,
) -> NodeEval {
    // Things we might want to know to decide priority of node.

    // How much do the trains overlap.
    let _interval_inner = occ_a.interval.intersect(&occ_b.interval);

    // How much time from the first entry to the last exit.
    let interval_outer = occ_a.interval.envelope(&occ_b.interval);

    // How much time do the trains spend in this resource.
    let _total_time = occ_a.interval.length() + occ_b.interval.length();

    // Is one interval contained in the other?
    let _contained = (occ_a.interval.time_start < occ_b.interval.time_start)
        != (occ_a.interval.time_end < occ_b.interval.time_end);

    let _delay_a_to = occ_b.interval.time_end;
    let _delay_b_to = occ_a.interval.time_end;

    let _slack_a = slacks[occ_a.train as usize][occ_a.block as usize] - occ_a.interval.time_start;
    let _new_slack_a = slacks[occ_a.train as usize][occ_a.block as usize] - occ_b.interval.time_end;
    let _slack_b = slacks[occ_b.train as usize][occ_b.block as usize] - occ_b.interval.time_start;
    let _new_slack_b = slacks[occ_b.train as usize][occ_b.block as usize] - occ_a.interval.time_end;

    // match (
    //     slack_a >= 0,
    //     new_slack_a >= 0,
    //     slack_b >= 0,
    //     new_slack_b >= 0,
    // ) {
    //     (_, false, _, true) => {
    //         return NodeEval {
    //             a_best: false,
    //             node_score: 0.1 * interval_outer.length() as f64,
    //         }
    //     }
    //     (_, true, _, false) => {
    //         return NodeEval {
    //             a_best: true,
    //             node_score: 0.1 * interval_outer.length() as f64,
    //         }
    //     }
    //     _ => {
    //         return NodeEval {
    //             a_best: new_slack_a >= new_slack_b,
    //             node_score: interval_outer.length() as f64,
    //         }
    //     }
    // }

    let mut _sum_score = 0.0;
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
    _sum_score += h1_score;
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
    _sum_score += h2_score;
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
    _sum_score += h3_score;
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
    let _significance = interval_outer.length() as f64;

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

    let res = NodeEval {
        a_best: choice,
        node_score: importance,
    };

    error!("NODE EVAL {:?}", res);
    res
}
