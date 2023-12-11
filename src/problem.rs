#[derive(Debug, Clone, Copy)]
pub enum DelayMeasurementType {
    EverywhereEarliest,
    AllStationArrivals,
    AllStationDepartures,
    FinalStationArrival,
}

#[derive(Debug, Clone, Copy)]
pub enum DelayCostType {
    FiniteSteps1_3Min,
    FiniteSteps1_5Min,
    FiniteSteps123,
    FiniteSteps12345,
    FiniteSteps139,
    InfiniteSteps60,
    InfiniteSteps180,
    InfiniteSteps360,
    Continuous,
}

#[derive(Debug)]
pub struct NamedProblem {
    pub problem: Problem,
    pub train_names: Vec<String>,
    pub resource_names: Vec<String>,
}

#[derive(Debug)]
pub struct Problem {
    pub trains: Vec<Train>,
    pub conflicts: Vec<(usize, usize)>,
}

#[derive(Debug)]
pub struct Train {
    pub visits: Vec<Visit>,
}

#[derive(Debug, Clone, Copy)]
pub struct Visit {
    pub resource_id: usize,
    pub earliest: i32,
    pub aimed: Option<i32>,
    pub travel_time: i32,
}

impl Problem {
    pub fn train_cost(
        &self,
        solution: &[Vec<i32>],
        delay_cost_type: DelayCostType,
        train_idx: usize,
    ) -> i32 {
        let mut sum_cost = 0;
        let train = &self.trains[train_idx];
        for (visit_idx, Visit { .. }) in train.visits.iter().enumerate() {
            let t1_in = solution[train_idx][visit_idx];

            let cost = train.visit_delay_cost(delay_cost_type, visit_idx, t1_in) as i32;
            if cost > 0 {
                // println!("Added cost for t{} v{} = {}", train_idx, visit_idx, cost);
                sum_cost += cost;
            }
        }

        sum_cost
    }

    pub fn cost(&self, solution: &[Vec<i32>], delay_cost_type: DelayCostType) -> i32 {
        let mut sum_cost = 0;

        // Check the running times and sum up the delays
        for (train_idx, train) in self.trains.iter().enumerate() {
            for visit_idx in 0..train.visits.len() {
                let t1_in = solution[train_idx][visit_idx];
                let cost = train.visit_delay_cost(delay_cost_type, visit_idx, t1_in) as i32;
                if cost > 0 {
                    // println!("Added cost for t{} v{} = {}", train_idx, visit_idx, cost);
                    sum_cost += cost;
                }
            }
        }
        sum_cost
    }

    pub fn verify_solution(
        &self,
        solution: &[Vec<i32>],
        delay_cost_type: DelayCostType,
    ) -> Option<i32> {
        let _p = hprof::enter("verify_solution");
        // Check the shape of the solution
        assert!(self.trains.len() == solution.len());
        for (train_idx, train) in self.trains.iter().enumerate() {
            assert!(solution[train_idx].len() == train.visits.len() + 1);
        }

        let mut sum_cost = 0;

        // Check the running times and sum up the delays
        for (train_idx, train) in self.trains.iter().enumerate() {
            for (
                visit_idx,
                Visit {
                    earliest,
                    travel_time,
                    ..
                },
            ) in train.visits.iter().enumerate()
            {
                let t1_in = solution[train_idx][visit_idx];
                let t1_out = solution[train_idx][visit_idx + 1];

                if t1_in < *earliest - 10 {
                    println!("Earliest entry conflict t{} v{}", train_idx, visit_idx);
                    println!("train {}: {:?}", train_idx, train);
                    println!("  train solution: {:?}", solution[train_idx]);
                    return None;
                }

                if t1_in + travel_time > t1_out {
                    println!("Travel time conflict t{} v{}", train_idx, visit_idx);
                    println!("train {}: {:?}", train_idx, train);
                    println!("  train solution: {:?}", solution[train_idx]);
                    return None;
                }

                let cost = train.visit_delay_cost(delay_cost_type, visit_idx, t1_in) as i32;
                if cost > 0 {
                    // println!("Added cost for t{} v{} = {}", train_idx, visit_idx, cost);
                    sum_cost += cost;
                }
            }
        }

        // Check all pairs of visits for resource conflicts.
        for (train_idx1, train1) in self.trains.iter().enumerate() {
            for (
                visit_idx1,
                Visit {
                    resource_id: r1, ..
                },
            ) in train1.visits.iter().enumerate()
            {
                for (train_idx2, train2) in self.trains.iter().enumerate() {
                    for (
                        visit_idx2,
                        Visit {
                            resource_id: r2, ..
                        },
                    ) in train2.visits.iter().enumerate()
                    {
                        if (train_idx1 != train_idx2 || visit_idx1 != visit_idx2)
                            && self.conflicts.contains(&(*r1, *r2))
                        {
                            // Different visits to the conflicting resources.

                            let t1_in = solution[train_idx1][visit_idx1];
                            let t1_out = solution[train_idx1][visit_idx1 + 1];
                            let t2_in = solution[train_idx2][visit_idx2];
                            let t2_out = solution[train_idx2][visit_idx2 + 1];

                            let ok = t1_in >= t2_out - 1 || t2_in >= t1_out - 1;
                            if !ok {
                                println!(
                                    "Resource conflict {}-{} in t{} v{} {}-{} t{} v{} {}-{}",
                                    *r1, *r2,
                                    train_idx1,
                                    visit_idx1,
                                    t1_in,
                                    t1_out,
                                    train_idx2,
                                    visit_idx2,
                                    t2_in,
                                    t2_out
                                );
                                println!("train1 {}: {:?}", train_idx1, train1);
                                println!("  train1 solution: {:?}", solution[train_idx1]);
                                println!("  {:?}", train1.visits[visit_idx1]);
                                println!("train2 {}: {:?}", train_idx2, train2);
                                println!("  train2 solution: {:?}", solution[train_idx2]);
                                println!("  {:?}", train2.visits[visit_idx2]);
                                return None;
                            }
                        }
                    }
                }
            }
        }

        println!("Solution verified. Cost {}", sum_cost);
        Some(sum_cost)
    }
}

impl Train {
    pub fn visit_delay_cost(
        &self,
        delay_cost_type: DelayCostType,
        path_idx: usize,
        t: i32,
    ) -> usize {
        if let Some(aimed) = self.visits[path_idx].aimed {
            let d = (t - aimed).max(0);
            match delay_cost_type {
                DelayCostType::FiniteSteps1_3Min => DelayCostThresholds::f1_3min().eval(d),
                DelayCostType::FiniteSteps1_5Min => DelayCostThresholds::f1_5min().eval(d),
                DelayCostType::FiniteSteps123 => DelayCostThresholds::f123().eval(d),
                DelayCostType::FiniteSteps12345 => DelayCostThresholds::f12345().eval(d),
                DelayCostType::FiniteSteps139 => DelayCostThresholds::f139().eval(d),
                DelayCostType::InfiniteSteps60 => infinite_staircase(d, 60),
                DelayCostType::InfiniteSteps180 => infinite_staircase(d, 180),
                DelayCostType::InfiniteSteps360 => infinite_staircase(d, 360),
                DelayCostType::Continuous => d as usize,
            }
        } else {
            0
        }
    }
}

pub fn infinite_staircase(delay: i32, interval: usize) -> usize {
    // if delay <= 0 {
    //     0
    // } else {
    (delay as f64 / interval as f64).ceil() as usize
    // }
}

pub fn iter_infinite_staircase(interval: usize) -> impl Iterator<Item = (i32, usize)> {
    (0..).map(move |x| (interval as i32 * x + 1, x as usize + 1))
}

pub struct DelayCostThresholds {
    pub thresholds: Vec<(i32, usize)>,
}

impl DelayCostThresholds {
    pub fn f1_3min() -> DelayCostThresholds {
        DelayCostThresholds {
            thresholds: vec![(5 * 60, 1)],
        }
    }
    pub fn f1_5min() -> DelayCostThresholds {
        DelayCostThresholds {
            thresholds: vec![(3 * 60, 1)],
        }
    }
    pub fn f123() -> DelayCostThresholds {
        DelayCostThresholds {
            thresholds: vec![(360, 3), (180, 2), (0, 1)],
        }
    }
    pub fn f12345() -> DelayCostThresholds {
        DelayCostThresholds {
            thresholds: vec![(3 * 360, 5), (2 * 360, 4), (360, 3), (180, 2), (0, 1)],
        }
    }
    pub fn f139() -> DelayCostThresholds {
        DelayCostThresholds {
            thresholds: vec![(360, 9), (180, 3), (0, 1)],
        }
    }

    pub fn eval(&self, delay: i32) -> usize {
        for (threshold, cost) in &self.thresholds {
            if delay > *threshold {
                return *cost;
            }
        }
        0
    }
}

fn visit(resource_id: usize, earliest: i32, travel_time: i32) -> Visit {
    Visit {
        resource_id,
        earliest,
        travel_time,
        aimed: Some(earliest),
    }
}

#[allow(unused)]
pub fn problem1_with_stations() -> Problem {
    // a = 0
    // b = 1
    // c = 2
    // d = 3
    // e = 4
    // f = 5
    // g = 6

    let travel_times = vec![6, 3, 4, 9, 10, 5, 8];
    Problem {
        trains: vec![
            Train {
                visits: vec![
                    visit(0, 0, travel_times[0]),
                    visit(7, 6, 0),
                    visit(1, 6, travel_times[1]),
                    visit(7, 9, 0),
                    visit(6, 9, travel_times[6]),
                ],
            },
            Train {
                visits: vec![
                    visit(2, 0, travel_times[2]),
                    visit(7, 4, 0),
                    visit(1, 4, travel_times[1]),
                ],
            },
            Train {
                visits: vec![
                    visit(3, 0, travel_times[3]),
                    visit(7, 9, 0),
                    visit(1, 9, travel_times[1]),
                    visit(7, 12, 0),
                    visit(5, 12, travel_times[5]),
                ],
            },
            Train {
                visits: vec![
                    visit(4, 0, travel_times[4]),
                    visit(7, 10, 0),
                    visit(5, 10, travel_times[5]),
                ],
            },
        ],

        conflicts: (0..=6).map(|i| (i, i)).collect(), // resources only conflict with themselves.
    }
}
#[allow(unused)]
pub fn problem1() -> Problem {
    // a = 0
    // b = 1
    // c = 2
    // d = 3
    // e = 4
    // f = 5
    // g = 6
    let travel_times = vec![6, 3, 4, 9, 10, 5, 8];
    Problem {
        trains: vec![
            Train {
                visits: vec![
                    visit(0, 0, travel_times[0]),
                    visit(1, 6, travel_times[1]),
                    visit(6, 9, travel_times[6]),
                ],
            },
            Train {
                visits: vec![visit(2, 0, travel_times[2]), visit(1, 4, travel_times[1])],
            },
            Train {
                visits: vec![
                    visit(3, 0, travel_times[3]),
                    visit(1, 9, travel_times[1]),
                    visit(5, 12, travel_times[5]),
                ],
            },
            Train {
                visits: vec![visit(4, 0, travel_times[4]), visit(5, 10, travel_times[5])],
            },
        ],

        conflicts: (0..=6).map(|i| (i, i)).collect(), // resources only conflict with themselves.
    }
}
