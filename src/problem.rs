#[derive(Debug)]
pub struct Problem {
    pub trains: Vec<Train>,
    pub conflicts: Vec<(usize, usize)>,
}

#[derive(Debug)]
pub struct Train {
    pub visits: Vec<(usize, i32, i32)>,
}

impl Problem {
    pub fn verify_solution(&self, solution: &[Vec<i32>]) -> Option<i32> {
        let _p = hprof::enter("verify_solution");
        // Check the shape of the solution
        assert!(self.trains.len() == solution.len());
        for (train_idx, train) in self.trains.iter().enumerate() {
            assert!(solution[train_idx].len() == train.visits.len() + 1);
        }

        let mut sum_cost = 0;

        // Check the running times and sum up the delays
        for (train_idx, train) in self.trains.iter().enumerate() {
            for (visit_idx, (_resource, earliest, travel_time)) in train.visits.iter().enumerate() {
                let t1_in = solution[train_idx][visit_idx];
                let t1_out = solution[train_idx][visit_idx + 1];

                if t1_in < *earliest {
                    println!("Earliest entry conflict t{} v{}", train_idx, visit_idx);
                    return None;
                }

                if t1_in + travel_time > t1_out {
                    println!("Travel time conflict t{} v{}", train_idx, visit_idx);
                    return None;
                }

                let cost = train.delay_cost(visit_idx, t1_in) as i32;
                if cost > 0 {
                    // println!("Added cost for t{} v{} = {}", train_idx, visit_idx, cost);
                    sum_cost += cost;
                }
            }
        }

        // Check all pairs of visits for resource conflicts.
        for (train_idx1, train1) in self.trains.iter().enumerate() {
            for (visit_idx1, (r1, _earliest1, _travel1)) in train1.visits.iter().enumerate() {
                for (train_idx2, train2) in self.trains.iter().enumerate() {
                    for (visit_idx2, (r2, _earliest2, _travel2)) in train2.visits.iter().enumerate()
                    {
                        if (train_idx1 != train_idx2 || visit_idx1 != visit_idx2)
                            && self.conflicts.contains(&(*r1, *r2))
                        {
                            // Different visits to the conflicting resources.

                            let t1_in = solution[train_idx1][visit_idx1];
                            let t1_out = solution[train_idx1][visit_idx1 + 1];
                            let t2_in = solution[train_idx2][visit_idx2];
                            let t2_out = solution[train_idx2][visit_idx2 + 1];

                            let ok = t1_in >= t2_out || t2_in >= t1_out;
                            if !ok {
                                println!(
                                    "Resource conflict in t{} v{} t{} v{}",
                                    train_idx1, visit_idx1, train_idx2, visit_idx2
                                );
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
    pub fn delay_cost(&self, path_idx: usize, t: i32) -> usize {
        let delay = t - self.visits[path_idx].1;
        let cost = if delay > 360 {
            // 5 + 3*(delay as usize -360)/360
            3
        } else if delay > 180 {
            2
        } else if delay > 0 {
            1
        } else {
            0
        };
        // println!("new delay cost {} {} {}", path_idx, t, cost);
        cost
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
                    (0, 0, travel_times[0]),
                    (7, 6, 0),
                    (1, 6, travel_times[1]),
                    (7, 9, 0),
                    (6, 9, travel_times[6]),
                ],
            },
            Train {
                visits: vec![(2, 0, travel_times[2]), (7, 4, 0), (1, 4, travel_times[1])],
            },
            Train {
                visits: vec![
                    (3, 0, travel_times[3]),
                    (7, 9, 0),
                    (1, 9, travel_times[1]),
                    (7, 12, 0),
                    (5, 12, travel_times[5]),
                ],
            },
            Train {
                visits: vec![(4, 0, travel_times[4]), (7,10,0), (5, 10, travel_times[5])],
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
                    (0, 0, travel_times[0]),
                    (1, 6, travel_times[1]),
                    (6, 9, travel_times[6]),
                ],
            },
            Train {
                visits: vec![(2, 0, travel_times[2]), (1, 4, travel_times[1])],
            },
            Train {
                visits: vec![
                    (3, 0, travel_times[3]),
                    (1, 9, travel_times[1]),
                    (5, 12, travel_times[5]),
                ],
            },
            Train {
                visits: vec![(4, 0, travel_times[4]), (5, 10, travel_times[5])],
            },
        ],

        conflicts: (0..=6).map(|i| (i, i)).collect(), // resources only conflict with themselves.
    }
}
