pub struct Problem {
    pub trains: Vec<Train>,
    pub resources: Vec<Resource>,
    pub conflicts: Vec<(usize, usize)>,
}

impl Problem {
    pub fn verify_solution(&self, solution: &Vec<Vec<i32>>) -> Option<i32> {
        // Check the shape of the solution
        assert!(self.trains.len() == solution.len());
        for (train_idx, train) in self.trains.iter().enumerate() {
            assert!(solution[train_idx].len() == train.path.len() + 1);
        }

        let mut sum_cost = 0;

        // Check the running times and sum up the delays
        for (train_idx, train) in self.trains.iter().enumerate() {
            for (visit_idx, (earliest, resource)) in train.path.iter().enumerate() {

                let t1_in = solution[train_idx][visit_idx];
                let t1_out = solution[train_idx][visit_idx + 1];

                if t1_in < *earliest {
                    println!("Earliest entry conflict t{} v{}", train_idx, visit_idx);
                    return None;
                }

                let travel_time = self.resources[*resource].travel_time;

                if t1_in + travel_time > t1_out {
                    println!("Travel time conflict t{} v{}", train_idx, visit_idx);
                    return None;
                }

                let cost = train.delay_cost(visit_idx, t1_in) as i32;
                if cost > 0 {
                    println!("Added cost for t{} v{} = {}", train_idx, visit_idx, cost);
                    sum_cost += cost;
                }

            }
        }

        // Check all pairs of visits for resource conflicts.
        for (train_idx1, train1) in self.trains.iter().enumerate() {
            for (visit_idx1, (_, r1)) in train1.path.iter().enumerate() {
                for (train_idx2, train2) in self.trains.iter().enumerate() {
                    for (visit_idx2, (_, r2)) in train2.path.iter().enumerate() {
                        if (train_idx1 != train_idx2 || visit_idx1 != visit_idx2) && r1 == r2 {
                            // Different visits to the same resource.

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

pub struct Resource {
    pub travel_time: i32,
}

pub struct Train {
    pub path: Vec<(i32, usize)>,
}

impl Train {
    pub fn delay_cost(&self, path_idx: usize, t: i32) -> usize {
        let delay = t - self.path[path_idx].0;
        if delay > 360 {
            3
        } else if delay > 180 {
            2
        } else if delay > 0 {
            1
        } else {
            0
        }
    }
}

pub fn problem1() -> Problem {
    // a = 0
    // b = 1
    // c = 2
    // d = 3
    // e = 4
    // f = 5
    // g = 6
    Problem {
        trains: vec![
            Train {
                path: vec![(0, 0), (6, 1), (9, 6)],
            },
            Train {
                path: vec![(0, 2), (4, 1)],
            },
            Train {
                path: vec![(0, 3), (9, 1), (12, 5)],
            },
            Train {
                path: vec![(0, 4), (10, 5)],
            },
        ],
        resources: vec![
            Resource { travel_time: 6 },
            Resource { travel_time: 3 },
            Resource { travel_time: 4 },
            Resource { travel_time: 9 },
            Resource { travel_time: 10 },
            Resource { travel_time: 5 },
            Resource { travel_time: 8 },
        ],
        conflicts: (0..=6).map(|i| (i, i)).collect(), // resources only conflict with themselves.
    }
}
