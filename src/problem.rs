pub struct Problem {
    pub trains: Vec<Train>,
    pub resources: Vec<Resource>,
    pub conflicts :Vec<(usize,usize)>,
}

pub struct Resource {
    pub travel_time: i32,
}

pub struct Train {
    pub path: Vec<(i32, usize)>,
}

impl Train {
    pub fn delay_cost(&self, path_idx :usize, t :i32) -> usize {
        let delay = t - self.path[path_idx].0;
        if delay > 360 {
            3
        } else if delay  > 180 {
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
        conflicts: (0..=6).map(|i| (i,i)).collect() // resources only conflict with themselves.
    }
}
