use heuristic::problem::{tiny_vec, Block, Problem, ResourceUsage, Train};

pub fn example_1() -> Problem {
    // 3 stations with 2 tracks each, plus two single tracks between consecutive stations

    let mut trains = Vec::new();

    for offset in [0, 5] {
        trains.push(Train {
            blocks: vec![
                Block {
                    minimum_travel_time: 0,
                    aimed_start: 0,
                    earliest_start: 0,
                    resource_usage: Default::default(),
                    nexts: tiny_vec!(1, 2),
                },
                Block {
                    minimum_travel_time: 10,
                    aimed_start: 0,
                    earliest_start: offset,
                    resource_usage: tiny_vec!(ResourceUsage {
                        resource: 0,
                        release_after: 9999,
                    }),
                    nexts: tiny_vec!(3),
                },
                Block {
                    minimum_travel_time: 10,
                    aimed_start: 0,
                    earliest_start: offset,
                    resource_usage: tiny_vec!(ResourceUsage {
                        resource: 1,
                        release_after: 9999,
                    }),
                    nexts: tiny_vec!(3),
                },
                Block {
                    minimum_travel_time: 100,
                    aimed_start: 0,
                    earliest_start: 0,
                    resource_usage: tiny_vec!(ResourceUsage {
                        resource: 2,
                        release_after: 9999,
                    }),
                    nexts: tiny_vec!(4, 5),
                },
                Block {
                    minimum_travel_time: 10,
                    aimed_start: 0,
                    earliest_start: 0,
                    resource_usage: tiny_vec!(ResourceUsage {
                        resource: 3,
                        release_after: 9999,
                    }),
                    nexts: tiny_vec!(6),
                },
                Block {
                    minimum_travel_time: 10,
                    aimed_start: 0,
                    earliest_start: 0,
                    resource_usage: tiny_vec!(ResourceUsage {
                        resource: 4,
                        release_after: 9999,
                    }),
                    nexts: tiny_vec!(6),
                },
                Block {
                    minimum_travel_time: 100,
                    aimed_start: 0,
                    earliest_start: 0,
                    resource_usage: tiny_vec!(ResourceUsage {
                        resource: 5,
                        release_after: 9999,
                    }),
                    nexts: tiny_vec!(7, 8),
                },
                Block {
                    minimum_travel_time: 10,
                    aimed_start: 0,
                    earliest_start: 0,
                    resource_usage: tiny_vec!(ResourceUsage {
                        resource: 6,
                        release_after: 9999,
                    }),
                    nexts: tiny_vec!(9),
                },
                Block {
                    minimum_travel_time: 10,
                    aimed_start: 0,
                    earliest_start: 0,
                    resource_usage: tiny_vec!(ResourceUsage {
                        resource: 7,
                        release_after: 9999,
                    }),
                    nexts: tiny_vec!(9),
                },
                Block {
                    minimum_travel_time: 0,
                    aimed_start: 0,
                    earliest_start: 0,
                    resource_usage: Default::default(),
                    nexts: tiny_vec!(),
                },
            ],
        });
    }

    Problem {
        n_resources: 3 * 2 + 2,
        trains,
    }
}
