use std::collections::HashSet;

use serde::{Deserialize, Serialize};
pub use tinyvec::{tiny_vec, TinyVec};

#[derive(Serialize, Deserialize, Debug)]
pub struct Problem {
    pub n_resources: usize,
    pub trains: Vec<Train>,
}

impl Problem {
    pub fn verify(&self) {

        for t in &self.trains {
            for (block_idx,b) in t.blocks.iter().enumerate() {
                for n in &b.nexts {
                    if (*n as usize) >= t.blocks.len() {
                        panic!("Invalid block reference.");
                    }

                    if (*n as usize) <= block_idx {
                        panic!("Blocks must be topologically ordered.");
                    }
                }
            }
        }

        // All resouce usages
        for t in &self.trains {
            for b in &t.blocks {
                for r in &b.resource_usage {
                    if (r.resource as usize) >= self.n_resources {
                        panic!("Invalid resource reference.");
                    }
                }
            }
        }
    }
}

pub type TrainRef = u32;
pub type BlockRef = u32;
pub type ResourceRef = u32;
pub type TimeValue = i32;

#[derive(Serialize, Deserialize, Debug)]
pub struct Train {
    pub blocks: Vec<Block>,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct Block {
    pub minimum_travel_time: TimeValue,
    pub aimed_start: TimeValue,
    pub delayed_after: Option<TimeValue>,
    pub earliest_start: TimeValue,
    pub resource_usage: TinyVec<[ResourceUsage; 4]>,
    pub nexts: TinyVec<[u32; 4]>,
}

#[derive(Default,Serialize,  Deserialize, Debug)]
pub struct ResourceUsage {
    pub resource: ResourceRef,
    pub release_after: TimeValue,
}



pub fn convert_ddd_problem(problem: &ddd_problem::problem::NamedProblem) -> Problem {
    assert!(
        problem
            .problem
            .conflicts
            .iter()
            .copied()
            .collect::<HashSet<_>>()
            == (1..problem.resource_names.len())
                .map(|i| (i, i))
                .collect::<HashSet<_>>()
    );
    let mut trains = Vec::new();
    for train in problem.problem.trains.iter() {
        let mut blocks = vec![Block {
            minimum_travel_time: 0,
            aimed_start: 0,
            earliest_start: 0,
            resource_usage: std::iter::empty().collect(),
            nexts: std::iter::once(1).collect(),
            delayed_after : None,
        }];
        for visit in train.visits.iter() {
            // pub struct Visit {
            //     pub resource_id: usize,
            //     pub earliest: i32,
            //     pub aimed: Option<i32>,
            //     pub travel_time: i32,  }

            let resource_usage = if visit.resource_id > 0 {
                std::iter::once(ResourceUsage {
                    release_after: 999999,
                    resource: visit.resource_id as ResourceRef - 1,
                })
                .collect()
            } else {
                std::iter::empty().collect()
            };

            assert!(visit.travel_time >= 0);

            let next_idx = blocks.len() + 1;
            blocks.push(Block {
                aimed_start: 0,
                earliest_start: visit.earliest as TimeValue,
                minimum_travel_time: visit.travel_time as TimeValue,
                nexts: std::iter::once(next_idx as BlockRef).collect(),
                resource_usage,
                delayed_after : visit.aimed,
            });
        }

        blocks.push(Block {
            aimed_start: 0,
            earliest_start: 0,
            minimum_travel_time: 0,
            nexts: std::iter::empty().collect(),
            resource_usage: std::iter::empty().collect(),
            delayed_after : None,
        });

        trains.push(Train { blocks })
    }
     Problem {
        n_resources: problem.resource_names.len() - 1,
        trains,
    }
}
