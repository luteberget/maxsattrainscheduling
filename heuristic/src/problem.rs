use serde::Deserialize;
pub use tinyvec::{tiny_vec, TinyVec};

#[derive(Deserialize, Debug)]
pub struct Problem {
    pub n_resources: usize,
    pub trains: Vec<Train>,
}

impl Problem {
    pub fn verify(&self) {
        // All block next references
        for t in &self.trains {
            for b in &t.blocks {
                for n in &b.nexts {
                    if (*n as usize) >= t.blocks.len() {
                        panic!("Invalid block reference.");
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
pub type TimeValue = u32;

#[derive(Deserialize, Debug)]
pub struct Train {
    pub blocks: Vec<Block>,
}

#[derive(Deserialize, Debug)]
pub struct Block {
    pub minimum_travel_time: TimeValue,
    pub aimed_start: TimeValue,
    pub earliest_start: TimeValue,
    pub resource_usage: TinyVec<[ResourceUsage; 4]>,
    pub nexts: TinyVec<[u32; 4]>,
}

#[derive(Default, Deserialize, Debug)]
pub struct ResourceUsage {
    pub resource: ResourceRef,
    pub release_after: TimeValue,
}
