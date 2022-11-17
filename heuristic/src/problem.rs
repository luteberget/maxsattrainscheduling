use serde::Deserialize;
use tinyvec::TinyVec;

#[derive(Deserialize, Debug)]
pub struct Problem {
    pub n_resources: usize,
    pub trains: Vec<Train>,
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
    pub earliest_departure: TimeValue,
    pub resource_usage: TinyVec<[ResourceUsage; 4]>,
    pub nexts: TinyVec<[u32; 4]>,
}

#[derive(Default, Deserialize, Debug)]
pub struct ResourceUsage {
    pub resource :ResourceRef,
    pub release_after :TimeValue,
}
