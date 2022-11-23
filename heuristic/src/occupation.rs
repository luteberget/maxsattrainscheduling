use log::{warn, trace};
use tinyvec::TinyVec;

use crate::{problem::{TrainRef, ResourceRef}, interval::TimeInterval};



#[derive(Debug)]
pub struct ResourceConflicts {
    pub conflicting_resource_set: Vec<u32>,
    pub resources: Vec<ResourceOccupations>,
}

#[derive(Debug)]
pub struct ResourceOccupations {
    pub conflicting_resource_set_idx: i32,
    pub occupations: TinyVec<[ResourceOccupation; 32]>,
}

impl ResourceOccupations {
    pub fn get_conflict(&self) -> Option<(&ResourceOccupation, &ResourceOccupation)> {
        // TODO this is not correct (but fast)
        self.occupations
            .iter()
            .zip(self.occupations.iter().skip(1))
            .find(|(a, b)| a.interval.overlap(&b.interval))
    }
}

#[derive(Debug, Default, PartialEq, Eq, PartialOrd, Ord)]
pub struct ResourceOccupation {
    pub interval: TimeInterval,
    pub train: TrainRef,
}

impl ResourceConflicts {
    pub fn empty(n: usize) -> Self {
        ResourceConflicts {
            conflicting_resource_set: vec![],
            resources: (0..n)
                .map(|_| ResourceOccupations {
                    conflicting_resource_set_idx: -1,
                    occupations: TinyVec::new(),
                })
                .collect(),
        }
    }

    pub fn add(&mut self, resource_idx: usize, occ: ResourceOccupation) {
        let resource = &mut self.resources[resource_idx];
        match resource.occupations.binary_search(&occ) {
            Ok(_) => {
                warn!("already occupied {} {:?}", resource_idx, occ);
            }
            Err(idx) => {
                resource.occupations.insert(idx, occ);
                trace!("Inserted res{} {:?} ", resource_idx, resource.occupations);
                if resource.conflicting_resource_set_idx < 0 && resource.get_conflict().is_some() {
                    resource.conflicting_resource_set_idx =
                        self.conflicting_resource_set.len() as i32;
                    self.conflicting_resource_set.push(resource_idx as u32);
                }
            }
        }
    }

    pub fn remove(&mut self, resource_idx: usize, occ: ResourceOccupation) {
        let resource = &mut self.resources[resource_idx];
        let idx = resource.occupations.binary_search(&occ).unwrap();
        resource.occupations.remove(idx);

        if resource.conflicting_resource_set_idx >= 0 && resource.get_conflict().is_none() {
            self.conflicting_resource_set
                .swap_remove(resource.conflicting_resource_set_idx as usize);
            if (resource.conflicting_resource_set_idx as usize)
                < self.conflicting_resource_set.len()
            {
                let other_resource =
                    self.conflicting_resource_set[resource.conflicting_resource_set_idx as usize];
                self.resources[other_resource as usize].conflicting_resource_set_idx =
                    resource.conflicting_resource_set_idx as i32;
            }
            self.resources[resource_idx].conflicting_resource_set_idx = -1;
        }
    }

    pub fn add_or_remove(
        &mut self,
        add: bool,
        train: TrainRef,
        resource: ResourceRef,
        interval: TimeInterval,
    ) {
        let occ = ResourceOccupation { train, interval };
        if add {
            trace!(
                "ADD train{} track{} [{} -> {}] ",
                train,
                resource,
                occ.interval.time_start,
                occ.interval.time_end
            );
            self.add(resource as usize, occ);
        } else {
            trace!(
                "DEL train{} track{} [{} -> {}] ",
                train,
                resource,
                occ.interval.time_start,
                occ.interval.time_end
            );
            self.remove(resource as usize, occ);
        }
    }
}
