use crate::{
    interval::TimeInterval,
    problem::{Block, BlockRef, ResourceRef, TimeValue, Train}, TrainSolverStatus,
};
use itertools::Itertools;
use log::{debug, error, info, trace, warn};
use std::rc::Rc;
use tinyvec::TinyVec;

pub struct DagTrainSolver {
    pub id: usize,
    pub train: Train,
    resource_to_blocks: Vec<TinyVec<[BlockRef; 4]>>,
    times: Vec<TinyVec<[(TimeValue, BlockRef); 8]>>,
    prevs: Vec<TinyVec<[u32; 4]>>,
    status: TrainSolverStatus,
    pub occupied: Vec<TinyVec<[(TimeValue, TimeValue); 4]>>,
}

impl DagTrainSolver {
    fn new(id: usize, train: Train) -> Self {
        let mut resource_to_blocks: Vec<TinyVec<[BlockRef; 4]>> = vec![];
        let mut prevs: Vec<TinyVec<[u32; 4]>> = train
            .blocks
            .iter()
            .map(|_| Default::default())
            .collect::<Vec<_>>();

        for (block_idx, block) in train.blocks.iter().enumerate() {
            for resource in block.resource_usage.iter() {
                while resource_to_blocks.len() < resource.resource as usize {
                    resource_to_blocks.push(Default::default());
                }

                resource_to_blocks[resource.resource as usize].push(block_idx as BlockRef);

                for next in block.nexts.iter() {
                    prevs[*next as usize].push(block_idx as u32);
                }
            }
        }

        let times = train.blocks.iter().map(|_| Default::default()).collect();
        let occupied = resource_to_blocks
            .iter()
            .map(|_| Default::default())
            .collect();

        Self {
            id,
            train,
            resource_to_blocks,
            times,
            prevs,
            status: TrainSolverStatus::Working,
            occupied,
        }
    }

    fn current_solution(&self) -> (i32, Vec<TimeValue>) {
        assert!(matches!(self.status, TrainSolverStatus::Optimal));
        todo!()
    }

    fn current_time(&self) -> TimeValue {
        todo!()
    }

    fn status(&self) -> TrainSolverStatus {
        todo!()
    }

    fn step(&mut self, use_resource: &mut impl FnMut(bool, ResourceRef, TimeInterval)) {
        assert!(matches!(self.status, TrainSolverStatus::Working));

        for (block_idx, block) in self.train.blocks.iter().enumerate() {
            let mut new_times: TinyVec<[(TimeValue, BlockRef); 8]> = Default::default();
            if block_idx == 0 {
                new_times.push((block.earliest_start, 0));
            } else {
                for &prev in self.prevs[block_idx].iter() {
                    for &(prev_time, _) in self.times[prev as usize].iter() {
                        let earliest_exit_prev = (prev_time
                            + self.train.blocks[prev as usize].minimum_travel_time)
                            .max(block.earliest_start);

                        let latest_exit_prev = TimeValue::MAX; // TODO

                        if earliest_exit_prev > latest_exit_prev {
                            continue;
                        }

                        new_times.push((earliest_exit_prev, prev));
                        for &(_, enter_after) in self.occupied[block_idx].iter() {
                            if enter_after >= earliest_exit_prev && enter_after <= latest_exit_prev
                            {
                                new_times.push((enter_after, prev));
                            }
                        }
                    }
                }

                // Sort and dedup
                new_times.sort_by_key(|(x, _)| *x);
                let mut i = 0;
                while i + 1 < new_times.len() {
                    let divided_by_conflict =
                        self.occupied[block_idx]
                            .iter()
                            .any(|&(exit_before, _enter_after)| {
                                (new_times[i].0 <= exit_before)
                                    != (new_times[i + 1].0 <= exit_before)
                            });

                    if !divided_by_conflict {
                        new_times.remove(i + 1);
                    } else {
                        i += 1;
                    }
                }
            }

            self.times[block_idx] = new_times;
        }
    }

    fn set_occupied(
        &mut self,
        add: bool,
        resource: ResourceRef,
        enter_after: TimeValue,
        exit_before: TimeValue,
        use_resource: &mut impl FnMut(bool, BlockRef, ResourceRef, TimeInterval),
    ) {
        todo!()
    }
}
