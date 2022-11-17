use std::{collections::BTreeSet, rc::Rc};

use log::{debug, trace, warn};

use crate::{
    interval::{TimeInterval, INTERVAL_MIN},
    problem::{Block, BlockRef, ResourceRef, TimeValue, Train},
};

#[derive(Debug)]
pub enum TrainSolverStatus {
    Failed,
    Optimal,
    Working,
}

pub struct TrainSolverNode {
    pub block: BlockRef,
    pub time: TimeValue,
    depth: u32,
    pub parent: Option<Rc<TrainSolverNode>>,
}

impl std::fmt::Debug for TrainSolverNode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TrainSolverNode")
            .field("block", &self.block)
            .field("time", &self.time)
            .field("depth", &self.depth)
            .field("has_parent", &self.parent.is_some())
            .finish()
    }
}

#[derive(PartialEq, Eq, PartialOrd, Ord, Debug, Copy, Clone)]
pub struct ResourceBlocked {
    resource: ResourceRef,
    interval: TimeInterval,
}

pub struct TrainSolver {
    pub train: Train,
    pub blocks: BTreeSet<ResourceBlocked>,

    root_node: Rc<TrainSolverNode>,
    pub queued_nodes: Vec<Rc<TrainSolverNode>>,
    pub current_node: Rc<TrainSolverNode>,
    pub solution: Option<Result<Rc<TrainSolverNode>, ()>>,

    pub total_nodes: usize,
}

impl TrainSolver {
    pub fn new(train: Train) -> Self {
        let root_node = Rc::new(TrainSolverNode {
            time: TimeValue::MIN,
            block: 0,
            parent: None,
            depth: 0,
        });

        Self {
            current_node: root_node.clone(),
            root_node,
            train,
            queued_nodes: vec![],
            solution: None,
            blocks: Default::default(),
            total_nodes: 1,
        }
    }

    pub fn current_time(&self) -> TimeValue {
        self.current_node.time
    }

    pub fn reset(&mut self, use_resource: impl FnMut(bool, ResourceRef, TimeInterval)) {
        // TODO extract struct TrainSearchState
        self.queued_nodes.clear();
        self.solution = None;
        self.switch_node(self.root_node.clone(), use_resource);
    }

    pub fn status(&self) -> TrainSolverStatus {
        match self.solution {
            Some(Ok(_)) => TrainSolverStatus::Optimal,
            Some(Err(())) => TrainSolverStatus::Failed,
            None => TrainSolverStatus::Working,
        }
    }

    pub fn step(&mut self, use_resource: impl FnMut(bool, ResourceRef, TimeInterval)) {
        assert!(matches!(self.status(), TrainSolverStatus::Working));

        for next_block in self.train.blocks[self.current_node.block as usize]
            .nexts
            .iter()
        {
            trace!(
                "  - next track {} has valid transfer times {:?}",
                next_block,
                valid_transfer_times(
                    &self.train,
                    &self.blocks,
                    self.current_node.block,
                    self.current_node.time,
                    *next_block,
                )
                .collect::<Vec<_>>()
            );

            for time in valid_transfer_times(
                &self.train,
                &self.blocks,
                self.current_node.block,
                self.current_node.time,
                *next_block,
            ) {
                if self.train.blocks[*next_block as usize].nexts.is_empty() {
                    panic!("solution");
                } else {
                    trace!("  adding node {} {}", next_block, time);
                    self.total_nodes += 1;

                    self.queued_nodes.push(Rc::new(TrainSolverNode {
                        time,
                        block: *next_block,
                        parent: Some(self.current_node.clone()),
                        depth: self.current_node.depth + 1,
                    }));
                }
            }
        }

        // if self.train.blocks[self.current_node.block as usize]
        //     .nexts
        //     .is_empty()
        // {
        //     // TODO special casing the last track segments.
        //     // This should be replaced by explicitly linking to TRAIN_FINISHED
        //     trace!("  adding train finished node");
        //     assert!(self.current_node.block >= 0);
        //     let travel_time =
        //         self.train.blocks[self.current_node.block as usize].minimum_travel_time;

        //     // We should already have made sure that the minium travel time is available in the resource(s).
        //     assert!(block_available(
        //         &self.blocks,
        //         ResourceBlocked {
        //             interval: TimeInterval::duration(self.current_node.time, travel_time),
        //             resource: self.current_node.block
        //         }
        //     ));

        //     self.total_nodes += 1;
        //     self.queued_nodes.push(Rc::new(TrainSolverNode {
        //         block: TRAIN_FINISHED,
        //         time: self.current_node.time + travel_time,
        //         depth: self.current_node.depth + 1,
        //         parent: Some(self.current_node.clone()),
        //     }))
        // }

        // debug!("Train step");
        // if self.current_node.track == TRAIN_FINISHED {
        //     // Detect solution.
        //     debug!("  solution detected");

        //     self.solution = Some(Ok(self.current_node.clone()));
        //     return;
        // }

        // if self.current_node.track == SENTINEL_TRACK {
        //     // First route segment.

        //     // This is special casing the first track segment,
        //     // and requiring it to start exactly at the `arrives_at` time.
        //     // TODO generalize this to be treated in the same way
        //     // TODO add general latest_departure?

        //     debug!("  First route segment");

        //     for (track, _tr) in self
        //         .problem
        //         .tracks
        //         .iter()
        //         .enumerate()
        //         .filter(|(_, tr)| tr.prevs.is_empty())
        //     {
        //         trace!(" candidate first route {}", track);
        //         let interval = TimeInterval::duration(
        //             self.current_node.time,
        //             self.problem.tracks[track].travel_time,
        //         );
        //         if block_available(
        //             &self.blocks,
        //             ResourceBlocked {
        //                 resource: track as TrackRef,
        //                 interval,
        //             },
        //         ) {
        //             trace!("  adding node {} {}", track, self.current_node.time);
        //             self.total_nodes += 1;

        //             self.queued_nodes.push(Rc::new(TrainSolverNode {
        //                 track: track as TrackRef,
        //                 time: self.current_node.time,
        //                 depth: self.current_node.depth + 1,
        //                 parent: Some(self.current_node.clone()),
        //             }));
        //         }
        //     }
        // } else {
        //     // We haven't reached a solution yet, so we expand the node.

        //     assert!(self.current_node.track >= 0);
        //     let current_track = &self.problem.tracks[self.current_node.track as usize];
        //     trace!(
        //         "Currentr track {} has next tracks {:?}",
        //         self.current_node.track,
        //         current_track.nexts
        //     );
        //     for next_track in current_track.nexts.iter() {
        //         trace!(
        //             "  - next track {} has valid transfer times {:?}",
        //             next_track,
        //             valid_transfer_times(
        //                 &self.problem,
        //                 &self.blocks,
        //                 self.current_node.track,
        //                 self.current_node.time,
        //                 *next_track,
        //             )
        //             .collect::<Vec<_>>()
        //         );

        //         for time in valid_transfer_times(
        //             &self.problem,
        //             &self.blocks,
        //             self.current_node.track,
        //             self.current_node.time,
        //             *next_track,
        //         ) {
        //             trace!("  adding node {} {}", next_track, time);
        //             self.total_nodes += 1;
        //             self.queued_nodes.push(Rc::new(TrainSolverNode {
        //                 time,
        //                 track: *next_track,
        //                 parent: Some(self.current_node.clone()),
        //                 depth: self.current_node.depth + 1,
        //             }));
        //         }
        //     }

        //     if current_track.nexts.is_empty() {
        //         // TODO special casing the last track segments.
        //         // This should be replaced by explicitly linking to TRAIN_FINISHED
        //         trace!("  adding train finished node");
        //         assert!(self.current_node.track >= 0);
        //         let travel_time = self.problem.tracks[self.current_node.track as usize].travel_time;

        //         // We should already have made sure that the minium travel time is available in the resource(s).
        //         assert!(block_available(
        //             &self.blocks,
        //             ResourceBlocked {
        //                 interval: TimeInterval::duration(self.current_node.time, travel_time),
        //                 resource: self.current_node.track
        //             }
        //         ));

        //         self.total_nodes += 1;
        //         self.queued_nodes.push(Rc::new(TrainSolverNode {
        //             track: TRAIN_FINISHED,
        //             time: self.current_node.time + travel_time,
        //             depth: self.current_node.depth + 1,
        //             parent: Some(self.current_node.clone()),
        //         }))
        //     }
        // }

        self.next_node(use_resource);
    }

    pub fn select_node(&mut self) -> Option<Rc<TrainSolverNode>> {
        trace!("  select node");
        self.queued_nodes.pop()
    }

    pub fn add_constraint(
        &mut self,
        resource: ResourceRef,
        interval: TimeInterval,
        use_resource: impl FnMut(bool, ResourceRef, TimeInterval),
    ) {
        debug!("train add constraint {:?}", (resource, interval));
        self.blocks.insert(ResourceBlocked { interval, resource });
        // TODO incremental algorithm
        self.reset(use_resource);
    }

    pub fn remove_constraint(
        &mut self,
        resource: ResourceRef,
        interval: TimeInterval,
        use_resource: impl FnMut(bool, ResourceRef, TimeInterval),
    ) {
        debug!("train remove constraint {:?}", (resource, interval));
        assert!(self.blocks.remove(&ResourceBlocked { interval, resource }));
        // TODO incremental algorithm
        self.reset(use_resource);
    }

    pub fn next_node(&mut self, use_resource: impl FnMut(bool, ResourceRef, TimeInterval)) {
        if let Some(new_node) = self.select_node() {
            self.switch_node(new_node, use_resource);
        } else {
            warn!("Train has no more nodes.");
            self.solution = Some(Err(()));
        }
    }

    fn switch_node(
        &mut self,
        new_node: Rc<TrainSolverNode>,
        mut use_resource: impl FnMut(bool, ResourceRef, TimeInterval),
    ) {
        debug!("switching to node {:?}", new_node);
        let mut backward = &self.current_node;
        let mut forward = &new_node;
        loop {
            debug!(
                " backw depth {} forw depth {}",
                backward.depth, forward.depth
            );

            if Rc::ptr_eq(backward, forward) {
                break;
            } else if backward.depth < forward.depth {
                let forward_prev = forward.parent.as_ref().unwrap();
                for (resource, interval) in occupations(
                    &self.train.blocks[forward_prev.block as usize],
                    forward_prev.time,
                    forward.time,
                ) {
                    debug!("train adds resource use {:?}", (resource, interval));
                    use_resource(true, resource, interval);
                }
                forward = forward_prev;
            } else {
                let backward_prev = backward.parent.as_ref().unwrap();
                for (resource, interval) in occupations(
                    &self.train.blocks[backward_prev.block as usize],
                    backward_prev.time,
                    backward.time,
                ) {
                    debug!("train removes resource use {:?}", (resource, interval));
                    use_resource(true, resource, interval);
                }
                backward = backward_prev;
            }
        }
        self.current_node = new_node;
    }
}

fn block_available(blocks: &BTreeSet<ResourceBlocked>, block: ResourceBlocked) -> bool {
    let range = ResourceBlocked {
        resource: block.resource,
        interval: INTERVAL_MIN,
    }..ResourceBlocked {
        resource: block.resource + 1,
        interval: INTERVAL_MIN,
    };

    !blocks.range(range).any(|other| {
        assert!(other.resource == block.resource);
        other.interval.overlap(&block.interval)
    })
}

fn latest_exit(blocks: &BTreeSet<ResourceBlocked>, r: ResourceRef, t: TimeValue) -> Option<TimeValue> {
    let range = ResourceBlocked {
        resource: r,
        interval: TimeInterval::duration(t, 0),
    }..ResourceBlocked {
        resource: r + 1,
        interval: INTERVAL_MIN,
    };

    blocks.range(range).next().map(|first_interval| {
        assert!(first_interval.resource == r);
        first_interval.interval.time_start
    })
}

fn valid_transfer_times<'a>(
    train: &'_ Train,
    blocks: &'a BTreeSet<ResourceBlocked>,
    b1: BlockRef,
    t1: TimeValue,
    r2: BlockRef,
) -> impl Iterator<Item = TimeValue> + 'a {
    // TODO, experiment: we could store the latest_exit on the node when it is first generated.
    trace!("valid transfer times r1={} t1={} r2={}", b1, t1, r2);
    let latest_exit = latest_exit(blocks, b1, t1).unwrap_or(TimeValue::MAX);
    trace!("  latest_exit {}", latest_exit);

    let travel_time = train.blocks[b1 as usize].minimum_travel_time;

    let earliest_entry = t1 + travel_time;
    trace!("  earliest_entry {}", earliest_entry);

    assert!(latest_exit >= earliest_entry);

    // Search through blockings on the r2.

    let next_travel_time = train.blocks[r2 as usize].minimum_travel_time;

    let range = ResourceBlocked {
        resource: r2,
        interval: TimeInterval::duration(earliest_entry, 0),
    }..ResourceBlocked {
        resource: r2 + 1,
        interval: INTERVAL_MIN,
    };

    {
        trace!("range {:?}", range);
        let candidate_start_times = std::iter::once(earliest_entry)
            .chain(blocks.range(range.clone()).map(|b| b.interval.time_end));

        trace!(
            "candidate start times {:?}",
            candidate_start_times.collect::<Vec<_>>()
        );
    }

    let candidate_start_times =
        std::iter::once(earliest_entry).chain(blocks.range(range).map(|b| b.interval.time_end));

    candidate_start_times.filter(move |t2| {
        assert!(*t2 >= earliest_entry);
        *t2 <= latest_exit
            && block_available(
                blocks,
                ResourceBlocked {
                    resource: r2,
                    interval: TimeInterval::duration(*t2, next_travel_time),
                },
            )
    })
}

fn occupations(
    block: &Block,
    t_in: TimeValue,
    t_out: TimeValue,
) -> impl Iterator<Item = (ResourceRef, TimeInterval)> + '_ {
    block.resource_usage.iter().map(move |r| {
        (
            r.resource,
            TimeInterval {
                time_start: t_in,
                time_end: t_out.min(r.release_after),
            },
        )
    })
}
