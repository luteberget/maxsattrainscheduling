use std::{collections::BTreeSet, rc::Rc};

use log::{debug, trace, warn};

use crate::{
    interval::{TimeInterval, INTERVAL_MIN},
    problem::{Problem, TimeValue, TrackRef, SENTINEL_TRACK, TRAIN_FINISHED},
};

#[derive(Debug)]
pub enum TrainSolverStatus {
    Failed,
    Optimal,
    Working,
}

pub struct TrainSolverNode {
    pub track: TrackRef,
    pub time: TimeValue,
    depth: u32,
    pub parent: Option<Rc<TrainSolverNode>>,
}

impl std::fmt::Debug for TrainSolverNode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TrainSolverNode")
            .field("track", &self.track)
            .field("time", &self.time)
            .field("depth", &self.depth)
            .field("has_parent", &self.parent.is_some())
            .finish()
    }
}

#[derive(PartialEq, Eq, PartialOrd, Ord, Debug, Copy, Clone)]
pub struct Block {
    track: TrackRef,
    interval: TimeInterval,
}

pub struct TrainSolver {
    pub problem: Rc<Problem>,
    pub train_idx: usize,
    pub blocks: BTreeSet<Block>,

    root_node: Rc<TrainSolverNode>,
    pub queued_nodes: Vec<Rc<TrainSolverNode>>,
    pub current_node: Rc<TrainSolverNode>,
    pub solution: Option<Result<Rc<TrainSolverNode>, ()>>,
}

impl TrainSolver {
    pub fn new(problem: Rc<Problem>, train_idx: usize) -> Self {
        let root_node = Self::start_node(&problem, train_idx);
        Self {
            current_node: root_node.clone(),
            root_node,
            problem,
            train_idx,
            queued_nodes: vec![],
            solution: None,
            blocks: Default::default(),
        }
    }

    pub fn reset(&mut self, use_resource: impl FnMut(bool, TrackRef, TimeInterval)) {
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

    pub fn step(&mut self, use_resource: impl FnMut(bool, TrackRef, TimeInterval)) {
        assert!(matches!(self.status(), TrainSolverStatus::Working));

        debug!("Train step");
        if self.current_node.track == TRAIN_FINISHED {
            // Detect solution.
            debug!("  solution detected");

            self.solution = Some(Ok(self.current_node.clone()));
            return;
        }

        if self.current_node.track == SENTINEL_TRACK {
            // First route segment.

            // This is special casing the first track segment,
            // and requiring it to start exactly at the `arrives_at` time.
            // TODO generalize this to be treated in the same way
            // TODO add general latest_departure?

            debug!("  First route segment");

            for (track, _tr) in self
                .problem
                .tracks
                .iter()
                .enumerate()
                .filter(|(_, tr)| tr.prevs.is_empty())
            {
                trace!(" candidate first route {}", track);
                let interval = TimeInterval::duration(
                    self.current_node.time,
                    self.problem.tracks[track].travel_time,
                );
                if block_available(
                    &self.blocks,
                    Block {
                        track: track as TrackRef,
                        interval,
                    },
                ) {
                    trace!("  adding node {} {}", track, self.current_node.time);
                    self.queued_nodes.push(Rc::new(TrainSolverNode {
                        track: track as TrackRef,
                        time: self.current_node.time,
                        depth: self.current_node.depth + 1,
                        parent: Some(self.current_node.clone()),
                    }));
                }
            }
        } else {
            // We haven't reached a solution yet, so we expand the node.

            assert!(self.current_node.track >= 0);
            let current_track = &self.problem.tracks[self.current_node.track as usize];
            trace!(
                "Currentr track {} has next tracks {:?}",
                self.current_node.track,
                current_track.nexts
            );
            for next_track in current_track.nexts.iter() {
                trace!(
                    "  - next track {} has valid transfer times {:?}",
                    next_track,
                    valid_transfer_times(
                        &self.problem,
                        &self.blocks,
                        self.current_node.track,
                        self.current_node.time,
                        *next_track,
                    )
                    .collect::<Vec<_>>()
                );

                for time in valid_transfer_times(
                    &self.problem,
                    &self.blocks,
                    self.current_node.track,
                    self.current_node.time,
                    *next_track,
                ) {
                    trace!("  adding node {} {}", next_track, time);
                    self.queued_nodes.push(Rc::new(TrainSolverNode {
                        time,
                        track: *next_track,
                        parent: Some(self.current_node.clone()),
                        depth: self.current_node.depth + 1,
                    }));
                }
            }

            if current_track.nexts.is_empty() {
                // TODO special casing the last track segments.
                // This should be replaced by explicitly linking to TRAIN_FINISHED
                trace!("  adding train finished node");
                assert!(self.current_node.track >= 0);
                let travel_time = self.problem.tracks[self.current_node.track as usize].travel_time;

                // We should already have made sure that the minium travel time is available in the resource(s).
                assert!(block_available(
                    &self.blocks,
                    Block {
                        interval: TimeInterval::duration(self.current_node.time, travel_time),
                        track: self.current_node.track
                    }
                ));

                self.queued_nodes.push(Rc::new(TrainSolverNode {
                    track: TRAIN_FINISHED,
                    time: self.current_node.time + travel_time,
                    depth: self.current_node.depth + 1,
                    parent: Some(self.current_node.clone()),
                }))
            }
        }

        self.next_node(use_resource);
    }

    pub fn select_node(&mut self) -> Option<Rc<TrainSolverNode>> {
        trace!("  select node");
        self.queued_nodes.pop()
    }

    pub fn add_constraint(
        &mut self,
        track: TrackRef,
        interval: TimeInterval,
        use_resource: impl FnMut(bool, TrackRef, TimeInterval),
    ) {
        debug!(
            "train {} add constraint {:?}",
            self.train_idx,
            (track, interval)
        );
        self.blocks.insert(Block { interval, track });
        // TODO incremental algorithm
        self.reset(use_resource);
    }

    pub fn remove_constraint(
        &mut self,
        track: TrackRef,
        interval: TimeInterval,
        use_resource: impl FnMut(bool, TrackRef, TimeInterval),
    ) {
        debug!(
            "train {} remove constraint {:?}",
            self.train_idx,
            (track, interval)
        );
        assert!(self.blocks.remove(&Block { interval, track }));
        // TODO incremental algorithm
        self.reset(use_resource);
    }

    pub fn next_node(&mut self, use_resource: impl FnMut(bool, TrackRef, TimeInterval)) {
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
        mut use_resource: impl FnMut(bool, i32, TimeInterval),
    ) {
        fn occupation(node: &TrainSolverNode) -> Option<(TrackRef, TimeInterval)> {
            let prev_node = node.parent.as_ref()?;
            if prev_node.track < 0 {
                return None;
            }

            Some((
                prev_node.track,
                TimeInterval {
                    time_start: prev_node.time,
                    time_end: node.time,
                },
            ))
        }
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
                if let Some((track, interval)) = occupation(forward) {
                    debug!(
                        "train {} adds resource use {:?}",
                        self.train_idx,
                        (track, interval)
                    );
                    use_resource(true, track, interval);
                }
                forward = forward.parent.as_ref().unwrap();
            } else {
                if let Some((track, interval)) = occupation(backward) {
                    debug!(
                        "train {} removes resource use {:?}",
                        self.train_idx,
                        (track, interval)
                    );
                    use_resource(false, track, interval);
                }
                backward = backward.parent.as_ref().unwrap();
            }
        }
        self.current_node = new_node;
    }

    fn start_node(problem: &Rc<Problem>, train_idx: usize) -> Rc<TrainSolverNode> {
        Rc::new(TrainSolverNode {
            time: problem.trains[train_idx].appears_at,
            track: SENTINEL_TRACK,
            parent: None,
            depth: 0,
        })
    }
}

fn block_available(blocks: &BTreeSet<Block>, block: Block) -> bool {
    let range = Block {
        track: block.track,
        interval: INTERVAL_MIN,
    }..Block {
        track: block.track + 1,
        interval: INTERVAL_MIN,
    };

    !blocks.range(range).any(|other| {
        assert!(other.track == block.track);
        other.interval.overlap(&block.interval)
    })
}

fn latest_exit(blocks: &BTreeSet<Block>, r: TrackRef, t: TimeValue) -> Option<TimeValue> {
    let r = (r >= 0).then_some(r)?;
    let range = Block {
        track: r,
        interval: TimeInterval::duration(t, 0),
    }..Block {
        track: r + 1,
        interval: INTERVAL_MIN,
    };

    blocks.range(range).next().map(|first_interval| {
        assert!(first_interval.track == r);
        first_interval.interval.time_start
    })
}

fn valid_transfer_times<'a>(
    problem: &'_ Problem,
    blocks: &'a BTreeSet<Block>,
    r1: TrackRef,
    t1: TimeValue,
    r2: TrackRef,
) -> impl Iterator<Item = TimeValue> + 'a {
    // TODO, experiment: we could store the latest_exit on the node when it is first generated.
    trace!("valid transfer times r1={} t1={} r2={}", r1, t1, r2);
    let latest_exit = latest_exit(blocks, r1, t1).unwrap_or(TimeValue::MAX);
    trace!("  latest_exit {}", latest_exit);

    let travel_time = (r1 >= 0)
        .then(|| problem.tracks[r1 as usize].travel_time)
        .unwrap_or(0);

    let earliest_entry = t1 + travel_time;
    trace!("  earliest_entry {}", earliest_entry);

    assert!(latest_exit >= earliest_entry);

    // Search through blockings on the r2.

    let next_travel_time = (r2 >= 0)
        .then(|| problem.tracks[r2 as usize].travel_time)
        .unwrap_or(0);

    let range = Block {
        track: r2,
        interval: TimeInterval::duration(earliest_entry, 0),
    }..Block {
        track: r2 + 1,
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
                Block {
                    track: r2,
                    interval: TimeInterval::duration(*t2, next_travel_time),
                },
            )
    })
}
