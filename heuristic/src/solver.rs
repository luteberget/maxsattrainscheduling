use crate::problem::*;
use log::{debug, trace, warn};
use std::{collections::BTreeSet, rc::Rc};
use tinyvec::TinyVec;

#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Debug)]
pub struct TimeInterval {
    pub time_start: TimeValue,
    pub time_end: TimeValue,
}

impl Default for TimeInterval {
    fn default() -> Self {
        INTERVAL_MAX
    }
}

pub const INTERVAL_MAX: TimeInterval = TimeInterval {
    time_start: TimeValue::MAX,
    time_end: TimeValue::MAX,
};

pub const INTERVAL_MIN: TimeInterval = TimeInterval {
    time_start: TimeValue::MIN,
    time_end: TimeValue::MIN,
};

impl TimeInterval {
    pub fn duration(start: TimeValue, duration: TimeValue) -> TimeInterval {
        TimeInterval {
            time_start: start,
            time_end: start + duration,
        }
    }

    pub fn overlap(&self, other: &Self) -> bool {
        !(self.time_end <= other.time_start || other.time_end <= self.time_start)
    }
}

pub struct ResourceConflicts {
    pub conflicting_resource_set: Vec<u32>,
    pub resources: Vec<ResourceOccupations>,
}

pub struct ResourceOccupations {
    pub conflicting_resource_set_idx: i32,
    pub occupations: TinyVec<[ResourceOccupation; 32]>,
}

impl ResourceOccupations {
    pub fn has_conflict(&self) -> bool {
        // TODO this is not correct (but fast)
        self.occupations
            .iter()
            .zip(self.occupations.iter().skip(1))
            .any(|(a, b)| a.interval.overlap(&b.interval))
    }
}

#[derive(Debug, Default, PartialEq, Eq, PartialOrd, Ord)]
pub struct ResourceOccupation {
    pub train: TrainRef,
    pub interval: TimeInterval,
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
                if resource.conflicting_resource_set_idx < 0 && resource.has_conflict() {
                    resource.conflicting_resource_set_idx =
                        self.conflicting_resource_set.len() as i32;
                    self.conflicting_resource_set.push(idx as u32);
                }
            }
        }
    }

    pub fn remove(&mut self, resource_idx: usize, occ: ResourceOccupation) {
        let resource = &mut self.resources[resource_idx];
        let idx = resource.occupations.binary_search(&occ).unwrap();
        resource.occupations.remove(idx);
        if resource.conflicting_resource_set_idx >= 0 && !resource.has_conflict() {
            self.conflicting_resource_set
                .swap_remove(resource.conflicting_resource_set_idx as usize);
            resource.conflicting_resource_set_idx = -1;
            if idx < self.conflicting_resource_set.len() {
                self.resources[self.conflicting_resource_set[idx] as usize]
                    .conflicting_resource_set_idx = idx as i32;
            }
        }
    }
}

#[derive(Debug)]
pub enum ConflictSolverStatus {
    Exhausted,
    SelectNode,
    SolveTrains,
}

pub struct ConflictSolverNode {}

pub struct ConflictSolver {
    pub problem: Rc<Problem>,
    pub trains: Vec<TrainSolver>,
    pub conflicts: ResourceConflicts,

    pub queued_nodes: Vec<Rc<ConflictSolverNode>>,

    pub current_node: Rc<ConflictSolverNode>,
    pub dirty_trains: Vec<u32>,
}

#[derive(Default)]
pub struct Conflict {}

pub struct TrackConflicts {
    pub conflicts: TinyVec<[Conflict; 4]>,
}

impl ConflictSolver {
    pub fn status(&self) -> ConflictSolverStatus {
        match (self.dirty_trains.is_empty(), self.queued_nodes.is_empty()) {
            (false, _) => ConflictSolverStatus::SolveTrains,
            (_, false) => ConflictSolverStatus::SelectNode,
            (true, true) => ConflictSolverStatus::Exhausted,
        }
    }

    pub fn step(&mut self) {
        if let Some(dirty_train) = self.dirty_trains.last() {
            match self.trains[*dirty_train as usize].status() {
                TrainSolverStatus::Failed => {
                    debug!("Train {} failed.", dirty_train);
                    self.pick_node();
                }
                TrainSolverStatus::Optimal => {
                    debug!("Train {} optimal.", dirty_train);
                    self.dirty_trains.pop();
                }
                TrainSolverStatus::Working => {
                    self.trains[*dirty_train as usize].step(|add, track, interval| {
                        let occ = ResourceOccupation {
                            train: *dirty_train as i32,
                            interval,
                        };
                        if add {
                            trace!(
                                "ADD train{} track{} [{} -> {}] ",
                                dirty_train,
                                track,
                                occ.interval.time_start,
                                occ.interval.time_end
                            );
                            self.conflicts.add(track as usize, occ);
                        } else {
                            trace!(
                                "DEL train{} track{} [{} -> {}] ",
                                dirty_train,
                                track,
                                occ.interval.time_start,
                                occ.interval.time_end
                            );
                            self.conflicts.remove(track as usize, occ);
                        }
                    });
                }
            }
        } else if !self.conflicts.conflicting_resource_set.is_empty() {
            self.branch();
        } else if !self.queued_nodes.is_empty() {
            self.pick_node();
        } else {
            debug!("Search exhausted.")
        }
    }

    /// Select a conflict and create constraints for it.
    fn branch(&mut self) {
        todo!()
    }

    fn pick_node(&mut self) {
        todo!()
    }

    pub fn new(problem: Rc<Problem>) -> Self {
        let trains: Vec<TrainSolver> = problem
            .trains
            .iter()
            .enumerate()
            .map(|(train_idx, _t)| {
                TrainSolver::new(
                    problem.clone(),
                    train_idx,
                    Default::default(), /* Empty blocking set */
                )
            })
            .collect();
        let conflicts = ResourceConflicts::empty(problem.tracks.len());
        let start_node = ConflictSolverNode {};
        let dirty_trains = (0..(trains.len() as u32)).collect();

        Self {
            problem,
            trains,
            conflicts,
            current_node: Rc::new(start_node),
            dirty_trains,
            queued_nodes: vec![],
        }
    }
}

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

#[derive(PartialEq, Eq, PartialOrd, Ord, Debug)]
pub struct Block {
    track: TrackRef,
    interval: TimeInterval,
}

pub struct TrainSolver {
    pub problem: Rc<Problem>,
    pub train_idx: usize,
    pub blocks: BTreeSet<Block>,

    pub queued_nodes: Vec<Rc<TrainSolverNode>>,
    pub current_node: Rc<TrainSolverNode>,
    pub solution: Option<Result<Rc<TrainSolverNode>, ()>>,
}

impl TrainSolver {
    pub fn new(problem: Rc<Problem>, train_idx: usize, blocks: BTreeSet<Block>) -> Self {
        let current_node = Rc::new(TrainSolverNode {
            time: problem.trains[train_idx].appears_at,
            track: SENTINEL_TRACK,
            parent: None,
            depth: 0,
        });

        Self {
            problem,
            train_idx,
            queued_nodes: vec![],
            current_node,
            solution: None,
            blocks,
        }
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
        } else if self.current_node.track == SENTINEL_TRACK {
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
            for next_track in current_track.nexts.iter() {
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
        }

        self.switch_node(use_resource);
    }

    pub fn select_node(&mut self) -> Option<Rc<TrainSolverNode>> {
        trace!("  select node");
        self.queued_nodes.pop()
    }

    pub fn switch_node(&mut self, mut use_resource: impl FnMut(bool, TrackRef, TimeInterval)) {
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

        if let Some(new_node) = self.select_node() {
            debug!("switching to node {:?}", new_node);
            let mut backward = &self.current_node;
            let mut forward = &new_node;

            loop {
                match backward.depth.cmp(&forward.depth) {
                    std::cmp::Ordering::Less => {
                        if let Some((track, interval)) = occupation(forward) {
                            use_resource(true, track, interval);
                        }
                        forward = forward.parent.as_ref().unwrap();
                    }
                    std::cmp::Ordering::Greater => {
                        if let Some((track, interval)) = occupation(forward) {
                            use_resource(false, track, interval);
                        }
                        backward = backward.parent.as_ref().unwrap();
                    }
                    std::cmp::Ordering::Equal => break,
                }
            }

            self.current_node = new_node;
        } else {
            warn!("Train has no more nodes.");
            self.solution = Some(Err(()));
        }
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

    let latest_exit = latest_exit(blocks, r1, t1).unwrap_or(TimeValue::MAX);

    let travel_time = (r1 >= 0)
        .then(|| problem.tracks[r1 as usize].travel_time)
        .unwrap_or(0);

    let earliest_entry = t1 + travel_time;

    assert!(latest_exit >= earliest_entry);

    // Search through blockings on the r2.

    let next_travel_time = (r2 >= 0)
        .then(|| problem.tracks[r2 as usize].travel_time)
        .unwrap_or(0);

    let range = Block {
        track: r1,
        interval: TimeInterval::duration(earliest_entry, 0),
    }..Block {
        track: r1 + 1,
        interval: INTERVAL_MIN,
    };

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
