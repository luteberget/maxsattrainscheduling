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
        track: TrackRef,
        interval: TimeInterval,
    ) {
        let occ = ResourceOccupation { train, interval };
        if add {
            trace!(
                "ADD train{} track{} [{} -> {}] ",
                train,
                track,
                occ.interval.time_start,
                occ.interval.time_end
            );
            self.add(track as usize, occ);
        } else {
            trace!(
                "DEL train{} track{} [{} -> {}] ",
                train,
                track,
                occ.interval.time_start,
                occ.interval.time_end
            );
            self.remove(track as usize, occ);
        }
    }
}

#[derive(Debug)]
pub enum ConflictSolverStatus {
    Exhausted,
    SelectNode,
    Conflict,
    SolveTrains,
}

#[derive(Debug)]
pub struct ConflictConstraint {
    pub train: TrainRef,
    pub track: TrackRef,
    pub interval: TimeInterval,
}

pub struct ConflictSolverNode {
    pub constraint: ConflictConstraint,
    pub depth: u32,
    pub parent: Option<Rc<ConflictSolverNode>>,
}

impl std::fmt::Debug for ConflictSolverNode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ConflictSolverNode")
            .field("constraint", &self.constraint)
            .field("depth", &self.depth)
            .field("has_parent", &self.parent.is_some())
            .finish()
    }
}

pub struct ConflictSolver {
    pub problem: Rc<Problem>,
    pub trains: Vec<TrainSolver>,
    pub conflicts: ResourceConflicts,
    pub queued_nodes: Vec<Rc<ConflictSolverNode>>,
    pub current_node: Option<Rc<ConflictSolverNode>>, // The root node is "None".
    pub dirty_trains: Vec<u32>,
    pub dirty_train_idxs: Vec<i32>,
}

impl ConflictSolver {
    pub fn status(&self) -> ConflictSolverStatus {
        match (
            self.dirty_trains.is_empty(),
            self.conflicts.conflicting_resource_set.is_empty(),
            self.queued_nodes.is_empty(),
        ) {
            (false, _, _) => ConflictSolverStatus::SolveTrains,
            (_, false, _) => ConflictSolverStatus::Conflict,
            (_, _, false) => ConflictSolverStatus::SelectNode,
            (true, true, true) => ConflictSolverStatus::Exhausted,
        }
    }

    pub fn step(&mut self) {
        // TODO possibly resolve conflicts before finishing solving all trains.

        if let Some(dirty_train) = self.dirty_trains.last() {
            let dirty_train_idx = *dirty_train as usize;
            match self.trains[dirty_train_idx].status() {
                TrainSolverStatus::Failed => {
                    debug!("Train {} failed.", dirty_train_idx);
                    self.switch_node();
                }
                TrainSolverStatus::Optimal => {
                    debug!("Train {} optimal.", dirty_train_idx);
                    Self::remove_dirty_train(
                        &mut self.dirty_trains,
                        &mut self.dirty_train_idxs,
                        dirty_train_idx,
                    );
                }
                TrainSolverStatus::Working => {
                    self.trains[dirty_train_idx].step(|add, track, interval| {
                        self.conflicts.add_or_remove(
                            add,
                            dirty_train_idx as TrainRef,
                            track,
                            interval,
                        )
                    });
                }
            }
        } else if !self.conflicts.conflicting_resource_set.is_empty() {
            self.branch();
        } else if !self.queued_nodes.is_empty() {
            debug!("Switching node");
            self.switch_node();
        } else {
            debug!("Search exhausted.")
        }
    }

    /// Select a conflict and create constraints for it.
    fn branch(&mut self) {
        assert!(!self.conflicts.conflicting_resource_set.is_empty());
        let conflict_resource = self.conflicts.conflicting_resource_set[0];
        // println!("CONFLICTS {:#?}", self.conflicts);
        let (occ_a, occ_b) = self.conflicts.resources[conflict_resource as usize]
            .get_conflict()
            .unwrap();

        // If both trains are running at their minimum travel time, then
        // we can create valid conflicting intervals.

        // TODO: remove assumptions that the trains are as early as possible to every event.

        let block_a = TimeInterval {
            time_start: occ_a.interval.time_start,
            time_end: occ_b.interval.time_end,
        };

        let block_b = TimeInterval {
            time_start: occ_b.interval.time_start,
            time_end: occ_a.interval.time_end,
        };

        assert!(block_a.overlap(&block_b));

        let constraint_a = ConflictConstraint {
            train: occ_a.train,
            track: conflict_resource as TrackRef,
            interval: block_a,
        };

        let constraint_b = ConflictConstraint {
            train: occ_b.train,
            track: conflict_resource as TrackRef,
            interval: block_b,
        };

        debug!(
            "Branch:\n - a: {:?}\n - b: {:?}",
            constraint_a, constraint_b
        );

        let node_a = ConflictSolverNode {
            constraint: constraint_a,
            depth: self.current_node.as_ref().map_or(0, |n| n.depth) + 1,
            parent: self.current_node.clone(),
        };

        let node_b = ConflictSolverNode {
            constraint: constraint_b,
            depth: self.current_node.as_ref().map_or(0, |n| n.depth) + 1,
            parent: self.current_node.clone(),
        };

        self.queued_nodes.push(Rc::new(node_a));
        self.queued_nodes.push(Rc::new(node_b));

        // TODO special-case DFS without putting the node on the queue.

        self.switch_node();
    }

    fn select_node(&mut self) -> Option<Rc<ConflictSolverNode>> {
        self.queued_nodes.pop()
    }

    fn switch_node(&mut self) {
        let new_node = self.select_node().unwrap();
        debug!("conflict search switching to node {:?}", new_node);
        let mut backward = self.current_node.as_ref();
        let mut forward = Some(&new_node);

        loop {
            // trace!(
            //     "Comparing backward depth1 {} to forawrd depth2 {}",
            //     depth1,
            //     depth2
            // );

            let same_node = match (backward, forward) {
                (Some(rc1), Some(rc2)) => Rc::ptr_eq(rc1, rc2),
                (None, None) => true,
                _ => false,
            };
            if same_node {
                break;
            } else {
                let depth1 = backward.as_ref().map_or(0, |n| n.depth);
                let depth2 = forward.as_ref().map_or(0, |n| n.depth);
                if depth1 < depth2 {
                    let node = forward.unwrap();

                    // Add constraint
                    let train = node.constraint.train;
                    self.trains[train as usize].add_constraint(
                        node.constraint.track,
                        node.constraint.interval,
                        |a, t, i| self.conflicts.add_or_remove(a, train, t, i),
                    );
                    Self::add_dirty_train(
                        &mut self.dirty_trains,
                        &mut self.dirty_train_idxs,
                        train as usize,
                    );

                    forward = node.parent.as_ref();
                } else {
                    let node = backward.unwrap();

                    // Remove constraint
                    let train = node.constraint.train;

                    self.trains[train as usize].remove_constraint(
                        node.constraint.track,
                        node.constraint.interval,
                        |a, t, i| self.conflicts.add_or_remove(a, train, t, i),
                    );
                    Self::add_dirty_train(
                        &mut self.dirty_trains,
                        &mut self.dirty_train_idxs,
                        train as usize,
                    );

                    backward = node.parent.as_ref();
                }
            }
        }

        self.current_node = Some(new_node);
    }

    fn remove_dirty_train(
        dirty_trains: &mut Vec<u32>,
        dirty_train_idxs: &mut [i32],
        train_idx: usize,
    ) {
        assert!(dirty_train_idxs[train_idx] >= 0);

        let dirty_idx = dirty_train_idxs[train_idx] as usize;
        dirty_trains.swap_remove(dirty_idx);
        if dirty_idx < dirty_trains.len() {
            let other_train = dirty_trains[dirty_idx] as usize;
            dirty_train_idxs[other_train] = dirty_idx as i32;
        }
        dirty_train_idxs[train_idx] = -1;
    }

    fn add_dirty_train(
        dirty_trains: &mut Vec<u32>,
        dirty_train_idxs: &mut [i32],
        train_idx: usize,
    ) {
        if dirty_train_idxs[train_idx] == -1 {
            dirty_train_idxs[train_idx] = dirty_trains.len() as i32;
            dirty_trains.push(train_idx as u32);
        }
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
        let dirty_trains = (0..(trains.len() as u32)).collect();
        let dirty_train_idxs = (0..(trains.len() as i32)).collect();
        Self {
            problem,
            trains,
            conflicts,
            current_node: None,
            dirty_trains,
            dirty_train_idxs,
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
    pub fn new(problem: Rc<Problem>, train_idx: usize, blocks: BTreeSet<Block>) -> Self {
        let root_node = Self::start_node(&problem, train_idx);
        Self {
            current_node: root_node.clone(),
            root_node,
            problem,
            train_idx,
            queued_nodes: vec![],
            solution: None,
            blocks,
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
