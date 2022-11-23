use crate::{
    interval::TimeInterval,
    problem::{Block, BlockRef, ResourceRef, TimeValue, Train},
};
use itertools::Itertools;
use log::{debug, info, trace, warn};
use std::rc::Rc;
use tinyvec::TinyVec;

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

#[derive(Default, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct IsOccupied {
    enter_after: TimeValue,
    exit_before: TimeValue,
}

pub struct TrainSolver {
    pub train: Train,
    pub occupied: Vec<TinyVec<[IsOccupied; 4]>>,

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

        let occupied = train.blocks.iter().map(|_| Default::default()).collect();

        Self {
            current_node: root_node.clone(),
            root_node,
            train,
            queued_nodes: vec![],
            solution: None,
            occupied,
            total_nodes: 1,
        }
    }

    pub fn current_solution(&self) -> Vec<TimeValue> {
        let mut node = &self.current_node;
        let mut values = Vec::new();
        while let Some(parent) = node.parent.as_ref() {
            values.push(node.time);
            node = parent;
        }
        values.reverse();
        values
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
        let _p = hprof::enter("train step");
        assert!(matches!(self.status(), TrainSolverStatus::Working));

        let nexts = &self.train.blocks[self.current_node.block as usize].nexts;
        if nexts.is_empty() {
            info!("Train solved.");
            self.solution = Some(Ok(self.current_node.clone()));
        } else {
            for next_block in nexts.iter() {
                let succ = successor_nodes(
                    &self.occupied,
                    &self.train.blocks,
                    self.current_node.block,
                    self.current_node.time,
                    *next_block,
                );

                let succ = succ.collect::<Vec<_>>();
                trace!(
                    "  - next track {} has valid transfer times {:?}",
                    next_block,
                    succ
                );

                for time in succ {
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
            
            // TODO special-case DFS without putting the node on the queue.
            self.next_node(use_resource);
        }
    }

    pub fn select_node(&mut self) -> Option<Rc<TrainSolverNode>> {
        trace!("  select node");
        self.queued_nodes.sort_by_key(|x| -(x.time as i32));
        self.queued_nodes.pop()
    }

    pub fn resource_to_blocks(
        train: &Train,
        resource: ResourceRef,
    ) -> impl Iterator<Item = BlockRef> + '_ {
        // TODO Prepare lookup table for performance?
        train.blocks.iter().enumerate().filter_map(move |(idx, b)| {
            (b.resource_usage.iter().any(|r| r.resource == resource)).then_some(idx as BlockRef)
        })
    }

    pub fn set_occupied(
        &mut self,
        resource: ResourceRef,
        enter_after: TimeValue,
        exit_before: TimeValue,
        use_resource: impl FnMut(bool, ResourceRef, TimeInterval),
    ) {
        for block in Self::resource_to_blocks(&self.train, resource) {
            debug!(
                "train add constraint for resource {} block {:?}",
                resource,
                (block, enter_after, exit_before)
            );

            let new_occ = IsOccupied {
                enter_after,
                exit_before,
            };

            let occ_list = &mut self.occupied[block as usize];
            let index = occ_list.binary_search(&new_occ).unwrap_err();
            self.occupied[block as usize].insert(index, new_occ);
        }
        // TODO incremental algorithm
        self.reset(use_resource);
    }

    pub fn remove_occupied(
        &mut self,
        resource: ResourceRef,
        enter_after: TimeValue,
        exit_before: TimeValue,
        use_resource: impl FnMut(bool, ResourceRef, TimeInterval),
    ) {
        for block in Self::resource_to_blocks(&self.train, resource) {
            debug!(
                "train remove constraint for resource {} block {:?}",
                resource,
                (block, enter_after, exit_before)
            );
            let len_before = self.occupied[block as usize].len();
            self.occupied[block as usize]
                .retain(|o| o.enter_after != enter_after && o.exit_before != exit_before);
            assert!(self.occupied[block as usize].len() + 1 == len_before);
        }
        // TODO incremental algorithm
        self.reset(use_resource);
    }

    pub fn next_node(&mut self, use_resource: impl FnMut(bool, ResourceRef, TimeInterval)) {
        if let Some(new_node) = self.select_node() {
            self.switch_node(new_node, use_resource);
        } else {
            warn!("Train has no more nodes.");
            if self.solution.is_none() {
                self.solution = Some(Err(()));
            }
        }
    }

    fn switch_node(
        &mut self,
        new_node: Rc<TrainSolverNode>,
        mut use_resource: impl FnMut(bool, ResourceRef, TimeInterval),
    ) {
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
                        time_end: t_out.min(t_in + r.release_after),
                    },
                )
            })
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
                    use_resource(false, resource, interval);
                }
                backward = backward_prev;
            }
        }
        self.current_node = new_node;
    }
}

fn successor_nodes<'a>(
    occupied: &'a [TinyVec<[IsOccupied; 4]>],
    blocks: &'a [Block],
    prev_block_idx: BlockRef,
    prev_block_entry: TimeValue,
    next_block_idx: BlockRef,
) -> impl Iterator<Item = TimeValue> + 'a {
    let prev_block = &blocks[prev_block_idx as usize];
    let next_block = &blocks[next_block_idx as usize];

    let earliest_exit_prev =
        (prev_block_entry + prev_block.minimum_travel_time).max(next_block.earliest_start);

    let latest_exit_prev = occupied[prev_block_idx as usize]
        .iter()
        .filter_map(|c| (prev_block_entry < c.enter_after).then_some(c.exit_before))
        .min()
        .unwrap_or(TimeValue::MAX);

    // Candidates, in order of increasing cost.
    let candidate_times = std::iter::once(next_block.aimed_start)
        .chain(std::iter::once(earliest_exit_prev))
        .chain(
            occupied[next_block_idx as usize]
                .iter()
                .map(|r| r.enter_after),
        );

    candidate_times
        .filter(move |c| *c >= earliest_exit_prev && *c < latest_exit_prev)
        .dedup()
}
