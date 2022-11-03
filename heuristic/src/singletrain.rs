use std::{collections::HashMap, rc::Rc};

use crate::problem::*;

pub struct TimeInterval {
    time_start: TimeValue,
    time_end: TimeValue,
}

pub const INTERVAL_MAX: TimeInterval = TimeInterval {
    time_start: TimeValue::MAX,
    time_end: TimeValue::MAX,
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

pub type Constraints = HashMap<TrackRef, Vec<TimeInterval>>;

pub type TrainSchedule = Vec<(TrackRef, TimeValue)>;

pub fn singletrain_optimal_path_under_constraints(
    train: &Train,
    tracks: &[Track],
    constraints: &Constraints,
) -> TrainSchedule {
    #[derive(Clone)]
    struct TrainState {
        time: TimeValue,
        resource: TrackRef,
        next_visit: usize,
        parent_node: Option<Rc<TrainState>>,
    }

    let mut queue = vec![Rc::new(TrainState {
        time: train.appears_at,
        resource: SENTINEL_TRACK,
        next_visit: 0,
        parent_node: None,
    })];

    while let Some(state) = queue.pop() {
        if state.next_visit < train.visits.len() {
            let visit = &train.visits[state.next_visit];
            for alt_res in visit.resource_alternatives.iter() {
                let running_time = tracks[*alt_res as usize].travel_time;
                let mut time = state.time + tracks[state.resource as usize].travel_time;
                let c = constraints.get(alt_res);

                let blocks = c
                    .iter()
                    .flat_map(move |v| v.iter())
                    .chain(std::iter::once(&INTERVAL_MAX));

                for block in blocks {
                    if block.time_end <= time {
                        continue;
                    }

                    if !TimeInterval::duration(time, running_time).overlap(block) {
                        // Check for blocks in the previous interval
                        let prev_interval = TimeInterval {
                            time_start: state.time,
                            time_end: time,
                        };
                        if constraints
                            .get(&state.resource)
                            .iter()
                            .flat_map(|v| v.iter())
                            .any(|b| b.overlap(&prev_interval))
                        {
                            continue;
                        }

                        queue.push(Rc::new(TrainState {
                            time,
                            resource: *alt_res,
                            next_visit: state.next_visit + 1,
                            parent_node: Some(state.clone()),
                        }));
                    }
                    time = block.time_end;
                }
                assert!(time == TimeValue::MAX);
            }
        } else {
            // Finished.
            let mut state = &state;
            let mut vec = vec![(
                SENTINEL_TRACK,
                state.time + tracks[state.resource as usize].travel_time,
            )];
            loop {
                vec.push((state.resource, state.time));
                if let Some(p) = state.parent_node.as_ref() {
                    state = p;
                } else {
                    break;
                }
            }

            return vec;
        }
    }
    panic!("no solution");
}
