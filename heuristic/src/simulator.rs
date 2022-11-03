use crate::problem::*;

pub fn objective_value(problem: &Problem, schedule: &Schedule) -> u32 {
    assert!(problem.trains.len() == schedule.len());
    problem
        .trains
        .iter()
        .zip(schedule.iter())
        .map(|(train, train_schedule)| {
            assert!(train.visits.len() == train_schedule.len());
            train
                .visits
                .iter()
                .zip(train_schedule.iter())
                .map(|(visit, visit_time)| {
                    visit
                        .measure_delay
                        .map(|d| (*visit_time as i32 - d).max(0))
                        .unwrap_or(0i32)
                })
                .sum::<i32>()
        })
        .sum::<i32>() as u32
}

pub enum DispatchingResult {
    Schedule(Schedule),
    Conflict(TrackRef, (TrainRef, TimeValue), (TrainRef, TimeValue)),
    Stuck,
}

pub enum Constraint {
    Disable(TrainRef, TrackRef),
    WaitFor(TrainRef, VisitRef, TrainRef, VisitRef),
}

pub struct Simulator<'a> {
    problem: &'a Problem,
}

impl<'a> Simulator<'a> {
    pub fn for_problem(problem: &'a Problem) -> Self {
        Self { problem }
    }

    pub fn push_constraint(&mut self, constraint: Constraint) {
        todo!()
    }

    pub fn pop_constraint(&mut self) {
        todo!()
    }

    pub fn run(&mut self) -> DispatchingResult {
        let problem = &self.problem;

        enum Event {
            TrainVisit(TrainRef, VisitRef),
        }

        struct TrainStatus {
            occupies_resource: TrackRef,
        }

        let mut event_queue = Vec::new();
        let mut resource_occupation = (0..problem.tracks.len())
            .map(|_| SENTINEL_TRAIN)
            .collect::<Vec<_>>();
        let mut train_status = (0..problem.trains.len())
            .map(|_| TrainStatus {
                occupies_resource: SENTINEL_TRACK,
            })
            .collect::<Vec<_>>();

        for (trainref, train) in problem.trains.iter().enumerate() {
            event_queue.push((
                train.appears_at,
                Event::TrainVisit(trainref as TrainRef, 0 as VisitRef),
            ));
        }

        loop {

            if event_queue.is_empty() {
                // TODO if trains are finished, return schedule.
                // if not, return stuck.
                return DispatchingResult::Stuck;
            }

            event_queue.sort_by_key(|(t, _)| *t);
            let (time, event) = event_queue.remove(0);
            match event {
                Event::TrainVisit(trainref, visitref) => {
                    let visit = &problem.trains[trainref as usize].visits[visitref as usize];

                    // Choose a path.
                    if let Some(next_res) = visit
                        .resource_alternatives
                        .iter()
                        .find(|track| resource_occupation[**track as usize] == SENTINEL_TRAIN)
                    {
                        // Unreserve the previous resource
                        let prev_res = train_status[trainref as usize].occupies_resource;
                        if prev_res != SENTINEL_TRACK {
                            resource_occupation[prev_res as usize] = SENTINEL_TRAIN;
                        }

                        // Reserve next resource.
                        let resource = &problem.tracks[*next_res as usize];
                        resource_occupation[*next_res as usize] = trainref;

                        // Stay until travel time has elapsed.
                        event_queue.push((
                            time + resource.travel_time,
                            Event::TrainVisit(trainref, visitref + 1),
                        ));
                    } else {
                        // The train is blocked, and we need to create a search branch.
                    }
                }
            }
        }
        todo!()
    }
}

fn no_branching(problem: &Problem) -> DispatchingResult {
    let mut simulator = Simulator::for_problem(problem);
    loop {
        let status = simulator.run();
        match status {
            DispatchingResult::Conflict(_, _, _) => todo!(),
            _ => {
                return status;
            }
        }
    }
}
