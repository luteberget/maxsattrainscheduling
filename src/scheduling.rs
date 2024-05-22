use std::{
    collections::{BTreeMap, HashMap, HashSet},
    time::Instant,
};

use log::debug;
use typed_index_collections::TiVec;

use crate::{
    ipamir::{MaxSatError, MaxSatSolver},
    problem::{DelayCostType, Problem},
};

#[derive(Clone, Copy, Debug)]
pub enum SolverError {
    NoSolution,
    Timeout,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
struct VisitId(u32);

impl From<VisitId> for usize {
    fn from(v: VisitId) -> Self {
        v.0 as usize
    }
}

impl From<usize> for VisitId {
    fn from(x: usize) -> Self {
        VisitId(x as u32)
    }
}

#[derive(Clone, Copy, Debug)]
struct ResourceId(u32);

impl From<ResourceId> for usize {
    fn from(v: ResourceId) -> Self {
        v.0 as usize
    }
}

impl From<usize> for ResourceId {
    fn from(x: usize) -> Self {
        ResourceId(x as u32)
    }
}

#[derive(PartialEq, Eq, PartialOrd, Ord, Debug)]
pub enum IterationType {
    Objective,
    TravelTimeConflict,
    ResourceConflict,
    TravelAndResourceConflict,
    Solution,
}

#[derive(Default)]
pub struct SolveStats {
    pub n_sat: usize,
    pub n_unsat: usize,
    pub n_travel: usize,
    pub n_conflict: usize,
    pub satsolver: String,
}

pub fn solve(
    mut solver: impl MaxSatSolver + std::fmt::Debug,
    problem: &Problem,
    timeout: f64,
    delay_cost_type: DelayCostType,
) -> Result<(Vec<Vec<i32>>, SolveStats), SolverError> {

    let start_time: Instant = Instant::now();
    let mut solver_time = std::time::Duration::ZERO;
    let mut stats = SolveStats::default();

    let mut visits: TiVec<VisitId, (usize, usize)> = TiVec::new();
    let mut resource_visits: Vec<Vec<VisitId>> = Vec::new();
    let mut occupations: TiVec<VisitId, Occ> = TiVec::new();
    let mut touched_intervals = Vec::new();
    let mut conflicts: HashMap<usize, Vec<usize>> = HashMap::new();
    let mut new_time_points = Vec::new();

    let mut iteration_types: BTreeMap<IterationType, usize> = BTreeMap::new();

    let mut n_timepoints = 0;
    let mut n_conflict_constraints = 0;

    for (a, b) in problem.conflicts.iter() {
        conflicts.entry(*a).or_default().push(*b);
        if *a != *b {
            conflicts.entry(*b).or_default().push(*a);
        }
    }

    let true_lit = solver.new_var();
    solver.add_clause(None, vec![true_lit]);
    let false_var = -true_lit;

    for (train_idx, train) in problem.trains.iter().enumerate() {
        for (visit_idx, visit) in train.visits.iter().enumerate() {
            let visit_id: VisitId = visits.push_and_get_key((train_idx, visit_idx));

            occupations.push(Occ {
                cost: vec![true_lit],
                // cost_tree: CostTree::new(),
                delays: vec![(true_lit, visit.earliest), (false_var, i32::MAX)],
                incumbent_idx: 0,
            });
            n_timepoints += 1;

            while resource_visits.len() <= visit.resource_id {
                resource_visits.push(Vec::new());
            }

            resource_visits[visit.resource_id].push(visit_id);
            touched_intervals.push(visit_id);
            new_time_points.push((visit_id, true.into(), visit.earliest));
        }
    }

    // The first iteration (0) does not need a solve call; we
    // know it's SAT because there are no constraints yet.
    
    let mut iteration = 1;
    let mut total_cost = 0;
    loop {
        if start_time.elapsed().as_secs_f64() > timeout {
            return Err(SolverError::Timeout);
        }

        let mut found_travel_time_conflict = false;
        let mut found_resource_conflict = false;

        // let mut touched_intervals = visits.keys().collect::<Vec<_>>();

        for visit_id in touched_intervals.iter().copied() {
            let (train_idx, visit_idx) = visits[visit_id];
            let next_visit: Option<VisitId> =
                if visit_idx + 1 < problem.trains[train_idx].visits.len() {
                    Some((usize::from(visit_id) + 1).into())
                } else {
                    None
                };

            // GATHER INFORMATION ABOUT TWO CONSECUTIVE TIME POINTS
            let t1_in = occupations[visit_id].incumbent_time();
            let visit = problem.trains[train_idx].visits[visit_idx];

            if let Some(next_visit) = next_visit {
                let v1 = &occupations[visit_id];
                let v2 = &occupations[next_visit];
                let t1_out = v2.incumbent_time();

                // TRAVEL TIME CONFLICT
                if t1_in + visit.travel_time > t1_out {
                    found_travel_time_conflict = true;

                    // Insert the new time point.
                    let t1_in_var = v1.delays[v1.incumbent_idx].0;
                    let new_t = v1.incumbent_time() + visit.travel_time;
                    let (t1_earliest_out_var, t1_is_new) =
                        occupations[next_visit].time_point(&mut solver, new_t);

                    // T1_IN delay implies T1_EARLIEST_OUT delay.

                    solver.add_clause(None, vec![-t1_in_var, t1_earliest_out_var]);

                    stats.n_travel += 1;
                    // The new timepoint might have a cost.
                    if t1_is_new {
                        new_time_points.push((next_visit, t1_earliest_out_var, new_t));
                    }
                }
            }
        }

        // SOLVE ALL SIMPLE PRESEDENCES BEFORE CONFLICTS

        touched_intervals.retain(|visit_id| {
            let visit_id = *visit_id;

            let (train_idx, visit_idx) = visits[visit_id];
            let next_visit: Option<VisitId> =
                if visit_idx + 1 < problem.trains[train_idx].visits.len() {
                    Some((usize::from(visit_id) + 1).into())
                } else {
                    None
                };

            // GATHER INFORMATION ABOUT TWO CONSECUTIVE TIME POINTS
            let t1_in = occupations[visit_id].incumbent_time();
            let visit = problem.trains[train_idx].visits[visit_idx];

            // RESOURCE CONFLICT
            let mut retain = false;

            if let Some(conflicting_resources) = conflicts.get(&visit.resource_id) {
                for other_resource in conflicting_resources.iter().copied() {
                    let t1_out = next_visit
                        .map(|nx| occupations[nx].incumbent_time())
                        .unwrap_or(t1_in + visit.travel_time);

                    // Waiting in stations, but not in tracks (where conflicts occur).
                    // assert!(t1_in + travel_time == t1_out);

                    for other_visit in resource_visits[other_resource].iter().copied() {
                        if usize::from(visit_id) == usize::from(other_visit) {
                            continue;
                        }

                        let _v1 = &occupations[visit_id];
                        let v2 = &occupations[other_visit];
                        let t2_in = v2.incumbent_time();
                        let (other_train_idx, other_visit_idx) = visits[other_visit];

                        // We have a train2 that is conflicting.
                        if other_train_idx == train_idx {
                            continue; // Assume for now that the train doesn't conflict with itself.
                        }

                        let other_next_visit: Option<VisitId> =
                            if other_visit_idx + 1 < problem.trains[other_train_idx].visits.len() {
                                Some((usize::from(other_visit) + 1).into())
                            } else {
                                None
                            };

                        let t2_out = other_next_visit
                            .map(|v| occupations[v].incumbent_time())
                            .unwrap_or_else(|| {
                                let other_visit =
                                    problem.trains[other_train_idx].visits[other_visit_idx];
                                t2_in + other_visit.travel_time
                            });

                        // let t2_earliest_out = t2_in
                        //     + problem.trains[other_train_idx].visits[other_visit_idx].2;
                        // let t1_earliest_out =
                        //     t1_in + problem.trains[train_idx].visits[visit_idx].2;

                        // They are not overlapping so not in conflict.
                        if t1_out <= t2_in || t2_out <= t1_in {
                            continue;
                        }

                        found_resource_conflict = true;
                        stats.n_conflict += 1;

                        // We should not need to use these.
                        #[allow(unused, clippy::let_unit_value)]
                        let t1_in = ();
                        #[allow(unused, clippy::let_unit_value)]
                        let t2_in = ();

                        // The constraint is:
                        // We can delay T1_IN until T2_OUT?
                        // .. OR we can delay T2_IN until T1_OUT
                        let (delay_t2, t2_is_new) =
                            occupations[other_visit].time_point(&mut solver, t1_out);
                        let (delay_t1, t1_is_new) =
                            occupations[visit_id].time_point(&mut solver, t2_out);

                        if t1_is_new {
                            new_time_points.push((visit_id, delay_t1, t2_out));
                        }

                        if t2_is_new {
                            new_time_points.push((other_visit, delay_t2, t1_out));
                        }

                        let v1 = &occupations[visit_id];
                        let v2 = &occupations[other_visit];

                        let _t1_in_lit = v1.delays[v1.incumbent_idx].0;
                        let t1_out_lit = next_visit
                            .map(|v| occupations[v].delays[occupations[v].incumbent_idx].0)
                            .unwrap_or_else(|| true_lit);
                        let _t2_in_lit = v2.delays[v2.incumbent_idx].0;
                        let t2_out_lit = other_next_visit
                            .map(|v| occupations[v].delays[occupations[v].incumbent_idx].0)
                            .unwrap_or_else(|| true_lit);

                        n_conflict_constraints += 1;
                        solver.add_clause(
                            None,
                            vec![
                                // !t1_in_lit,
                                -t1_out_lit,
                                // !t2_in_lit,
                                -t2_out_lit,
                                delay_t1,
                                delay_t2,
                            ],
                        );
                    }
                }
            }

            retain
        });

        let iterationtype = if found_travel_time_conflict && found_resource_conflict {
            IterationType::TravelAndResourceConflict
        } else if found_travel_time_conflict {
            IterationType::TravelTimeConflict
        } else if found_resource_conflict {
            IterationType::ResourceConflict
        } else {
            IterationType::Solution
        };

        *iteration_types.entry(iterationtype).or_default() += 1;

        if !(found_resource_conflict || found_travel_time_conflict) {
            // Incumbent times are feasible and optimal.

            let trains = extract_solution(problem, &occupations);

            debug!(
                "Finished with cost {} iterations {} solver {:?}",
                total_cost, iteration, solver
            );

            stats.satsolver = format!("{:?}", solver);
            return Ok((trains, stats));
        }

        assert!(found_resource_conflict || found_travel_time_conflict);

        for (visit, new_timepoint_var, new_t) in new_time_points.drain(..) {
            n_timepoints += 1;
            let (train_idx, visit_idx) = visits[visit];
            // let resource = problem.trains[train_idx].visits[visit_idx].resource_id;

            let new_timepoint_cost =
                problem.trains[train_idx].visit_delay_cost(delay_cost_type, visit_idx, new_t);

            if new_timepoint_cost > 0 {

                    for cost in occupations[visit].cost.len()..=new_timepoint_cost {
                        let prev_cost_var = occupations[visit].cost[cost - 1];
                        let next_cost_var = solver.new_var();

                        solver.add_clause(None, vec![-next_cost_var, prev_cost_var]);

                        occupations[visit].cost.push(next_cost_var);
                        assert!(cost + 1 == occupations[visit].cost.len());

                        // soft_constraints.insert(!next_cost_var, (Soft::Delay, 1, 1));
                        solver.add_clause(Some(1), vec![-next_cost_var]);
                    }

                    solver.add_clause(
                        None,
                        vec![
                            -new_timepoint_var,
                            occupations[visit].cost[new_timepoint_cost],
                        ],
                    );
            }
        }

        debug!(
            "solving it{} with {} timepoints {} conflicts",
            iteration,
            n_timepoints,
            n_conflict_constraints
        );

        let solve_start = Instant::now();
        let result: Result<(i32, Vec<bool>), MaxSatError> = {
            solver.optimize(
                Some(timeout - solve_start.elapsed().as_secs_f64()),
                std::iter::empty(),
            )
        };
        solver_time += solve_start.elapsed();

        let (new_cost, sol) = result.map_err(|e| match e {
            MaxSatError::NoSolution => SolverError::NoSolution,
            MaxSatError::Timeout => SolverError::Timeout,
        })?;

        total_cost = new_cost;

        let model_value = |l: isize| {
            if l > 0 {
                sol[l as usize - 1]
            } else {
                !(sol[(-l) as usize - 1])
            }
        };

        stats.n_sat += 1;

        for (visit, this_occ) in occupations.iter_mut_enumerated() {
            // let old_time = this_occ.incumbent_time();
            let mut touched = false;

            while model_value(this_occ.delays[this_occ.incumbent_idx + 1].0) {
                this_occ.incumbent_idx += 1;
                touched = true;
            }
            while !model_value(this_occ.delays[this_occ.incumbent_idx].0) {
                this_occ.incumbent_idx -= 1;
                touched = true;
            }
            let (_train_idx, visit_idx) = visits[visit];

            if touched {
                // We are really interested not in the visits, but the resource occupation
                // intervals. Therefore, also the previous visit has been touched by this visit.
                if visit_idx > 0 {
                    let prev_visit = (Into::<usize>::into(visit) - 1).into();
                    if touched_intervals.last() != Some(&prev_visit) {
                        touched_intervals.push(prev_visit);
                    }
                }
                touched_intervals.push(visit);
            }
        }

        const USE_LOCAL_MINIMIZE: bool = true;
        if USE_LOCAL_MINIMIZE {
            let mut last_mod = 0;
            let mut i = 0;
            let occs_len = occupations.len();
            assert!(visits.len() == occupations.len());
            while last_mod < occs_len {
                let mut touched = false;
                let visit_id = VisitId(i % occs_len as u32);
                while occupations[visit_id].incumbent_idx > 0 {
                    // We can always leave earlier, so the critical interval is
                    // from this event to the next.

                    let t1_in = occupations[visit_id].delays[occupations[visit_id].incumbent_idx].1;
                    let t1_in_new =
                        occupations[visit_id].delays[occupations[visit_id].incumbent_idx - 1].1;

                    let (train_idx, visit_idx) = visits[visit_id];
                    let visit = problem.trains[train_idx].visits[visit_idx];

                    let next_visit: Option<VisitId> =
                        if visit_idx + 1 < problem.trains[train_idx].visits.len() {
                            Some((usize::from(visit_id) + 1).into())
                        } else {
                            None
                        };

                    let prev_visit: Option<VisitId> = if visit_idx > 0 {
                        Some((usize::from(visit_id) - 1).into())
                    } else {
                        None
                    };

                    let t1_prev_earliest_out = prev_visit
                        .map(|v| {
                            let (tidx, vidx) = visits[v];
                            let travel_time = problem.trains[tidx].visits[vidx].travel_time;
                            occupations[v].incumbent_time() + travel_time
                        })
                        .unwrap_or(i32::MIN);

                    let travel_ok = t1_prev_earliest_out <= t1_in_new;

                    let t1_out = next_visit
                        .map(|nx| occupations[nx].incumbent_time())
                        .unwrap_or(t1_in + visit.travel_time);

                    let can_reduce = travel_ok
                        && conflicts
                            .get(&visit.resource_id)
                            .iter()
                            .flat_map(|rs| rs.iter())
                            .copied()
                            .all(|other_resource| {
                                resource_visits[other_resource]
                                    .iter()
                                    .copied()
                                    .filter(|other_visit| {
                                        usize::from(visit_id) != usize::from(*other_visit)
                                    })
                                    .filter(|other_visit| visits[*other_visit].0 != train_idx)
                                    .all(|other_visit| {
                                        let v2 = &occupations[other_visit];
                                        let t2_in = v2.incumbent_time();
                                        let (other_train_idx, other_visit_idx) =
                                            visits[other_visit];
                                        let other_next_visit: Option<VisitId> = if other_visit_idx
                                            + 1
                                            < problem.trains[other_train_idx].visits.len()
                                        {
                                            Some((usize::from(other_visit) + 1).into())
                                        } else {
                                            None
                                        };

                                        let t2_out = other_next_visit
                                            .map(|v| occupations[v].incumbent_time())
                                            .unwrap_or_else(|| {
                                                let other_visit = problem.trains[other_train_idx]
                                                    .visits[other_visit_idx];
                                                t2_in + other_visit.travel_time
                                            });
                                        t1_out <= t2_in || t2_out <= t1_in_new
                                    })
                            });

                    if can_reduce {
                        occupations[visit_id].incumbent_idx -= 1;
                        touched = true;
                        last_mod = 0;
                    } else {
                        break;
                    }
                }

                i += 1;

                if touched {
                    let visit_idx = visits[visit_id].1;
                    if visit_idx > 0 {
                        let prev_visit = (Into::<usize>::into(visit_id) - 1).into();
                        if touched_intervals.last() != Some(&prev_visit) {
                            touched_intervals.push(prev_visit);
                        }
                    }
                    touched_intervals.push(visit_id);
                } else {
                    last_mod += 1;
                }
            }
        }

        iteration += 1;
    }
}

fn extract_solution(problem: &Problem, occupations: &TiVec<VisitId, Occ>) -> Vec<Vec<i32>> {
    let mut trains = Vec::new();
    let mut i = 0;
    for (train_idx, train) in problem.trains.iter().enumerate() {
        let mut train_times = Vec::new();
        for _ in train.visits.iter().enumerate() {
            train_times.push(occupations[VisitId(i)].incumbent_time());
            i += 1;
        }

        let visit = problem.trains[train_idx].visits[train_times.len() - 1];
        let last_t = train_times[train_times.len() - 1] + visit.travel_time;
        train_times.push(last_t);

        trains.push(train_times);
    }
    trains
}

#[derive(Debug)]
struct Occ {
    cost: Vec<isize>,
    // cost_tree: CostTree<L>,
    delays: Vec<(isize, i32)>,
    incumbent_idx: usize,
}

impl Occ {
    pub fn incumbent_time(&self) -> i32 {
        self.delays[self.incumbent_idx].1
    }

    pub fn time_point(&mut self, solver: &mut impl MaxSatSolver, t: i32) -> (isize, bool) {
        // The inserted time should be between the neighboring times.
        // assert!(idx == 0 || self.delays[idx - 1].1 < t);
        // assert!(idx == self.delays.len() || self.delays[idx + 1].1 > t);

        let idx = self.delays.partition_point(|(_, t0)| *t0 < t);

        assert!(idx > 0 || t == self.delays[0].1); // cannot insert before the earliest time.
        assert!(idx < self.delays.len()); // cannot insert after infinity.

        assert!(idx == 0 || self.delays[idx - 1].1 < t);
        assert!(self.delays[idx].1 >= t);

        if self.delays[idx].1 == t || (idx > 0 && self.delays[idx - 1].1 == t) {
            return (self.delays[idx].0, false);
        }

        let var = solver.new_var();
        self.delays.insert(idx, (var, t));

        if idx > 0 {
            solver.add_clause(None, vec![-var, self.delays[idx - 1].0]);
        }

        if idx < self.delays.len() {
            solver.add_clause(None, vec![-self.delays[idx + 1].0, var]);
        }

        (var, true)
    }
}
