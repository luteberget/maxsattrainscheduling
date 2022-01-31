use std::{
    cell::RefCell,
    collections::{BTreeMap, HashMap},
};

#[allow(unused)]
use crate::{
    debug::{ResourceInterval, SolverAction},
    problem::Problem,
    minimize_core,
    trim_core,
};
use satcoder::{
    constraints::Totalizer, prelude::SymbolicModel, Bool, SatInstance, SatSolverWithCore,
};
use typed_index_collections::TiVec;

#[derive(Clone, Copy, PartialEq, Eq)]
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
}

#[derive(Default)]
pub struct SolveStats {
    pub n_sat: usize,
    pub n_unsat: usize,
    pub n_travel: usize,
    pub n_conflict: usize,
    pub satsolver: String,
}

pub fn solve<L: satcoder::Lit + Copy + std::fmt::Debug>(
    solver: impl SatInstance<L> + SatSolverWithCore<Lit = L> + std::fmt::Debug,
    problem: &Problem,
) -> Result<(Vec<Vec<i32>>, SolveStats), SolverError> {
    solve_debug(solver, problem, |_| {})
}

thread_local! { pub static  WATCH : std::cell::RefCell<Option<(usize,usize)>>  = RefCell::new(None);}

use crate::debug::DebugInfo;

use super::SolverError;
pub fn solve_debug<L: satcoder::Lit + Copy + std::fmt::Debug>(
    mut solver: impl SatInstance<L> + SatSolverWithCore<Lit = L> + std::fmt::Debug,
    problem: &Problem,
    debug_out: impl Fn(DebugInfo),
) -> Result<(Vec<Vec<i32>>, SolveStats), SolverError> {
    // TODO
    //  - more eager constraint generation
    //    - propagate simple presedences?
    //    - update all conflicts and presedences when adding new time points?
    //    - smt refinement of the simple presedences?
    //  - get rid of the multiple adding of constraints
    //  - cadical doesn't use false polarity, so it can generate unlimited conflicts when cost is maxed. Two trains pushing each other forward.

    let _p = hprof::enter("solver");

    let mut stats = SolveStats::default();

    let mut visits: TiVec<VisitId, (usize, usize)> = TiVec::new();
    let mut resource_visits: Vec<Vec<VisitId>> = Vec::new();
    let mut occupations: TiVec<VisitId, Occ<_>> = TiVec::new();
    let mut touched_intervals = Vec::new();
    let mut conflicts: HashMap<_, Vec<_>> = HashMap::new();
    let mut new_time_points = Vec::new();

    #[allow(unused)]
    let mut core_sizes: BTreeMap<usize, usize> = BTreeMap::new();
    #[allow(unused)]
    let mut processed_core_sizes: BTreeMap<usize, usize> = BTreeMap::new();
    let mut iteration_types: BTreeMap<IterationType, usize> = BTreeMap::new();

    for (a, b) in problem.conflicts.iter() {
        conflicts.entry(*a).or_default().push(*b);
        conflicts.entry(*b).or_default().push(*a);
    }

    for (train_idx, train) in problem.trains.iter().enumerate() {
        for (visit_idx, visit) in train.visits.iter().enumerate() {
            let visit_id: VisitId = visits.push_and_get_key((train_idx, visit_idx));

            occupations.push(Occ {
                cost: vec![true.into()],
                delays: vec![(true.into(), visit.earliest), (false.into(), i32::MAX)],
                incumbent: 0,
            });

            while resource_visits.len() <= visit.resource_id {
                resource_visits.push(Vec::new());
            }

            resource_visits[visit.resource_id].push(visit_id);
            touched_intervals.push(visit_id);
            new_time_points.push((visit_id, true.into(), visit.earliest));
        }
    }

    let mut iteration = 0;
    let mut total_cost = 0;
    let mut soft_constraints = HashMap::new();
    let mut is_sat = true;
    let mut debug_actions = Vec::new();

    loop {
        let _p = hprof::enter("iteration");
        if is_sat {
            // println!("Iteration {} conflict detection starting...", iteration);

            let mut found_conflict = false;

            // let mut touched_intervals = visits.keys().collect::<Vec<_>>();

            for visit_id in touched_intervals.iter().copied() {
                let _p = hprof::enter("travel time check");
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
                        found_conflict = true;
                        // println!(
                        //     "  - TRAVEL time conflict train{} visit{} resource{} in{} travel{} out{}",
                        //     train_idx, visit_idx, this_resource_id, t1_in, travel_time, t1_out
                        // );
                        *iteration_types
                            .entry(IterationType::TravelTimeConflict)
                            .or_default() += 1;

                        debug_actions.push(SolverAction::TravelTimeConflict(ResourceInterval {
                            train_idx,
                            visit_idx,
                            resource_idx: visit.resource_id,
                            time_in: t1_in,
                            time_out: t1_out,
                        }));

                        // Insert the new time point.
                        let t1_in_var = v1.delays[v1.incumbent].0;
                        let new_t = v1.incumbent_time() + visit.travel_time;
                        let (t1_earliest_out_var, t1_is_new) =
                            occupations[next_visit].time_point(&mut solver, new_t);

                        // T1_IN delay implies T1_EARLIEST_OUT delay.
                        SatInstance::add_clause(&mut solver, vec![!t1_in_var, t1_earliest_out_var]);
                        stats.n_travel += 1;
                        // The new timepoint might have a cost.
                        if t1_is_new {
                            new_time_points.push((next_visit, t1_earliest_out_var, new_t));
                        }
                    }

                    // let v1 = &occupations[visit];
                    // let v2 = &occupations[next_visit];

                    // // TRAVEL TIME CONFLICT
                    // if t1_in + travel_time < t1_out {
                    //     found_conflict = true;
                    //     println!(
                    //                             "  - TRAVEL OVERtime conflict train{} visit{} resource{} in{} travel{} out{}",
                    //                             train_idx, visit_idx, this_resource_id, t1_in, travel_time, t1_out
                    //                         );

                    //     // Insert the new time point.
                    //     let t1_in_var = v1.delays[v1.incumbent].0;
                    //     let new_t = v1.incumbent_time() + travel_time;
                    //     let (t1_earliest_out_var, t1_is_new) =
                    //         occupations[next_visit].time_point(&mut solver, new_t);

                    //     // T1_IN delay implies T1_EARLIEST_OUT delay.
                    //     SatInstance::add_clause(&mut solver, vec![!t1_in_var, t1_earliest_out_var]);
                    //     // The new timepoint might have a cost.
                    //     if t1_is_new {
                    //         new_time_points.push((next_visit, t1_in_var, new_t));
                    //     }
                    // }
                }
            }

            // SOLVE ALL SIMPLE PRESEDENCES BEFORE CONFLICTS
            // if !found_conflict {
            for visit_id in touched_intervals.iter().copied() {
                let _p = hprof::enter("conflict check");
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

                            let other_next_visit: Option<VisitId> = if other_visit_idx + 1
                                < problem.trains[other_train_idx].visits.len()
                            {
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

                            // let t2_out = t2_earliest_out;
                            // let t1_out = t1_earliest_out;
                            // if t1_out <= t2_in || t2_out <= t1_in {
                            //     panic!("kejks");
                            // }

                            found_conflict = true;
                            *iteration_types
                                .entry(IterationType::ResourceConflict)
                                .or_default() += 1;
                            stats.n_conflict += 1;
                            // println!(
                            //         " - RESOURCE conflict between t{}-v{}-r{}-in{}-out{} t{}-v{}-r{}-in{}-out{}",
                            //         train_idx,
                            //         visit_idx,
                            //         problem.trains[train_idx].visits[visit_idx].0,
                            //         t1_in,
                            //         t1_out,
                            //         other_train_idx,
                            //         other_visit_idx,
                            //         problem.trains[other_train_idx].visits[other_visit_idx].0,
                            //         t2_in,
                            //         t2_out,
                            //     );

                            // We should not need to use these.
                            #[allow(unused)]
                            let t1_in = ();
                            #[allow(unused)]
                            let t2_in = ();

                            // The constraint is:
                            // We can delay T1_IN until T2_OUT?
                            // .. OR we can delay T2_IN until T1_OUT
                            let (delay_t2, t2_is_new) =
                                occupations[other_visit].time_point(&mut solver, t1_out);
                            let (delay_t1, t1_is_new) =
                                occupations[visit_id].time_point(&mut solver, t2_out);

                            if !t2_is_new && !t1_is_new {
                                // println!("Did we solve this before?");
                            }

                            if t1_is_new {
                                new_time_points.push((visit_id, delay_t1, t2_out));
                            }

                            if t2_is_new {
                                new_time_points.push((other_visit, delay_t2, t1_out));
                            }

                            let v1 = &occupations[visit_id];
                            let v2 = &occupations[other_visit];

                            let _t1_in_lit = v1.delays[v1.incumbent].0;
                            let t1_out_lit = next_visit
                                .map(|v| occupations[v].delays[occupations[v].incumbent].0)
                                .unwrap_or_else(|| true.into());
                            let _t2_in_lit = v2.delays[v2.incumbent].0;
                            let t2_out_lit = other_next_visit
                                .map(|v| occupations[v].delays[occupations[v].incumbent].0)
                                .unwrap_or_else(|| true.into());

                            const USE_CHOICE_VAR: bool = true;

                            if USE_CHOICE_VAR {
                                let choose = SatInstance::new_var(&mut solver);
                                SatInstance::add_clause(
                                    &mut solver,
                                    vec![!choose, !t1_out_lit, delay_t2],
                                );
                                SatInstance::add_clause(
                                    &mut solver,
                                    vec![choose, !t2_out_lit, delay_t1],
                                );
                            } else {
                                SatInstance::add_clause(
                                    &mut solver,
                                    vec![
                                        // !t1_in_lit,
                                        !t1_out_lit,
                                        // !t2_in_lit,
                                        !t2_out_lit,
                                        delay_t1,
                                        delay_t2,
                                    ],
                                );
                            }
                        }
                    }
                }
            }

            touched_intervals.clear();
            assert!(touched_intervals.is_empty());
            // }

            if !found_conflict {
                // Incumbent times are feasible and optimal.

                let trains = extract_solution(problem, &occupations);

                println!(
                    "Finished with cost {} iterations {} solver {:?}",
                    total_cost, iteration, solver
                );
                println!("Core size bins {:?}", core_sizes);
                println!("Iteration types {:?}", iteration_types);
                debug_out(DebugInfo {
                    iteration,
                    actions: std::mem::take(&mut debug_actions),
                    solution: extract_solution(problem, &occupations),
                });

                stats.satsolver = format!("{:?}", solver);
                return Ok((trains, stats));
            }
        }
        enum Soft<L: satcoder::Lit> {
            Delay,
            Totalizer(Totalizer<L>, usize),
        }

        for (visit, new_var, new_t) in new_time_points.drain(..) {
            let (train_idx, visit_idx) = visits[visit];
            // let resource = problem.trains[train_idx].visits[visit_idx].resource_id;

            let new_var_cost = problem.trains[train_idx].visit_delay_cost(visit_idx, new_t);
            // println!(
            //     "new var for t{} v{} t{} cost{}",
            //     train_idx, visit_idx, new_t, new_var_cost
            // );
            for cost in occupations[visit].cost.len()..=new_var_cost {
                let prev_cost_var = occupations[visit].cost[cost - 1];
                let next_cost_var = SatInstance::new_var(&mut solver);

                SatInstance::add_clause(&mut solver, vec![!next_cost_var, prev_cost_var]);

                occupations[visit].cost.push(next_cost_var);
                assert!(cost + 1 == occupations[visit].cost.len());

                soft_constraints.insert(!next_cost_var, (Soft::Delay, 1, 1));
                // println!(
                //     "Extending t{}v{} to cost {} {:?}",
                //     train_idx, visit_idx, cost, next_cost_var
                // );
            }

            // set the cost for this new time point.

            // WATCH.with(|x| {
            //     if *x.borrow() == Some((train_idx, visit_idx)) {
            // println!(
            //     "Soft constraint for t{}-v{}-r{} t{} cost{} lit{:?}",
            //     train_idx, visit_idx, resource, new_t, new_var_cost, new_var
            // );
            // println!(
            //     "   new var implies cost {}=>{:?}",
            //     new_var_cost, occupations[visit].cost[new_var_cost]
            // );
            //     }
            // });
            // println!(
            //     "Soft constraint for t{}-v{}-r{} t{} cost{} lit{:?}",
            //     train_idx, visit_idx, resource, new_t, new_var_cost, new_var
            // );
            // println!(
            //     "   new var implies cost {}=>{:?}",
            //     new_var_cost, occupations[visit].cost[new_var_cost]
            // );
            SatInstance::add_clause(
                &mut solver,
                vec![!new_var, occupations[visit].cost[new_var_cost]],
            );
        }

        let core = {
            let result = {
                // println!("solving");
                let _p = hprof::enter("sat check");
                #[allow(clippy::needless_collect)]
                let assumptions = soft_constraints.keys().copied().collect::<Vec<_>>();
                SatSolverWithCore::solve_with_assumptions(&mut solver, assumptions.into_iter())
            };

            // println!("solving done");
            match result {
                satcoder::SatResultWithCore::Sat(model) => {
                    is_sat = true;
                    stats.n_sat += 1;
                    let _p = hprof::enter("update times");

                    for (visit, this_occ) in occupations.iter_mut_enumerated() {
                        // let old_time = this_occ.incumbent_time();
                        let mut touched = false;

                        while model.value(&this_occ.delays[this_occ.incumbent + 1].0) {
                            this_occ.incumbent += 1;
                            touched = true;
                        }
                        while !model.value(&this_occ.delays[this_occ.incumbent].0) {
                            this_occ.incumbent -= 1;
                            touched = true;
                        }
                        let (_train_idx, visit_idx) = visits[visit];

                        // let resource = problem.trains[train_idx].visits[visit_idx].resource_id;
                        // let new_time = this_occ.incumbent_time();

                        // WATCH.with(|x| {
                        //     if *x.borrow() == Some((train_idx, visit_idx))  && touched {
                        //         println!("Delays {:?}", this_occ.delays);
                        //         println!(
                        //             "Updated t{}-v{}-r{}  t={}-->{}",
                        //             train_idx, visit_idx, resource, old_time, new_time
                        //         );
                        //     }
                        // });

                        if touched {
                            // println!(
                            //     "Updated t{}-v{}-r{}  t={}-->{}",
                            //     train_idx, visit_idx, resource, old_time, new_time
                            // );

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

                        let cost = this_occ
                            .cost
                            .iter()
                            .map(|c| if model.value(c) { 1 } else { 0 })
                            .sum::<isize>()
                            - 1;
                        if cost > 0 {
                            // println!("t{}-v{}  cost={}", train_idx, visit_idx, cost);
                        }
                    }

                    // println!(
                    //     "Touched {}/{} occupations",
                    //     touched_intervals.len(),
                    //     occupations.len()
                    // );

                    debug_out(DebugInfo {
                        iteration,
                        actions: std::mem::take(&mut debug_actions),
                        solution: extract_solution(problem, &occupations),
                    });

                    None
                }
                satcoder::SatResultWithCore::Unsat(core) => {
                    is_sat = false;
                    stats.n_unsat += 1;
                    Some(core)
                }
            }
        };

        if let Some(core) = core {
            let _p = hprof::enter("treat core");
            // println!("Got core length {}", core.len());
            // Do weighted RC2

            if core.len() == 0 {
                return Err(SolverError::NoSolution); // UNSAT
            }

            let core = core.iter().map(|c| Bool::Lit(*c)).collect::<Vec<_>>();

            // println!("Core size {}", core.len());
            // *core_sizes.entry(core.len()).or_default() += 1;
            // trim_core(&mut core, &mut solver);
            // minimize_core(&mut core, &mut solver);

            // *processed_core_sizes.entry(core.len()).or_default() += 1;
            // println!("  pre sizes {:?}", core_sizes);
            // println!("  post sizes {:?}", processed_core_sizes);
            *iteration_types.entry(IterationType::Objective).or_default() += 1;
            debug_actions.push(SolverAction::Core(core.len()));

            let min_weight = core.iter().map(|c| soft_constraints[c].1).min().unwrap();
            assert!(min_weight == 1);

            // println!("Core min weight {}", min_weight);

            for c in core.iter() {
                let (soft, cost, original_cost) = soft_constraints.remove(c).unwrap();
                
                // let soft_str = match &soft {
                //     Soft::Delay => "delay".to_string(),
                //     Soft::Totalizer(_, b) => format!("totalizer w/bound={}", b),
                // };
                // println!("{} {}", soft_str, cost);

                assert!(cost >= min_weight);
                let new_cost = cost - min_weight;
                assert!(new_cost == 0);
                assert!(original_cost == 1);
                match soft {
                    Soft::Delay => { /* primary soft constraint, when we relax we are done */ }
                    Soft::Totalizer(mut tot, bound) => {
                        // totalizer: need to extend its bound
                        let new_bound = bound + 1;
                        tot.increase_bound(&mut solver, new_bound as u32);
                        if new_bound < tot.rhs().len() {
                            soft_constraints.insert(
                                !tot.rhs()[new_bound], // tot <= 2, 3, 4...
                                (
                                    Soft::Totalizer(tot, new_bound),
                                    original_cost,
                                    original_cost,
                                ),
                            );
                        }
                    }
                }
            }

            total_cost += 1;
            if core.len() > 1 {
                let bound = 1;
                let tot = Totalizer::count(&mut solver, core.iter().map(|c| !*c), bound as u32);
                assert!(bound < tot.rhs().len());
                soft_constraints.insert(
                    !tot.rhs()[bound], // tot <= 1
                    (Soft::Totalizer(tot, bound), min_weight, min_weight),
                );
            } else {
                // panic!();
                SatInstance::add_clause(&mut solver, vec![!core[0]]);
            }
        }

        iteration += 1;
    }
}

fn extract_solution<L: satcoder::Lit>(
    problem: &Problem,
    occupations: &TiVec<VisitId, Occ<L>>,
) -> Vec<Vec<i32>> {
    let _p = hprof::enter("extract solution");
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
struct Occ<L: satcoder::Lit> {
    cost: Vec<Bool<L>>,
    delays: Vec<(Bool<L>, i32)>,
    incumbent: usize,
}

impl<L: satcoder::Lit> Occ<L> {
    pub fn incumbent_time(&self) -> i32 {
        self.delays[self.incumbent].1
    }

    pub fn time_point(&mut self, solver: &mut impl SatInstance<L>, t: i32) -> (Bool<L>, bool) {
        // The inserted time should be between the neighboring times.
        // assert!(idx == 0 || self.delays[idx - 1].1 < t);
        // assert!(idx == self.delays.len() || self.delays[idx + 1].1 > t);

        let idx = self.delays.partition_point(|(_, t0)| *t0 < t);

        // println!("idx {} t {}   delays{:?}", idx, t, self.delays);

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
            solver.add_clause(vec![!var, self.delays[idx - 1].0]);
        }

        if idx < self.delays.len() {
            solver.add_clause(vec![!self.delays[idx + 1].0, var]);
        }

        (var, true)
    }
}
