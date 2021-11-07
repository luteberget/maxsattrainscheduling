use std::collections::HashMap;

use crate::problem::Problem;
use satcoder::{
    constraints::Totalizer,
    prelude::SymbolicModel,
    solvers::minisat::{self, Bool, Lit},
    SatInstance, SatSolverWithCore,
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

pub fn solve(problem: &Problem) -> Result<Vec<Vec<i32>>, ()> {
    // TODO
    //  - optimization: move weight over to higher bound like original RC2
    //  - more eager constraint generation
    //    - propagate simple presedences?
    //    - update all conflicts and presedences when adding new time points?
    //    - smt refinement of the simple presedences?
    //  - get rid of the multiple adding of constraints

    let mut solver = minisat::Solver::new();
    let mut visits: TiVec<VisitId, (usize, usize)> = TiVec::new();
    let mut resource_visits: Vec<Vec<VisitId>> = Vec::new();
    let mut occupations: TiVec<VisitId, Occ> = TiVec::new();
    let mut touched_intervals = Vec::new();
    let mut conflicts: HashMap<_, Vec<_>> = HashMap::new();
    let mut new_time_points = Vec::new();

    for (a, b) in problem.conflicts.iter() {
        conflicts.entry(*a).or_default().push(*b);
        conflicts.entry(*b).or_default().push(*a);
    }

    for (train_idx, train) in problem.trains.iter().enumerate() {
        for (visit_idx, (resource, earliest_t, _travel_time)) in train.visits.iter().enumerate() {
            let visit: VisitId = visits.push_and_get_key((train_idx, visit_idx));

            occupations.push(Occ {
                cost: vec![true.into()],
                delays: vec![(true.into(), *earliest_t), (false.into(), i32::MAX)],
                incumbent: 0,
            });

            while resource_visits.len() <= *resource {
                resource_visits.push(Vec::new());
            }

            resource_visits[*resource].push(visit);
            touched_intervals.push(visit);
            new_time_points.push((visit, true.into(), *earliest_t));
        }
    }

    let mut iteration = 0;
    let mut total_cost = 0;
    let mut soft_constraints = HashMap::new();
    let mut is_sat = true;

    loop {
        if is_sat {
            println!("Iteration {} conflict detection starting...", iteration);

            let mut found_conflict = false;
            for visit in touched_intervals.drain(..) {
                let (train_idx, visit_idx) = visits[visit];
                let next_visit: Option<VisitId> = if visit_idx + 1 < problem.trains[train_idx].visits.len() {
                    Some((usize::from(visit) + 1).into())
                } else {
                    None
                };

                // GATHER INFORMATION ABOUT TWO CONSECUTIVE TIME POINTS
                let t1_in = occupations[visit].incumbent_time();
                let (this_resource_id, _earliest, travel_time) = problem.trains[train_idx].visits[visit_idx];

                if let Some(next_visit) = next_visit {
                    let v1 = &occupations[visit];
                    let v2 = &occupations[next_visit];
                    let t1_out = v2.incumbent_time();

                    // TRAVEL TIME CONFLICT
                    if t1_in + travel_time > t1_out {
                        found_conflict = true;
                        println!(
                            "  - TRAVEL time conflict train{} visit{} resource{} in{} travel{} out{}",
                            train_idx, visit_idx, this_resource_id, t1_in, travel_time, t1_out
                        );

                        // Insert the new time point.
                        let t1_in_var = v1.delays[v1.incumbent].0;
                        let new_t = v1.incumbent_time() + travel_time;
                        let (t1_earliest_out_var, t1_is_new) = occupations[next_visit].time_point(&mut solver, new_t);

                        // T1_IN delay implies T1_EARLIEST_OUT delay.
                        SatInstance::add_clause(&mut solver, vec![!t1_in_var, t1_earliest_out_var]);

                        // The new timepoint might have a cost.
                        if t1_is_new {
                            new_time_points.push((next_visit, t1_in_var, new_t));
                        }
                    }
                }
                // RESOURCE CONFLICT

                if let Some(conflicting_resources) = conflicts.get(&this_resource_id) {
                    for other_resource in conflicting_resources.iter().copied() {
                        let t1_out = next_visit
                            .map(|v| occupations[v].incumbent_time())
                            .unwrap_or(t1_in + travel_time);

                        for other_visit in resource_visits[other_resource].iter().copied() {
                            if usize::from(visit) == usize::from(other_visit) {
                                continue;
                            }
                            let _v1 = &occupations[visit];
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
                                    let (_other_resource_id, _e, travel_time) =
                                        problem.trains[other_train_idx].visits[other_visit_idx];
                                    t2_in + travel_time
                                });

                            // They are not overlapping so not in conflict.
                            if t1_out <= t2_in || t2_out <= t1_in {
                                continue;
                            }

                            found_conflict = true;

                            println!(
                                " - RESOURCE conflict between t{}-v{}-r{}-in{}-out{} t{}-v{}-r{}-in{}-out{}",
                                train_idx,
                                visit_idx,
                                problem.trains[train_idx].visits[visit_idx].1,
                                t1_in,
                                t1_out,
                                other_train_idx,
                                other_visit_idx,
                                problem.trains[other_train_idx].visits[other_visit_idx].1,
                                t2_in,
                                t2_out,
                            );

                            // We should not need to use these.
                            #[allow(unused)]
                            let t1_in = ();
                            #[allow(unused)]
                            let t2_in = ();

                            // The constraint is:
                            // We can delay T1_IN until T2_OUT?
                            // .. OR we can delay T2_IN until T1_OUT
                            let (delay_t2, t2_is_new) = occupations[other_visit].time_point(&mut solver, t1_out);
                            let (delay_t1, t1_is_new) = occupations[visit].time_point(&mut solver, t2_out);

                            if !t2_is_new && !t1_is_new {
                                println!("Did we solve this before?");
                            }

                            if t1_is_new {
                                new_time_points.push((visit, delay_t1, t2_out));
                            }

                            if t2_is_new {
                                new_time_points.push((other_visit, delay_t2, t1_out));
                            }

                            let v1 = &occupations[visit];
                            let v2 = &occupations[other_visit];

                            const USE_CHOICE_VAR: bool = false;

                            if USE_CHOICE_VAR {
                                let choose = SatInstance::new_var(&mut solver);
                                SatInstance::add_clause(
                                    &mut solver,
                                    vec![!choose, !v1.delays[v1.incumbent].0, delay_t2],
                                );
                                SatInstance::add_clause(
                                    &mut solver,
                                    vec![choose, !v2.delays[v2.incumbent].0, delay_t1],
                                );
                            } else {
                                SatInstance::add_clause(
                                    &mut solver,
                                    vec![
                                        !v1.delays[v1.incumbent].0,
                                        !v2.delays[v2.incumbent].0,
                                        delay_t1,
                                        delay_t2,
                                    ],
                                );
                            }
                        }
                    }
                }
            }

            if !found_conflict {
                // Incumbent times are feasible and optimal.

                let mut trains = Vec::new();
                let mut i = 0;
                for (train_idx, train) in problem.trains.iter().enumerate() {
                    let mut train_times = Vec::new();
                    for _ in train.visits.iter().enumerate() {
                        train_times.push(occupations[VisitId(i)].incumbent_time());
                        i += 1;
                    }

                    let (_last_resource, _, travel_time) = problem.trains[train_idx].visits[train_times.len() - 1];
                    let last_t = train_times[train_times.len() - 1] + travel_time;
                    train_times.push(last_t);

                    trains.push(train_times);
                }

                println!("Finished with cost {} iterations {} solver {:?}", total_cost, iteration, solver);
                return Ok(trains);
            }
        }
        enum Soft {
            Delay,
            Totalizer(Totalizer<Lit>, usize),
        }

        for (visit, new_var, new_t) in new_time_points.drain(..) {
            
            let (train_idx, visit_idx) = visits[visit];
            let resource = problem.trains[train_idx].visits[visit_idx].1;
            
            let new_var_cost = problem.trains[train_idx].delay_cost(visit_idx, new_t);
            println!("new var for t{} v{} t{} cost{}", train_idx, visit_idx, new_t, new_var_cost);
            for cost in occupations[visit].cost.len()..=new_var_cost {
                println!("Extending t{}v{} to cost {}", train_idx, visit_idx, cost);
                let prev_cost_var = occupations[visit].cost[cost - 1];
                let next_cost_var = SatInstance::new_var(&mut solver);

                SatInstance::add_clause(&mut solver, vec![!next_cost_var, prev_cost_var]);
                occupations[visit].cost.push(next_cost_var);

                println!(
                    "Soft constraint for t{}-v{}-r{} t{} cost{} lit{:?}",
                    train_idx, visit_idx, resource, new_t, new_var_cost, new_var
                );
                soft_constraints.insert(!next_cost_var, (Soft::Delay, 1, 1));
            }

            // set the cost for this new time point.
            println!("   new var implies cost {}=>{:?}", new_var_cost, occupations[visit].cost[new_var_cost]);
            SatInstance::add_clause(&mut solver, vec![!new_var, occupations[visit].cost[new_var_cost]]);
        }

        let core = {
            let assumptions = soft_constraints.keys().copied();
            let result = SatSolverWithCore::solve_with_assumptions(&mut solver, assumptions);
            match result {
                satcoder::SatResultWithCore::Sat(model) => {
                    is_sat = true;
                    for (visit, this_occ) in occupations.iter_mut_enumerated() {
                        let old_time = this_occ.incumbent_time();
                        let mut touched = false;

                        while model.value(&this_occ.delays[this_occ.incumbent + 1].0) {
                            this_occ.incumbent += 1;
                            touched = true;
                        }
                        while !model.value(&this_occ.delays[this_occ.incumbent].0) {
                            this_occ.incumbent -= 1;
                            touched = true;
                        }

                        if touched {
                            let (train_idx, visit_idx) = visits[visit];
                            let resource = problem.trains[train_idx].visits[visit_idx].1;
                            let new_time = this_occ.incumbent_time();
                            println!(
                                "Updated t{}-v{}-r{}  t={}-->{}",
                                train_idx, visit_idx, resource, old_time, new_time
                            );

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

                    None
                }
                satcoder::SatResultWithCore::Unsat(core) => {
                    if core.is_empty() {
                        return Err(()); // UNSAT
                    }
                    Some(core)
                }
            }
        };

        if let Some(core) = core {
            is_sat = false;
            println!("Got core length {}", core.len());
            // Do weighted RC2

            if core.len() == 0 {
                return Err(()); // UNSAT
            }

            let min_weight = core.iter().map(|c| soft_constraints[&Bool::Lit(*c)].1).min().unwrap();
            assert!(min_weight == 1);

            println!("Core min weight {}", min_weight);

            for c in core.iter() {
                let (soft, cost, original_cost) = soft_constraints.remove(&Bool::Lit(*c)).unwrap();
                let soft_str = match &soft {
                    Soft::Delay => "delay".to_string(),
                    Soft::Totalizer(_, b) => format!("totalizer w/bound={}", b),
                };
                println!("{} {}", soft_str, cost);

                assert!(cost >= min_weight);
                let new_cost = cost - min_weight;
                assert!(new_cost == 0);

                match soft {
                    Soft::Delay => { /* primary soft constraint, when we relax we are done */ }
                    Soft::Totalizer(mut tot, bound) => {
                        // totalizer: need to extend its bound
                        let new_bound = bound + 1;
                        tot.increase_bound(&mut solver, new_bound as u32);
                        if new_bound < tot.rhs().len() {
                            soft_constraints.insert(
                                !tot.rhs()[new_bound],
                                (Soft::Totalizer(tot, new_bound), original_cost, original_cost),
                            );
                        }
                    }
                }
            }

            total_cost += 1;
            if core.len() > 1 {
                let bound = 1;
                let tot = Totalizer::count(&mut solver, core.iter().map(|c| Bool::Lit(!*c)), bound as u32);
                soft_constraints.insert(!tot.rhs()[bound], (Soft::Totalizer(tot, bound), min_weight, min_weight));
            } else {
                SatInstance::add_clause(&mut solver, vec![Bool::Lit(core[0])]);
            }
        }

        iteration += 1;
    }
}

#[derive(Debug)]
struct Occ {
    cost: Vec<Bool>,
    delays: Vec<(Bool, i32)>,
    incumbent: usize,
}

impl Occ {
    pub fn incumbent_time(&self) -> i32 {
        self.delays[self.incumbent].1
    }

    pub fn time_point(&mut self, solver: &mut impl SatInstance<minisat::Lit>, t: i32) -> (Bool, bool) {
        // The inserted time should be between the neighboring times.
        // assert!(idx == 0 || self.delays[idx - 1].1 < t);
        // assert!(idx == self.delays.len() || self.delays[idx + 1].1 > t);

        let idx = self.delays.partition_point(|(_, t0)| *t0 < t);

        println!("idx {} t {}   delays{:?}", idx, t, self.delays);

        assert!(idx > 0); // cannot insert before the earliest time.
        assert!(idx < self.delays.len()); // cannot insert after infinity.

        assert!(self.delays[idx - 1].1 < t);
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
