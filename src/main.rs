use std::{collections::HashSet, fmt::Write};

use ddd::{
    parser,
    problem::{self, Visit},
    solver,
};

fn main() {
    let a_instances = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    let b_instances = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20];
    #[allow(unused)]
    let c_instances = [21, 22, 23, 24];

    let mut perf_out = String::new();

    for instance_id in a_instances
        .into_iter()
        .chain(b_instances)
        .chain(c_instances)
    {
        let filename = format!("instances/Instance{}.xml", instance_id);
        println!("Reading {}", filename);
        #[allow(unused)]
        let (problem, train_names, res_names) =
            parser::read_file(&filename, problem::DelayMeasurementType::FinalStationArrival);
        let problemstats = print_problem_stats(&problem);

        let ref_solution =
            std::fs::read_to_string(&format!("solutions/Instance{}.txt", instance_id))
                .ok()
                .map(|txt| parser::parse_solution_txt(&txt).unwrap());

        let refsolscore = ref_solution.as_ref().map(|s| {
            s.iter()
            .map(|t| {
                let cost = problem::delay_cost(t.final_delay);
                let last_visit = t.visits.last().unwrap();
                println!("ref train {} final time scheduled {} aimed departure {} delay {} final delay {} cost {}", t.name, 
                last_visit.final_time_scheduled, last_visit.aimed_departure_time, last_visit.delay, t.final_delay, cost);
                assert!(last_visit.final_time_scheduled - last_visit.aimed_departure_time == t.final_delay);

                cost
            })
            .sum::<usize>()
        });

        if let Some(ref_solution) = ref_solution.as_ref() {
            let problemtrains = train_names.iter().collect::<HashSet<_>>();
            let solutiontrains = ref_solution.iter().map(|t| &t.name).collect::<HashSet<_>>();
            let diff1 = problemtrains
                .difference(&solutiontrains)
                .collect::<Vec<_>>();
            let diff2 = solutiontrains
                .difference(&problemtrains)
                .collect::<Vec<_>>();
            println!("Trains not in solution: {:?}", diff1);
            println!("Trains not in problem: {:?}", diff2);
            // assert!(diff1.is_empty() && diff2.is_empty());
        }

        println!("Solving.");
        // let problem = problem::problem1();
        hprof::start_frame();
        let (solution, solvestats) =
            solver::solve(satcoder::solvers::minisat::Solver::new(), &problem).unwrap();
        hprof::end_frame();

        // for train_idx in 0..problem.trains.len() {
        //     let after_earliest = solution[train_idx].last().unwrap() - problem.trains[train_idx].visits.last().unwrap().earliest;
        //     println!("Train {:?}", train_names[train_idx]);
        //     for visit in problem.trains[train_idx].visits.iter() {
        //         println!("  visit {} {:?}", res_names[visit.resource_id], visit);
        //     }
        //     println!(
        //         "sol train {} delay {} cost {} afterearliest {}",
        //         train_names[train_idx],
        //         solution[train_idx].last().unwrap(),
        //         problem.train_cost(&solution, train_idx),
        //         after_earliest
        //     );
        // }

        let cost = problem.verify_solution(&solution);

        println!(
            "Verifying solution {:?} trains {} (ref.sol. cost {:?} trains {:?})",
            cost,
            problem.trains.len(),
            refsolscore,
            ref_solution.map(|s| s.len())
        );
        let cost = cost.unwrap();
        // assert!(cost == refsolscore as i32);

        hprof::profiler().print_timing();

        let _root = hprof::profiler().root();
        // println!("NODE {:?} {}", root.name, root.total_time.get());
        let sol_time = hprof::profiler().root().total_time.get() as f64 / 1_000_000f64;

        // table columns
        //  1. intstance name
        //  2. trains
        //  3. tracks/resources
        //  4. avg tracks
        //  5. conflicting track pairs
        //  6. cost
        //  8. sat iterations
        //  8. unsat iterations
        //  7. solution time in ms
        //  9.

        let vars = {
            let varstring = "variables: ";
            let vars_start = solvestats.satsolver.find(varstring).unwrap();
            let vars_start = &solvestats.satsolver[vars_start + varstring.as_bytes().len()..];
            let vars_end = vars_start.find(',').unwrap();
            &vars_start[..vars_end]
        };

        let clausestring = "clauses: ";
        let clauses_start = solvestats.satsolver.find(clausestring).unwrap();
        let clauses_start = &solvestats.satsolver[clauses_start + clausestring.as_bytes().len()..];
        let clauses_end = clauses_start.find(' ').unwrap();
        let clauses = &clauses_start[..clauses_end];

        writeln!(
            perf_out,
            "{}\t & {}\t & {:?}\t & {}",
            instance_id, cost, refsolscore, sol_time,
        )
        .unwrap();

        // writeln!(
        //     perf_out,
        //     "{}\t& {}\t& {}\t& {:.2}\t& {}\t& {}\t& {}\t& {}\t& {}\t& {}\t& {}\t& {}\t& {:.2}",
        //     instance_id,
        //     problemstats.trains,
        //     problemstats.conflicts,
        //     problemstats.avg_tracks,
        //     problemstats.conflicting_visit_pairs,
        //     cost,
        //     solvestats.n_sat,
        //     solvestats.n_unsat,
        //     solvestats.n_travel,
        //     solvestats.n_conflict,
        //     vars,
        //     clauses,
        //     sol_time,
        // )
        // .unwrap();
    }
    println!("{}", perf_out);
}

struct ProblemStats {
    trains: usize,
    conflicts: usize,
    avg_tracks: f32,
    conflicting_visit_pairs: usize,
}

fn print_problem_stats(problem: &problem::Problem) -> ProblemStats {
    let avg_tracks = problem
        .trains
        .iter()
        .map(|t| {
            t.visits
                .iter()
                .filter(|v| problem.conflicts.contains(&(v.resource_id, v.resource_id)))
                .count()
        })
        .sum::<usize>() as f32
        / problem.trains.len() as f32;
    let mut conflicting_visit_pairs = 0;
    for t1 in 0..problem.trains.len() {
        for t2 in (t1 + 1)..problem.trains.len() {
            for Visit {
                resource_id: r1, ..
            } in problem.trains[t1].visits.iter()
            {
                for Visit {
                    resource_id: r2, ..
                } in problem.trains[t2].visits.iter()
                {
                    if problem.conflicts.contains(&(*r1, *r2)) {
                        conflicting_visit_pairs += 1;
                    }
                }
            }
        }
    }

    let delays = 0;
    let avgdelay = 0;

    let trains = problem.trains.len();
    let conflicts = problem.conflicts.len();
    println!(
        "trains {} tracks {} avgtracks {:.2} trackpairs {} delays {} avgdelay{}",
        trains, conflicts, avg_tracks, conflicting_visit_pairs, delays, avgdelay,
    );
    ProblemStats {
        trains,
        conflicts,
        avg_tracks,
        conflicting_visit_pairs,
    }
}

#[cfg(test)]
mod tests {
    #[test]
    pub fn testproblem() {
        let problem = crate::problem::problem1_with_stations();
        let result = crate::solver::solve(satcoder::solvers::minisat::Solver::new(), &problem)
            .unwrap()
            .0;
        let score = problem.verify_solution(&result);
        assert!(score.is_some());
    }

    #[test]
    pub fn samescore_trivial() {
        let problem = crate::problem::problem1_with_stations();

        let result = crate::solver::solve(satcoder::solvers::minisat::Solver::new(), &problem)
            .unwrap()
            .0;
        let first_score = problem.verify_solution(&result);

        for _ in 0..100 {
            let result = crate::solver::solve(satcoder::solvers::minisat::Solver::new(), &problem)
                .unwrap()
                .0;
            let score = problem.verify_solution(&result);
            assert!(score == first_score);
        }
        println!("ALL COSTS WERE {:?}", first_score);
    }

    #[test]
    pub fn samescore_all_instances() {
        for delaytype in [
            ddd::problem::DelayMeasurementType::AllStationArrivals,
            ddd::problem::DelayMeasurementType::FinalStationArrival,
        ] {
            for instance_number in [
                1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
            ] {
                println!("{}", instance_number);
                let (problem, _train_names, _res_names) = crate::parser::read_file(
                    &format!("instances/Instance{}.xml", instance_number,),
                    delaytype,
                );

                let result =
                    crate::solver::solve(satcoder::solvers::minisat::Solver::new(), &problem)
                        .unwrap()
                        .0;
                let first_score = problem.verify_solution(&result);

                for iteration in 0..100 {
                    println!("iteration {} {}", instance_number, iteration);
                    let result =
                        crate::solver::solve(satcoder::solvers::minisat::Solver::new(), &problem)
                            .unwrap()
                            .0;
                    let score = problem.verify_solution(&result);
                    assert!(score == first_score);
                }

                println!("ALL COSTS WERE {:?}", first_score);
            }
        }
    }
}
