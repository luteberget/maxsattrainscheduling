use ddd::*;
use satcoder::{SatInstance, SatSolverWithCore};

fn main() {
    let a_instances = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    let b_instances = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20];
    let c_instances = [21, 22, 23, 24];
    for instance_id in a_instances.into_iter().chain(b_instances).chain(c_instances) {
        hprof::start_frame();
        let filename = format!("instances/Instance{}.xml", instance_id);
        println!("Reading {}", filename);
        let (problem, train_names, res_names) = parser::read_file(&filename);
        print_problem_stats(&problem);
        println!("Solving.");
        // let problem = problem::problem1();
        let result = solver::solve(    satcoder::solvers::minisat::Solver::new()
        , &problem).unwrap();
        println!("Verifying solution {:?}", problem.verify_solution(&result));
        // println!("Result {:#?}", result);
        // println!("t0 {}", train_names[0]);
        // println!("t0v1 {}", res_names[problem.trains[0].visits[1].0]);
        // println!("t3 {}", train_names[3]);
        // println!("t3v0 {}", res_names[problem.trains[3].visits[0].0]);
        // println!("Problem: {:#?}", problem);
        // println!("Trains: {:?}", train_names);
        // println!("Resources: {:?}", res_names);
        // println!("Conflict resources: {}", problem.conflicts.len());

        hprof::profiler().print_timing();
        hprof::end_frame();
    }
}

fn print_problem_stats(problem: &problem::Problem) {
    let avg_tracks = problem.trains.iter().map(|t| {
        t.visits
            .iter()
            .filter(|v| problem.conflicts.contains(&(v.0, v.0)))
            .count()
    }).sum::<usize>() as f32 / problem.trains.len() as f32;
    let mut conflicting_visit_pairs = 0;
    for t1 in 0..problem.trains.len() {
        for t2 in (t1 + 1)..problem.trains.len() {
            for (r1, _, _) in problem.trains[t1].visits.iter() {
                for (r2, _, _) in problem.trains[t2].visits.iter() {
                    if problem.conflicts.contains(&(*r1, *r2)) {
                        conflicting_visit_pairs += 1;
                    }
                }
            }
        }
    }

    let delays = 0;
    let avgdelay = 0;

    println!(
        "trains {} tracks {} avgtracks {:.2} trackpairs {} delays {} avgdelay{}",
        problem.trains.len(),
        problem.conflicts.len(),
        avg_tracks,
        conflicting_visit_pairs,
        delays,
        avgdelay,
    );
}

#[cfg(test)]
mod tests {
    #[test]
    pub fn testproblem() {
        let problem = crate::problem::problem1_with_stations();
        let result = crate::solver::solve(satcoder::solvers::minisat::Solver::new(), &problem).unwrap();
        let score = problem.verify_solution(&result);
        assert!(score.is_some());
    }

    #[test]
    pub fn samescore() {
        let problem = crate::problem::problem1_with_stations();

        let result = crate::solver::solve(satcoder::solvers::minisat::Solver::new(), &problem).unwrap();
        let first_score = problem.verify_solution(&result);

        for _ in 0..100 {
            let result = crate::solver::solve(satcoder::solvers::minisat::Solver::new(), &problem).unwrap();
            let score = problem.verify_solution(&result);
            assert!(score == first_score);
        }
    }

    #[test]
    pub fn samescore2() {
        let (problem, _train_names, _res_names) =
            crate::parser::read_file("instances/Instance20.xml");

        let result = crate::solver::solve(satcoder::solvers::minisat::Solver::new(), &problem).unwrap();
        let first_score = problem.verify_solution(&result);

        for _ in 0..100 {
            let result = crate::solver::solve(satcoder::solvers::minisat::Solver::new(), &problem).unwrap();
            let score = problem.verify_solution(&result);
            assert!(score == first_score);
        }
    }
}
