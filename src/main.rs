mod parser;
mod problem;
mod solver;

fn main() {
    let (problem, train_names, res_names) = parser::read_file("instances/Instance20.xml");

    println!("Problem: {:#?}", problem);
    println!("Trains: {:?}", train_names);
    println!("Resources: {:?}", res_names);
    println!("Conflict resources: {}", problem.conflicts.len());

    // let problem = problem::problem1();
    let result = solver::solve(&problem).unwrap();
    println!("Verifying solution {:?}", problem.verify_solution(&result));
    println!("Result {:#?}", result);
    println!("t0 {}", train_names[0]);
    println!("t0v1 {}", res_names[problem.trains[0].visits[1].0]);
    println!("t3 {}", train_names[3]);
    println!("t3v0 {}", res_names[problem.trains[3].visits[0].0]);
}

#[cfg(test)]
mod tests {
    #[test]
    pub fn testproblem() {
        let problem = crate::problem::problem1_with_stations();
        let result = crate::solver::solve(&problem).unwrap();
        let score = problem.verify_solution(&result);
        assert!(score.is_some());
    }

    #[test]
    pub fn samescore() {
        let problem = crate::problem::problem1_with_stations();

        let result = crate::solver::solve(&problem).unwrap();
        let first_score = problem.verify_solution(&result);

        for _ in 0..100 {
            let result = crate::solver::solve(&problem).unwrap();
            let score = problem.verify_solution(&result);
            assert!(score == first_score);
        }
    }

    #[test]
    pub fn samescore2() {
        let (problem, _train_names, _res_names) = crate::parser::read_file("instances/Instance20.xml");

        let result = crate::solver::solve(&problem).unwrap();
        let first_score = problem.verify_solution(&result);

        for _ in 0..100 {
            let result = crate::solver::solve(&problem).unwrap();
            let score = problem.verify_solution(&result);
            assert!(score == first_score);
        }
    }
}
