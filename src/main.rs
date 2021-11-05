

mod problem;
mod solver;
mod intervals;

fn main() {
    let problem = problem::problem1();
    let result = solver::solve(&problem).unwrap();
    println!("Verifying solution {:?}", problem.verify_solution(&result));
    println!("Result {:#?}", result);
}
