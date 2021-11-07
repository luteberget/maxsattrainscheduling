

mod problem;
mod solver;
mod parser;

fn main() {

    let problem = parser::read_file("instances/Instance20.xml");

    let problem = problem::problem1();
    let result = solver::solve(&problem).unwrap();
    println!("Verifying solution {:?}", problem.verify_solution(&result));
    println!("Result {:#?}", result);
}
