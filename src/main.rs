

mod problem;
mod solver;
mod railway;

fn main() {
    let result = solver::solve(&problem::problem1());
    println!("Result {:#?}", result);
}
