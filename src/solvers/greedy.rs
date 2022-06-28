use crate::problem::Problem;

use super::SolverError;

pub fn solve(problem: &Problem) -> Result<Vec<Vec<i32>>, SolverError> {

    #[allow(unused)]
    let mut constraints :Vec<()>= Vec::new();

    for train in problem.trains.iter() {
        for _visit in train.visits.iter() {

        }
    }
    

    todo!()
}