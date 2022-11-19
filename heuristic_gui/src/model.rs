
use heuristic::problem::*;
use serde::Deserialize;

#[derive(Deserialize, Debug)]
pub struct Input {
    pub problem :Problem,
    // pub levels :Vec<u32>,
}

#[derive(Deserialize, Debug)]
pub struct DrawTrack {
    pub p_a :[f64;2],
    pub p_b :[f64;2],
    // pub name :String, 
}

pub struct Model {
    // pub problem :Rc<Problem>,
    // pub levels :Vec<u32>,
    pub solver :heuristic::solver::ConflictSolver,
    pub selected_train :usize,
}