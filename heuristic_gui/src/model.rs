use std::rc::Rc;

use heuristic::problem::*;
use serde::Deserialize;

#[derive(Deserialize, Debug)]
pub struct Input {
    pub symmetric :bool,
    pub problem :Problem,
    pub draw_tracks :Vec<DrawTrack>,
}

#[derive(Deserialize, Debug)]
pub struct DrawTrack {
    pub p_a :[f64;2],
    pub p_b :[f64;2],
    pub name :String, 
}

pub struct Model {
    pub problem :Rc<Problem>,
    pub draw_tracks :Rc<Vec<DrawTrack>>,
    pub solver :heuristic::solver::ConflictSolver,
    pub selected_train :usize,
}