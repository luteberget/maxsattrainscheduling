use std::collections::HashMap;

use eframe::epaint::Color32;
use heuristic::problem::*;
use serde::Deserialize;

#[derive(Deserialize, Debug)]
pub struct Input {
    pub problem: Problem,
}

pub struct Model {
    pub solver: heuristic::solver::ConflictSolver<heuristic::queue_train::QueueTrainSolver>,
    pub selected_train: usize,
    pub current_cost: Option<i32>,
    pub locations: HashMap<String, i32>,
}

pub struct AutoColor {
    next_auto_color_idx: u32,
}

impl AutoColor {
    pub fn new() -> Self {
        Self {
            next_auto_color_idx: 0,
        }
    }
    pub fn next(&mut self) -> Color32 {
        let i = self.next_auto_color_idx;
        self.next_auto_color_idx += 1;
        let golden_ratio = (5.0_f32.sqrt() - 1.0) / 2.0; // 0.61803398875
        let h = i as f32 * golden_ratio;
        eframe::epaint::color::Hsva::new(h, 0.85, 0.5, 1.0).into() // TODO(emilk): OkLab or some other perspective color space
    }
}
