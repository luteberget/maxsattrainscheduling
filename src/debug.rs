
#[derive(Debug)]
pub struct DebugInfo {
    pub iteration: usize,
    pub solution: Vec<Vec<i32>>,
    pub actions: Vec<SolverAction>,
}

#[derive(Debug)]
pub struct ResourceInterval {
    pub train_idx: usize,
    pub visit_idx: usize,
    pub resource_idx: usize,
    pub time_in: i32,
    pub time_out: i32,
}

#[derive(Debug)]
pub enum SolverAction {
    TravelTimeConflict(ResourceInterval),
    ResourceConflict(ResourceInterval, ResourceInterval),
    Core(usize),
}
