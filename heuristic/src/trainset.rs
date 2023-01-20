use crate::TrainSolver;

pub struct TrainSet<Train> {
    pub trains: Vec<Train>,
    pub slacks: Vec<Vec<i32>>,
    pub train_lbs: Vec<i32>,
    pub train_const_lbs: Vec<i32>,
    pub lb: i32,
    pub dirty_trains: Vec<u32>,
    pub original_trains :Vec<crate::problem::Train>,
}

impl<Train:TrainSolver> TrainSet<Train> {
    pub fn is_dirty(&self) -> bool {
        !self.dirty_trains.is_empty()
    }
}