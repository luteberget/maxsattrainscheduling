
pub struct Problem {
    pub trains: Vec<Train>,
    pub tracks: Vec<Track>,
}

pub type TrackRef = i32;
pub type TrainRef = i32;
pub type TimeValue = u32;
pub type VisitRef = u32;

pub const SENTINEL_TRACK: TrackRef = -1;
pub const SENTINEL_TRAIN: TrainRef = -1;

pub struct Track {
    pub prevs: Vec<TrackRef>,
    pub nexts: Vec<TrackRef>,
    pub travel_time: TimeValue,
}

pub struct Train {
    pub appears_at: TimeValue,
    pub visits: Vec<Visit>,
}

pub struct Visit {
    pub resource_alternatives: Vec<TrackRef>,
    pub earliest_out: TimeValue,
    pub measure_delay: Option<i32>,
    pub slack: u32,
}

pub type Schedule = Vec<Vec<TimeValue>>;