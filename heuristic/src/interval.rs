use crate::problem::TimeValue;

#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Debug)]
pub struct TimeInterval {
    pub time_start: TimeValue,
    pub time_end: TimeValue,
}

impl Default for TimeInterval {
    fn default() -> Self {
        INTERVAL_MAX
    }
}

pub const INTERVAL_MAX: TimeInterval = TimeInterval {
    time_start: TimeValue::MAX,
    time_end: TimeValue::MAX,
};

pub const INTERVAL_MIN: TimeInterval = TimeInterval {
    time_start: TimeValue::MIN,
    time_end: TimeValue::MIN,
};

impl TimeInterval {
    pub fn duration(start: TimeValue, duration: TimeValue) -> TimeInterval {
        TimeInterval {
            time_start: start,
            time_end: start + duration,
        }
    }

    pub fn overlap(&self, other: &Self) -> bool {
        !(self.time_end <= other.time_start || other.time_end <= self.time_start)
    }
}
