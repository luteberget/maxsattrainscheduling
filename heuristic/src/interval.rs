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

    pub fn intersect(&self, other: &Self) -> Self {
        let time_start = self.time_start.max(other.time_start);
        let time_end = self.time_end.min(other.time_end);

        if time_start <= time_end {
            Self {
                time_start,
                time_end,
            }
        } else {
            let midpoint = (time_start + time_end) / 2;
            Self {
                time_start: midpoint,
                time_end: midpoint,
            }
        }
    }

    pub fn envelope(&self, other: &Self) -> Self {
        let time_start = self.time_start.min(other.time_start);
        let time_end = self.time_end.max(other.time_end);

        Self {
            time_start,
            time_end,
        }
    }

    pub fn length(&self) -> TimeValue {
        self.time_end - self.time_start
    }
}
