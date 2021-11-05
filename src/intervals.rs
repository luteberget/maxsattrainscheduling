pub struct ResourceOccupation {
    intervals: NonOverlappingIntervals,
    items: Vec<Option<ResourceOccupationItem>>,
    conflicts: Vec<u32>,
}

struct ResourceOccupationItem {
    a: i32,
    b: i32,
    conflicts_with: Option<u32>,
}

impl ResourceOccupation {
    pub fn new() -> Self {
        ResourceOccupation {
            intervals: NonOverlappingIntervals::new(),
            items: Vec::new(),
            conflicts: Vec::new(),
        }
    }

    pub fn set(&mut self, item: u32, a: i32, b: i32) {
        
    }
}

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Debug)]
enum IntervalSide {
    End,
    Begin,
}

pub struct NonOverlappingIntervals {
    intervals: Vec<((i32, IntervalSide), u32)>,
}

impl NonOverlappingIntervals {
    pub fn new() -> Self {
        NonOverlappingIntervals {
            intervals: Vec::new(),
        }
    }

    pub fn try_insert(&mut self, a: i32, b: i32, item: u32) -> Result<(), u32> {
        assert!(a < b);

        let start = (a, IntervalSide::Begin);
        let end = (b, IntervalSide::End);

        let idx1 = match self.intervals.binary_search_by_key(&&start, |(k, _)| k) {
            Ok(idx) => {
                return Err(self.intervals[idx].1);
            }
            Err(idx) => idx,
        };

        if idx1 > 0 && !matches!(self.intervals[idx1 - 1].0 .1, IntervalSide::End) {
            return Err(self.intervals[idx1 - 1].1);
        }

        let idx2 = match self.intervals.binary_search_by_key(&&end, |(k, _)| k) {
            Ok(idx) => {
                return Err(self.intervals[idx].1);
            }
            Err(idx) => idx,
        };

        if idx2 < self.intervals.len() && !matches!(self.intervals[idx2].0 .1, IntervalSide::Begin)
        {
            return Err(self.intervals[idx2].1);
        }

        assert!(idx1 <= idx2);
        if idx1 != idx2 {
            return Err(self.intervals[idx2 - 1].1);
        }

        self.intervals
            .splice(idx1..idx1, [(start, item), (end, item)]);

        Ok(())
    }

    pub fn remove(&mut self, a: i32, b: i32) -> Option<u32> {
        let start = (a, IntervalSide::Begin);
        let end = (b, IntervalSide::End);

        let idx1 = match self.intervals.binary_search_by_key(&&start, |(k, _)| k) {
            Ok(idx) => idx,
            Err(_) => {
                return None;
            }
        };

        let idx2 = match self.intervals.binary_search_by_key(&&end, |(k, _)| k) {
            Ok(idx) => idx,
            Err(_) => {
                return None;
            }
        };

        assert!(idx1 + 1 == idx2);
        assert!(self.intervals[idx1].1 == self.intervals[idx2].1);
        let item = self.intervals[idx1].1;
        self.intervals.drain(idx1..=idx2);

        Some(item)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    pub fn test1() {
        {
            let mut x = NonOverlappingIntervals::new();
            for i in (-100..100).step_by(2) {
                assert!(x.try_insert(i, i + 1, 0).is_ok());
            }
        }
        {
            let mut x = NonOverlappingIntervals::new();
            for i in (-100..100).step_by(2).rev() {
                assert!(x.try_insert(i, i + 1, 0).is_ok());
            }
        }
        {
            let mut x = NonOverlappingIntervals::new();
            for i in (-100..100).step_by(1) {
                assert!(x.try_insert(i, i + 1, 0).is_ok());
            }
        }
        {
            let mut x = NonOverlappingIntervals::new();
            for i in (-100..100).step_by(1).rev() {
                assert!(x.try_insert(i, i + 1, 0).is_ok());
            }
        }
    }

    #[test]
    pub fn test2() {
        let mut x = NonOverlappingIntervals::new();
        assert!(x.try_insert(50, 60, 999).is_ok());
        assert!(x.try_insert(40, 50, 0).is_ok());
        assert!(x.remove(40, 50) == Some(0));

        assert!(x.try_insert(45, 55, 0) == Err(999));
        assert!(x.try_insert(50, 55, 0) == Err(999));
        assert!(x.try_insert(55, 56, 0) == Err(999));

        assert!(x.try_insert(45, 60, 0) == Err(999));
        assert!(x.try_insert(50, 60, 0) == Err(999));
        assert!(x.try_insert(55, 60, 0) == Err(999));

        assert!(x.try_insert(45, 65, 0) == Err(999));
        assert!(x.try_insert(50, 65, 0) == Err(999));
        assert!(x.try_insert(55, 65, 0) == Err(999));

        assert!(x.try_insert(61, 65, 0).is_ok());
        assert!(x.try_insert(60, 61, 0).is_ok());
    }
}
