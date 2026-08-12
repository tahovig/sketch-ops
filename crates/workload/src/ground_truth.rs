use sketches::exact::ExactCounter;

pub struct GroundTruth {
    counter: ExactCounter,
}

impl GroundTruth {
    pub fn from_stream(stream: &[u64]) -> Self {
        let mut counter = ExactCounter::new();
        for &item in stream {
            counter.insert(item);
        }
        Self { counter }
    }

    pub fn query(&self, item: u64) -> u64 {
        self.counter.query(item)
    }

    pub fn top_k(&self, k: usize) -> Vec<(u64, u64)> {
        self.counter.top_k(k)
    }

    pub fn cardinality(&self) -> usize {
        self.counter.cardinality()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ground_truth_matches_hand_crafted_stream() {
        let stream = vec![1u64, 1, 2, 3, 3, 3, 4];
        let gt = GroundTruth::from_stream(&stream);
        assert_eq!(gt.query(1), 2);
        assert_eq!(gt.query(2), 1);
        assert_eq!(gt.query(3), 3);
        assert_eq!(gt.query(4), 1);
        assert_eq!(gt.query(999), 0);
        assert_eq!(gt.cardinality(), 4);
        assert_eq!(gt.top_k(2), vec![(3u64, 3u64), (1u64, 2u64)]);
    }
}
