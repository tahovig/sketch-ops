use std::collections::HashMap;

pub struct ExactCounter {
    counts: HashMap<u64, u64>,
}

impl ExactCounter {
    pub fn new() -> Self {
        Self { counts: HashMap::new() }
    }

    pub fn insert(&mut self, item: u64) {
        *self.counts.entry(item).or_insert(0) += 1;
    }

    pub fn query(&self, item: u64) -> u64 {
        *self.counts.get(&item).unwrap_or(&0)
    }

    pub fn top_k(&self, k: usize) -> Vec<(u64, u64)> {
        let mut entries: Vec<(u64, u64)> = self.counts.iter().map(|(&item, &count)| (item, count)).collect();
        entries.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
        entries.truncate(k);
        entries
    }

    pub fn cardinality(&self) -> usize {
        self.counts.len()
    }
}

impl Default for ExactCounter {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn insert_and_query_track_exact_counts() {
        let mut counter = ExactCounter::new();
        counter.insert(42);
        counter.insert(42);
        counter.insert(7);
        assert_eq!(counter.query(42), 2);
        assert_eq!(counter.query(7), 1);
        assert_eq!(counter.query(999), 0);
    }

    #[test]
    fn top_k_orders_by_count_descending_with_stable_tie_break() {
        let mut counter = ExactCounter::new();
        for item in [1u64, 1, 1, 2, 2, 3] {
            counter.insert(item);
        }
        assert_eq!(counter.top_k(2), vec![(1u64, 3u64), (2u64, 2u64)]);
        assert_eq!(counter.cardinality(), 3);
    }

    #[test]
    fn top_k_with_k_larger_than_cardinality_returns_everything() {
        let mut counter = ExactCounter::new();
        counter.insert(5);
        counter.insert(6);
        assert_eq!(counter.top_k(10).len(), 2);
    }
}
