use std::collections::HashMap;

use crate::indexed_heap::{HeapEntry, IndexedMinHeap};
use crate::traits::HeavyHitterSketch;

const SPACE_SAVING_BYTES_PER_ENTRY: usize = 32;

pub struct SpaceSaving {
    m: usize,
    k: usize,
    heap: IndexedMinHeap,
    errors: HashMap<u64, u64>,
    memory_bytes: usize,
}

impl SpaceSaving {
    pub fn new(budget_bytes: usize, k: usize, _seed: u64) -> Self {
        let heap_bytes = k * std::mem::size_of::<HeapEntry>();
        let remaining = budget_bytes.saturating_sub(heap_bytes);
        let m = (remaining / SPACE_SAVING_BYTES_PER_ENTRY).max(1);
        let actual_memory = heap_bytes + m * SPACE_SAVING_BYTES_PER_ENTRY;
        Self {
            m,
            k,
            heap: IndexedMinHeap::new(),
            errors: HashMap::new(),
            memory_bytes: actual_memory,
        }
    }

    pub fn error_of(&self, item: u64) -> u64 {
        self.errors.get(&item).copied().unwrap_or(0)
    }

    pub fn monitored_count(&self) -> usize {
        self.heap.len()
    }

    pub fn m(&self) -> usize {
        self.m
    }
}

impl HeavyHitterSketch for SpaceSaving {
    fn new_with_budget(budget_bytes: usize, k: usize, seed: u64) -> Self {
        Self::new(budget_bytes, k, seed)
    }

    fn insert(&mut self, item: u64) {
        if self.heap.contains(item) {
            let new_count = self.heap.count_of(item).unwrap() + 1;
            self.heap.set_count(item, new_count);
        } else if self.heap.len() < self.m {
            self.heap.push(item, 1);
            self.errors.insert(item, 0);
        } else if let Some(min) = self.heap.peek_min() {
            let evicted = self.heap.replace_min(item, min.count + 1);
            self.errors.remove(&evicted.item);
            self.errors.insert(item, min.count);
        }
    }

    fn query(&self, item: u64) -> u64 {
        self.heap.count_of(item).unwrap_or(0)
    }

    fn top_k(&self, k: usize) -> Vec<(u64, u64)> {
        let mut entries: Vec<(u64, u64)> = self.heap.entries().map(|e| (e.item, e.count)).collect();
        entries.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
        entries.truncate(k);
        entries
    }

    fn memory_bytes(&self) -> usize {
        self.memory_bytes
    }

    fn name(&self) -> &'static str {
        "space_saving"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::exact::ExactCounter;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn space_saving_error_bound_holds(
            items in proptest::collection::vec(0u64..500, 1..2000)
        ) {
            let mut exact = ExactCounter::new();
            let mut ss = SpaceSaving::new_with_budget(4096, 10, 7);
            for &item in &items {
                exact.insert(item);
                ss.insert(item);
            }
            for &item in &items {
                let estimate = ss.query(item);
                if estimate > 0 {
                    let error = ss.error_of(item);
                    let true_count = exact.query(item);
                    prop_assert!(estimate.saturating_sub(error) <= true_count);
                    prop_assert!(true_count <= estimate);
                }
            }
        }
    }

    #[test]
    fn space_saving_is_exact_when_cardinality_leq_m() {
        let mut ss = SpaceSaving::new_with_budget(1_000_000, 10, 7);
        let items = vec![1u64, 1, 2, 3, 3, 3, 4, 5];
        let mut exact = ExactCounter::new();
        for &item in &items {
            exact.insert(item);
            ss.insert(item);
        }
        assert!(exact.cardinality() <= ss.m());
        for item in [1u64, 2, 3, 4, 5] {
            assert_eq!(ss.query(item), exact.query(item));
            assert_eq!(ss.error_of(item), 0);
        }
    }
}
