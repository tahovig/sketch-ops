use crate::hash::HashFamily;
use crate::indexed_heap::{HeapEntry, IndexedMinHeap};
use crate::traits::HeavyHitterSketch;

const CMS_DEPTH: usize = 4;

pub struct CountMinSketch {
    depth: usize,
    width: usize,
    counters: Vec<u32>,
    hash: HashFamily,
    heap: IndexedMinHeap,
    k: usize,
    memory_bytes: usize,
}

impl CountMinSketch {
    pub fn new(budget_bytes: usize, k: usize, seed: u64) -> Self {
        let heap_bytes = k * std::mem::size_of::<HeapEntry>();
        let remaining = budget_bytes.saturating_sub(heap_bytes);
        let cell_bytes = std::mem::size_of::<u32>();
        let total_cells = (remaining / cell_bytes).max(CMS_DEPTH);
        let width = (total_cells / CMS_DEPTH).max(1);
        let counters = vec![0u32; CMS_DEPTH * width];
        let hash = HashFamily::new(CMS_DEPTH, seed);
        let actual_memory = heap_bytes + CMS_DEPTH * width * cell_bytes;
        Self {
            depth: CMS_DEPTH,
            width,
            counters,
            hash,
            heap: IndexedMinHeap::new(),
            k,
            memory_bytes: actual_memory,
        }
    }

    fn index(&self, row: usize, col: usize) -> usize {
        row * self.width + col
    }

    fn update_top_k(&mut self, item: u64, estimate: u64) {
        if self.k == 0 {
            return;
        }
        if self.heap.contains(item) {
            self.heap.set_count(item, estimate);
        } else if self.heap.len() < self.k {
            self.heap.push(item, estimate);
        } else if let Some(min) = self.heap.peek_min() {
            if estimate > min.count {
                self.heap.replace_min(item, estimate);
            }
        }
    }
}

impl HeavyHitterSketch for CountMinSketch {
    fn new_with_budget(budget_bytes: usize, k: usize, seed: u64) -> Self {
        Self::new(budget_bytes, k, seed)
    }

    fn insert(&mut self, item: u64) {
        for row in 0..self.depth {
            let col = self.hash.hash_to_width(row, item, self.width);
            let idx = self.index(row, col);
            self.counters[idx] = self.counters[idx].saturating_add(1);
        }
        let estimate = self.query(item);
        self.update_top_k(item, estimate);
    }

    fn query(&self, item: u64) -> u64 {
        (0..self.depth)
            .map(|row| {
                let col = self.hash.hash_to_width(row, item, self.width);
                self.counters[self.index(row, col)] as u64
            })
            .min()
            .unwrap_or(0)
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
        "count_min"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::exact::ExactCounter;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn count_min_never_undercounts(
            items in proptest::collection::vec(0u64..500, 1..2000)
        ) {
            let mut exact = ExactCounter::new();
            let mut cms = CountMinSketch::new_with_budget(8192, 10, 7);
            for &item in &items {
                exact.insert(item);
                cms.insert(item);
            }
            for &item in &items {
                prop_assert!(cms.query(item) >= exact.query(item));
            }
        }
    }

    #[test]
    fn memory_bytes_reserves_heap_budget_before_sizing_counters() {
        // With budget=64, k=50: heap_bytes = 50*16 = 800 > 64,
        // so remaining clamps to 0, width clamps to minimum (CMS_DEPTH cells = 1),
        // giving exact predictable memory_bytes.
        // If width were incorrectly sized from full budget instead of remaining,
        // this test would fail.
        let budget = 64;
        let k = 50;
        let tiny = CountMinSketch::new_with_budget(budget, k, 1);

        // Calculate expected value derived from constants, not hardcoded:
        let heap_bytes = k * std::mem::size_of::<HeapEntry>();
        let remaining = budget.saturating_sub(heap_bytes);
        let cell_bytes = std::mem::size_of::<u32>();
        let total_cells = (remaining / cell_bytes).max(CMS_DEPTH);
        let width = (total_cells / CMS_DEPTH).max(1);
        let expected_memory = heap_bytes + CMS_DEPTH * width * cell_bytes;

        assert_eq!(tiny.memory_bytes(), expected_memory,
            "with heap-budget exhausted, memory should match tight calculation");

        // Also verify generous budget allocates more memory
        let generous = CountMinSketch::new_with_budget(1_000_000, 50, 1);
        assert!(generous.memory_bytes() > tiny.memory_bytes());
    }
}
