use crate::hash::HashFamily;
use crate::indexed_heap::{HeapEntry, IndexedMinHeap};
use crate::traits::HeavyHitterSketch;

const PEEL_DEPTH: usize = 4;

#[derive(Debug, Clone, Copy)]
struct Cell {
    candidate_fingerprint: u32,
    vote_margin: u32,
    raw_total: u32,
}

pub struct PeelSketch {
    depth: usize,
    width: usize,
    cells: Vec<Cell>,
    hash: HashFamily,
    heap: IndexedMinHeap,
    k: usize,
    memory_bytes: usize,
}

impl PeelSketch {
    pub fn new(budget_bytes: usize, k: usize, seed: u64) -> Self {
        let heap_bytes = k * std::mem::size_of::<HeapEntry>();
        let remaining = budget_bytes.saturating_sub(heap_bytes);
        let cell_bytes = std::mem::size_of::<Cell>();
        let total_cells = (remaining / cell_bytes).max(PEEL_DEPTH);
        let width = (total_cells / PEEL_DEPTH).max(1);
        Self::build(PEEL_DEPTH, width, k, seed, heap_bytes)
    }

    pub fn with_dimensions(depth: usize, width: usize, k: usize, seed: u64) -> Self {
        let heap_bytes = k * std::mem::size_of::<HeapEntry>();
        Self::build(depth, width, k, seed, heap_bytes)
    }

    fn build(depth: usize, width: usize, k: usize, seed: u64, heap_bytes: usize) -> Self {
        let cells = vec![Cell { candidate_fingerprint: 0, vote_margin: 0, raw_total: 0 }; depth * width];
        let hash = HashFamily::new(depth + 1, seed);
        let actual_memory = heap_bytes + depth * width * std::mem::size_of::<Cell>();
        Self {
            depth,
            width,
            cells,
            hash,
            heap: IndexedMinHeap::new(),
            k,
            memory_bytes: actual_memory,
        }
    }

    pub fn fingerprint_of(&self, item: u64) -> u32 {
        let raw = self.hash.hash(self.depth, item) as u32;
        if raw == 0 {
            1
        } else {
            raw
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

impl HeavyHitterSketch for PeelSketch {
    fn new_with_budget(budget_bytes: usize, k: usize, seed: u64) -> Self {
        Self::new(budget_bytes, k, seed)
    }

    fn insert(&mut self, item: u64) {
        let fp = self.fingerprint_of(item);
        for row in 0..self.depth {
            let col = self.hash.hash_to_width(row, item, self.width);
            let idx = self.index(row, col);
            let cell = self.cells[idx];
            let new_total = cell.raw_total.saturating_add(1);
            if cell.raw_total == 0 {
                self.cells[idx] = Cell { candidate_fingerprint: fp, vote_margin: 1, raw_total: new_total };
            } else if fp == cell.candidate_fingerprint {
                self.cells[idx] = Cell {
                    candidate_fingerprint: cell.candidate_fingerprint,
                    vote_margin: cell.vote_margin.saturating_add(1),
                    raw_total: new_total,
                };
            } else {
                let new_vote = cell.vote_margin - 1;
                if new_vote == 0 {
                    self.cells[idx] = Cell { candidate_fingerprint: fp, vote_margin: 1, raw_total: new_total };
                } else {
                    self.cells[idx] = Cell {
                        candidate_fingerprint: cell.candidate_fingerprint,
                        vote_margin: new_vote,
                        raw_total: new_total,
                    };
                }
            }
        }
        let estimate = self.query(item);
        self.update_top_k(item, estimate);
    }

    fn query(&self, item: u64) -> u64 {
        let fp = self.fingerprint_of(item);
        (0..self.depth)
            .map(|row| {
                let col = self.hash.hash_to_width(row, item, self.width);
                let cell = self.cells[self.index(row, col)];
                if cell.candidate_fingerprint == fp {
                    cell.raw_total as u64
                } else {
                    cell.raw_total.saturating_sub(cell.vote_margin) as u64
                }
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
        "peel_sketch"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn hand_traced_boyer_moore_sequence_is_deterministic() {
        let seed = 42u64;
        let mut ps = PeelSketch::with_dimensions(1, 1, 5, seed);
        let item_a = 100u64;
        let item_b = 200u64;
        let fp_a = ps.fingerprint_of(item_a);
        let fp_b = ps.fingerprint_of(item_b);
        assert_ne!(fp_a, fp_b, "test requires distinct fingerprints for these items/seed");

        // Insert A: empty cell -> candidate=fp_a, vote_margin=1, raw_total=1.
        ps.insert(item_a);
        assert_eq!(ps.query(item_a), 1, "A is the sole occupant, reads raw_total directly");
        assert_eq!(ps.query(item_b), 0, "raw_total(1) - vote_margin(1) = 0 for the non-candidate");

        // Insert A again: matches candidate -> vote_margin=2, raw_total=2.
        ps.insert(item_a);
        assert_eq!(ps.query(item_a), 2);
        assert_eq!(ps.query(item_b), 0, "raw_total(2) - vote_margin(2) = 0");

        // Insert B: mismatch -> vote_margin 2->1, raw_total=3, candidate stays A.
        ps.insert(item_b);
        assert_eq!(ps.query(item_a), 3, "A is still candidate, reads raw_total=3 directly");
        assert_eq!(ps.query(item_b), 2, "raw_total(3) - vote_margin(1) = 2");

        // Insert B again: mismatch -> vote_margin 1->0 -> reset: candidate=fp_b, vote_margin=1, raw_total=4.
        ps.insert(item_b);
        assert_eq!(ps.query(item_b), 4, "B is now candidate, reads raw_total=4 directly");
        assert_eq!(ps.query(item_a), 3, "raw_total(4) - vote_margin(1) = 3");
    }

    #[test]
    fn minority_item_residual_is_bounded_and_tighter_than_raw_total() {
        let seed = 7u64;
        let mut ps = PeelSketch::with_dimensions(1, 1, 3, seed);
        let heavy = 111u64;
        let light = 222u64;
        let fp_heavy = ps.fingerprint_of(heavy);
        let fp_light = ps.fingerprint_of(light);
        assert_ne!(fp_heavy, fp_light, "test requires distinct fingerprints for these items/seed");

        for _ in 0..10 {
            ps.insert(heavy);
        }
        ps.insert(light);
        // After 10 inserts of heavy: candidate=fp_heavy, vote_margin=10, raw_total=10.
        // Insert light (mismatch): vote_margin 10->9, raw_total=11, candidate stays heavy.
        let raw_total = 11u64;

        assert_eq!(ps.query(heavy), raw_total, "majority item reads raw_total directly");
        let light_residual = ps.query(light);
        assert_eq!(light_residual, raw_total.saturating_sub(9), "raw_total(11) - vote_margin(9) = 2");
        assert!(
            light_residual < raw_total,
            "the residual must be a strictly tighter bound than the raw, unfiltered cell total"
        );
    }

    #[test]
    fn memory_bytes_reserves_heap_budget_before_sizing_cells() {
        // With budget=64, k=50: heap_bytes = 50*16 = 800 > 64,
        // so remaining clamps to 0, width clamps to minimum (PEEL_DEPTH cells = 1),
        // giving exact predictable memory_bytes.
        let budget = 64;
        let k = 50;
        let tiny = PeelSketch::new_with_budget(budget, k, 1);

        let heap_bytes = k * std::mem::size_of::<HeapEntry>();
        let remaining = budget.saturating_sub(heap_bytes);
        let cell_bytes = std::mem::size_of::<Cell>();
        let total_cells = (remaining / cell_bytes).max(PEEL_DEPTH);
        let width = (total_cells / PEEL_DEPTH).max(1);
        let expected_memory = heap_bytes + PEEL_DEPTH * width * cell_bytes;

        assert_eq!(
            tiny.memory_bytes(),
            expected_memory,
            "with heap-budget exhausted, memory should match tight calculation"
        );

        let generous = PeelSketch::new_with_budget(1_000_000, 50, 1);
        assert!(
            generous.memory_bytes() > tiny.memory_bytes(),
            "generous budget should yield more memory than tiny budget"
        );
    }

    proptest! {
        #[test]
        fn vote_margin_never_exceeds_true_count_of_final_candidate(
            items in proptest::collection::vec(0u64..20, 1..200)
        ) {
            // depth=1, width=1: every item collides into the single cell,
            // so the cell's history is exactly the full `items` sequence.
            let mut ps = PeelSketch::with_dimensions(1, 1, 1, 99);
            for &item in &items {
                ps.insert(item);
            }
            let final_candidate_fp = ps.cells[0].candidate_fingerprint;
            let vote_margin = ps.cells[0].vote_margin;

            let true_count = items
                .iter()
                .filter(|&&item| ps.fingerprint_of(item) == final_candidate_fp)
                .count() as u64;

            prop_assert!(
                (vote_margin as u64) <= true_count,
                "vote_margin={} true_count={}",
                vote_margin,
                true_count
            );
        }

        #[test]
        fn structural_invariants_hold_after_arbitrary_inserts(
            items in proptest::collection::vec(0u64..1000, 1..500)
        ) {
            let mut ps = PeelSketch::new_with_budget(4096, 10, 123);
            for &item in &items {
                ps.insert(item);
            }
            for cell in &ps.cells {
                prop_assert!(
                    cell.raw_total as usize <= items.len(),
                    "a single cell cannot receive more contributions than the whole stream"
                );
                if cell.raw_total > 0 {
                    let is_real = items.iter().any(|&item| ps.fingerprint_of(item) == cell.candidate_fingerprint);
                    prop_assert!(is_real, "candidate_fingerprint must correspond to an actually-inserted item");
                }
            }
        }
    }
}
