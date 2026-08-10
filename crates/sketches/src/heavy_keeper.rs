use crate::hash::{HashFamily, SplitMix64};
use crate::indexed_heap::{HeapEntry, IndexedMinHeap};
use crate::traits::HeavyHitterSketch;

const HK_DEPTH: usize = 4;
const HK_DECAY_BASE: f64 = 1.08;

#[derive(Debug, Clone, Copy)]
struct Cell {
    fingerprint: u32,
    count: u32,
}

fn decay_rng_seed(seed: u64) -> u64 {
    seed ^ 0xD1B54A32D192ED03
}

pub struct HeavyKeeper {
    depth: usize,
    width: usize,
    cells: Vec<Cell>,
    hash: HashFamily,
    prng: SplitMix64,
    decay_base: f64,
    heap: IndexedMinHeap,
    k: usize,
    memory_bytes: usize,
}

impl HeavyKeeper {
    pub fn new(budget_bytes: usize, k: usize, seed: u64) -> Self {
        let heap_bytes = k * std::mem::size_of::<HeapEntry>();
        let remaining = budget_bytes.saturating_sub(heap_bytes);
        let cell_bytes = std::mem::size_of::<Cell>();
        let total_cells = (remaining / cell_bytes).max(HK_DEPTH);
        let width = (total_cells / HK_DEPTH).max(1);
        Self::build(HK_DEPTH, width, k, seed, heap_bytes)
    }

    pub fn with_dimensions(depth: usize, width: usize, k: usize, seed: u64) -> Self {
        let heap_bytes = k * std::mem::size_of::<HeapEntry>();
        Self::build(depth, width, k, seed, heap_bytes)
    }

    fn build(depth: usize, width: usize, k: usize, seed: u64, heap_bytes: usize) -> Self {
        let cells = vec![Cell { fingerprint: 0, count: 0 }; depth * width];
        let hash = HashFamily::new(depth + 1, seed);
        let prng = SplitMix64::new(decay_rng_seed(seed));
        let actual_memory = heap_bytes + depth * width * std::mem::size_of::<Cell>();
        Self {
            depth,
            width,
            cells,
            hash,
            prng,
            decay_base: HK_DECAY_BASE,
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

    fn decay_survives(&mut self, count: u32) -> bool {
        let probability = self.decay_base.powi(-(count as i32));
        self.prng.next_f64() < probability
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

impl HeavyHitterSketch for HeavyKeeper {
    fn new_with_budget(budget_bytes: usize, k: usize, seed: u64) -> Self {
        Self::new(budget_bytes, k, seed)
    }

    fn insert(&mut self, item: u64) {
        let fp = self.fingerprint_of(item);
        for row in 0..self.depth {
            let col = self.hash.hash_to_width(row, item, self.width);
            let idx = self.index(row, col);
            let cell = self.cells[idx];
            if cell.count == 0 {
                self.cells[idx] = Cell { fingerprint: fp, count: 1 };
            } else if cell.fingerprint == fp {
                self.cells[idx].count = cell.count.saturating_add(1);
            } else if self.decay_survives(cell.count) {
                let new_count = cell.count - 1;
                if new_count == 0 {
                    self.cells[idx] = Cell { fingerprint: fp, count: 1 };
                } else {
                    self.cells[idx].count = new_count;
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
                if cell.fingerprint == fp {
                    cell.count as u64
                } else {
                    0
                }
            })
            .max()
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
        "heavy_keeper"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hand_traced_decay_sequence_is_deterministic() {
        let seed = 42u64;
        let mut hk = HeavyKeeper::with_dimensions(1, 1, 5, seed);
        let mut shadow_prng = SplitMix64::new(decay_rng_seed(seed));

        let item_a = 100u64;
        let item_b = 200u64;

        // First insert of A: empty cell -> claimed at count 1, no PRNG draw.
        hk.insert(item_a);
        assert_eq!(hk.query(item_a), 1);

        // Second insert of A: fingerprint matches -> count increments, no PRNG draw.
        hk.insert(item_a);
        assert_eq!(hk.query(item_a), 2);

        // Insert of B: fingerprint mismatch against A's cell (count=2) -> one PRNG draw.
        let survives = shadow_prng.next_f64() < 1.08_f64.powi(-2);
        hk.insert(item_b);
        if survives {
            assert_eq!(hk.query(item_a), 1, "A should have decayed by one");
            assert_eq!(hk.query(item_b), 0, "B not installed yet, cell still holds A's fingerprint");
        } else {
            assert_eq!(hk.query(item_a), 2, "A's count is unchanged when decay is rejected");
            assert_eq!(hk.query(item_b), 0, "B still absent when decay is rejected");
        }
    }

    #[test]
    fn fingerprint_collision_does_not_panic() {
        let mut hk = HeavyKeeper::with_dimensions(1, 4, 10, 42);
        let item_a = 111u64;
        let item_b = 222u64;
        let fp_b = hk.fingerprint_of(item_b);

        let col_b = hk.hash.hash_to_width(0, item_b, hk.width);
        let idx_b = hk.index(0, col_b);
        // Simulate a collision: the cell at item_b's slot already holds a
        // count under a fingerprint that happens to equal item_b's own
        // fingerprint (as if a different original item collided there).
        hk.cells[idx_b] = Cell { fingerprint: fp_b, count: 7 };

        let result = hk.query(item_b);
        assert_eq!(result, 7, "fingerprint collision must misattribute the existing count, not panic");

        // Continuing to insert must not panic even though the cell's history
        // is not really item_b's own.
        hk.insert(item_a);
        hk.insert(item_b);
    }
}
