# PeelSketch Phase 2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement PeelSketch — a heavy-hitter sketch whose cells determine
purity from an in-place, statistics-only majority-vote signal (no auxiliary
key-tracking structure) — wire it into the Phase 1 benchmark harness as a
fourth algorithm, and empirically compare it against HeavyKeeper to apply
the design's fallback-trigger criterion.

**Architecture:** One new file, `crates/sketches/src/peel_sketch.rs`,
implementing the existing `HeavyHitterSketch` trait exactly as CMS/Space-
Saving/HeavyKeeper do. Each cell is `{candidate_fingerprint, vote_margin,
raw_total}`; `insert` runs a Boyer-Moore majority-vote update per row;
`query` reads each row's cell directly if the queried item is that row's
current majority candidate, or falls back to a local
`raw_total - vote_margin` upper bound (a proven single-counter Misra-Gries
guarantee, not a cross-row lookup) when it isn't, then takes the `min`
across rows exactly as Count-Min Sketch does. One follow-on change wires
the new sketch into `crates/bench/src/sweep.rs`'s `Algorithm` enum.

**Tech Stack:** Rust (edition 2021, matching the existing workspace); no new
dependencies — reuses `sketches::hash::HashFamily`,
`sketches::indexed_heap::{HeapEntry, IndexedMinHeap}`, and `proptest`
(already a dev-dependency of `sketches`).

## Global Constraints

- Item type is the existing non-generic `u64` — no generics on
  `insert`/`query`/`top_k`, matching every other sketch in this crate.
- `memory_bytes()` is an analytical/logical byte count
  (`size_of::<Cell>() * depth * width`, plus the heap reservation) — never
  OS-reported RSS. `new`/`new_with_budget` must reserve
  `k * size_of::<HeapEntry>()` bytes for the top-k heap *before* sizing the
  rest of the sketch from what remains, exactly like CMS/HeavyKeeper.
- All hashing is the existing hand-rolled `HashFamily`/fingerprint scheme
  (`hash.hash(depth, item) as u32`, with a `0` result coerced to `1`) —
  never a generic `Hasher`.
- `PEEL_DEPTH` is fixed at `4` for the primary `new`/`new_with_budget`
  constructors, matching `CMS_DEPTH`/`HK_DEPTH` exactly, so any accuracy
  difference measured against baselines is attributable to the cell/query
  design, not a confounding change in row count. A `with_dimensions`
  constructor (mirroring HeavyKeeper's) exists for direct testing only —
  never wired into the CLI or `Algorithm` enum.
- The query mechanism is **local to the single cell being read** — no
  cross-row lookups. Each row's residual estimate is either the cell's own
  `raw_total` (if the queried item is that row's majority candidate) or
  `raw_total.saturating_sub(vote_margin)` (a valid upper bound on
  everything else in that cell, per the single-counter Misra-Gries
  guarantee: `vote_margin <= true_count(candidate)` for any insert
  sequence, not only when a true majority exists). Do not implement any
  version that looks up another row's cell to estimate a blocking
  candidate's count — that mechanism was tried during design and found
  uncomputable without an auxiliary key-recovery structure (see the design
  spec's revision note in the Query / top-k section).
- `raw_total` and `vote_margin` use saturating arithmetic (`saturating_add`/
  manual saturating decrement), never wrapping — streams in this harness
  run into the millions of items.
- PeelSketch does **not** honestly satisfy CMS's never-undercount
  invariant. Do not write a test asserting `query(x) >= exact.query(x)` for
  PeelSketch — that would be testing a property the design spec explicitly
  says does not hold. The property that *does* hold and must be tested is
  the general Misra-Gries bound above.
- Exactly two files change in this plan: `crates/sketches/src/peel_sketch.rs`
  (new) and `crates/bench/src/sweep.rs` (modified). No changes to
  `workload/`, `metrics.rs`, `report.rs`, `cli.rs`, or `scripts/plot.py` —
  they are already algorithm-agnostic.
- `cargo build --workspace --release` is required before any timing/
  throughput numbers are trusted, per the existing project-wide constraint.

---

### Task 1: `PeelSketch` — struct, `HeavyHitterSketch` impl, full test suite

**Files:**
- Create: `crates/sketches/src/peel_sketch.rs`
- Modify: `crates/sketches/src/lib.rs`
- Test: inline `#[cfg(test)] mod tests` in `crates/sketches/src/peel_sketch.rs`

**Interfaces:**
- Consumes: `sketches::hash::HashFamily` (`HashFamily::new(count, seed) -> Self`, `.hash(row, item) -> u64`, `.hash_to_width(row, item, width) -> usize`); `sketches::indexed_heap::{HeapEntry, IndexedMinHeap}` (`IndexedMinHeap::new()`, `.contains(item)`, `.len()`, `.push(item, count)`, `.peek_min() -> Option<HeapEntry>`, `.replace_min(item, count)`, `.set_count(item, count)`, `.entries() -> impl Iterator<Item = &HeapEntry>`); `sketches::traits::HeavyHitterSketch`.
- Produces: `pub struct PeelSketch` implementing `HeavyHitterSketch`; `pub fn PeelSketch::with_dimensions(depth: usize, width: usize, k: usize, seed: u64) -> Self`; `pub fn PeelSketch::fingerprint_of(&self, item: u64) -> u32`; `name() -> "peel_sketch"`.

- [ ] **Step 1: Write the failing tests**

```rust
// crates/sketches/src/peel_sketch.rs
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
                vote_margin <= true_count,
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test -p sketches peel_sketch`
Expected: FAIL with a compile error — `PeelSketch` doesn't exist yet.

- [ ] **Step 3: Write the implementation**

```rust
// crates/sketches/src/peel_sketch.rs (add above the tests module, below the Cell struct)
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
```

```rust
// crates/sketches/src/lib.rs
//! Pure heavy-hitter sketch algorithms: no I/O, no CLI dependencies.
pub mod count_min;
pub mod exact;
pub mod hash;
pub mod heavy_keeper;
pub mod indexed_heap;
pub mod peel_sketch;
pub mod space_saving;
pub mod traits;
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test -p sketches peel_sketch`
Expected: PASS — all 5 tests (hand-traced sequence, minority residual bound,
memory-budget reservation ordering, and the two `proptest` property tests).

- [ ] **Step 5: Run the full sketches test suite to check for regressions**

Run: `cargo test -p sketches`
Expected: PASS — all pre-existing tests (ExactCounter, hash, CMS, Space-
Saving, HeavyKeeper, indexed heap) remain green.

- [ ] **Step 6: Commit**

```bash
git add crates/sketches/src/peel_sketch.rs crates/sketches/src/lib.rs
git commit -m "Add PeelSketch: statistics-only purity heavy-hitter sketch"
```

---

### Task 2: Wire `PeelSketch` into the sweep harness

**Files:**
- Modify: `crates/bench/src/sweep.rs`

**Interfaces:**
- Consumes: `sketches::peel_sketch::PeelSketch::new_with_budget(budget_bytes: usize, k: usize, seed: u64) -> Self` (Task 1).
- Produces: `Algorithm::PeelSketch` variant; `Algorithm::all() -> [Algorithm; 4]`.

- [ ] **Step 1: Write the failing test**

Modify the existing test in `crates/bench/src/sweep.rs`'s `#[cfg(test)] mod tests`:

```rust
// crates/bench/src/sweep.rs — inside tiny_sweep_produces_expected_row_count_and_sane_values
let expected_rows =
    args.cardinality.len() * args.skew.len() * args.memory_budgets.len() * 4 * args.trials as usize;
```

(This replaces the existing `* 3 *` with `* 4 *` — the only line in that
test that changes.)

- [ ] **Step 2: Run the test to verify it fails**

Run: `cargo test -p bench sweep`
Expected: FAIL — `tiny_sweep_produces_expected_row_count_and_sane_values`
now expects 4× the row count per (cardinality, skew, budget, trial)
combination, but `Algorithm::all()` still only returns 3 algorithms, so
`results.len()` won't match `expected_rows`.

- [ ] **Step 3: Wire in the new algorithm**

```rust
// crates/bench/src/sweep.rs — replace the top-of-file sketches:: imports
// (this is the complete, final import block — one new line added)
use sketches::count_min::CountMinSketch;
use sketches::heavy_keeper::HeavyKeeper;
use sketches::peel_sketch::PeelSketch;
use sketches::space_saving::SpaceSaving;
use sketches::traits::HeavyHitterSketch;
```

```rust
// crates/bench/src/sweep.rs — replace the Algorithm enum and impl in full
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Algorithm {
    CountMin,
    SpaceSaving,
    HeavyKeeper,
    PeelSketch,
}

impl Algorithm {
    pub fn all() -> [Algorithm; 4] {
        [Algorithm::CountMin, Algorithm::SpaceSaving, Algorithm::HeavyKeeper, Algorithm::PeelSketch]
    }

    pub fn name(&self) -> &'static str {
        match self {
            Algorithm::CountMin => "count_min",
            Algorithm::SpaceSaving => "space_saving",
            Algorithm::HeavyKeeper => "heavy_keeper",
            Algorithm::PeelSketch => "peel_sketch",
        }
    }
}
```

```rust
// crates/bench/src/sweep.rs — inside run_sweep, replace the
// `let row = match algorithm { ... };` block in full (this is the complete
// block including the three pre-existing, unchanged arms plus the new
// Algorithm::PeelSketch arm at the end)
let row = match algorithm {
    Algorithm::CountMin => {
        let (sketch, ctor_ns, insert_ns) =
            run_one::<CountMinSketch>(&stream, warmup_len, budget, args.top_k, sk_seed);
        build_row(&ctx, algorithm, trial, sk_seed, budget, ctor_ns, insert_ns, &sketch)
    }
    Algorithm::SpaceSaving => {
        let (sketch, ctor_ns, insert_ns) =
            run_one::<SpaceSaving>(&stream, warmup_len, budget, args.top_k, sk_seed);
        build_row(&ctx, algorithm, trial, sk_seed, budget, ctor_ns, insert_ns, &sketch)
    }
    Algorithm::HeavyKeeper => {
        let (sketch, ctor_ns, insert_ns) =
            run_one::<HeavyKeeper>(&stream, warmup_len, budget, args.top_k, sk_seed);
        build_row(&ctx, algorithm, trial, sk_seed, budget, ctor_ns, insert_ns, &sketch)
    }
    Algorithm::PeelSketch => {
        let (sketch, ctor_ns, insert_ns) =
            run_one::<PeelSketch>(&stream, warmup_len, budget, args.top_k, sk_seed);
        build_row(&ctx, algorithm, trial, sk_seed, budget, ctor_ns, insert_ns, &sketch)
    }
};
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `cargo test -p bench sweep`
Expected: PASS — the tiny sweep now produces
`1 * 1 * 2 * 4 * 2 = 16` rows (one cardinality × one skew × two budgets ×
four algorithms × two trials), and the Space-Saving-exactness oracle
assertion still holds since Space-Saving itself is unchanged.

- [ ] **Step 5: Run the full bench test suite and a manual smoke run**

Run: `cargo test -p bench`
Expected: PASS — all pre-existing bench tests remain green.

Run: `cargo run -p bench -- sweep --cardinality 500 --stream-length 20000 --skew 1.1 --memory-budgets 8192 --top-k 10 --trials 1 --seed 1 --output /tmp/peelsketch_smoke.csv`
Expected: prints `wrote 4 rows to /tmp/peelsketch_smoke.csv` (one row per
algorithm now, not three) with no panic.

- [ ] **Step 6: Commit**

```bash
git add crates/bench/src/sweep.rs
git commit -m "Wire PeelSketch into the sweep harness as a fourth algorithm"
```

---

### Task 3: Comparison sweep, fallback-trigger verdict, and results write-up

**Files:**
- Create: `docs/superpowers/specs/2026-08-16-peelsketch-phase2-results.md`
- No source files change in this task — it is a verification and
  documentation pass over Tasks 1–2's output, matching the design's
  Verification Plan.

**Interfaces:**
- Consumes: the `bench` binary's `sweep` subcommand (Task 2, unmodified
  CLI); `scripts/plot.py` (Phase 1, unmodified).
- Produces: a committed results document recording the fallback-trigger
  verdict from the design spec's Fallback Plan.

- [ ] **Step 1: Run the full workspace test suite and release build**

Run: `cargo test --workspace`
Expected: PASS — every test from Tasks 1–2 plus every pre-existing Phase 1
test is green.

Run: `cargo build --workspace --release`
Expected: PASS — release build succeeds with `lto = true` and
`codegen-units = 1` from the root `Cargo.toml` (unchanged from Phase 1).

- [ ] **Step 2: Run the comparison sweep**

Run:
```bash
cargo run --release --bin bench -- sweep \
  --cardinality 100000 \
  --stream-length 2000000 \
  --skew 0.8,1.2 \
  --memory-budgets 4096,65536,1048576 \
  --top-k 20 \
  --trials 3 \
  --seed 42 \
  --output results/phase2_comparison.csv
```
Expected: prints `wrote 72 rows to results/phase2_comparison.csv` (1
cardinality × 2 skews × 3 budgets × 4 algorithms × 3 trials = 72), and the
file exists with a header line plus 72 data rows.

- [ ] **Step 3: Apply the fallback-trigger comparison**

Run:
```bash
python3 -c "
import pandas as pd

df = pd.read_csv('results/phase2_comparison.csv')
stats = df.groupby(['algorithm', 'skew', 'requested_memory_bytes'])['f1_at_k'].agg(['mean', 'std']).reset_index()
stats['std'] = stats['std'].fillna(0.0)

peel = stats[stats.algorithm == 'peel_sketch'].set_index(['skew', 'requested_memory_bytes'])
keeper = stats[stats.algorithm == 'heavy_keeper'].set_index(['skew', 'requested_memory_bytes'])

header = 'skew'.rjust(6) + ' ' + 'budget'.rjust(10) + ' ' + 'peel_f1'.rjust(10) + ' ' + 'hk_f1'.rjust(10) + ' ' + 'margin'.rjust(10) + ' ' + 'noise_floor'.rjust(12) + ' ' + 'verdict'.rjust(10)
print(header)
any_real_win = False
for key in peel.index:
    if key not in keeper.index:
        continue
    p_mean, p_std = peel.loc[key, 'mean'], peel.loc[key, 'std']
    h_mean, h_std = keeper.loc[key, 'mean'], keeper.loc[key, 'std']
    margin = p_mean - h_mean
    noise_floor = p_std + h_std
    is_real_win = margin > noise_floor and margin > 0
    if is_real_win:
        any_real_win = True
    skew, budget = key
    verdict_label = 'WIN' if is_real_win else 'no'
    print(f'{skew:>6} {budget:>10} {p_mean:>10.4f} {h_mean:>10.4f} {margin:>10.4f} {noise_floor:>12.4f} {verdict_label:>10}')

if any_real_win:
    print('\nVERDICT: PeelSketch beats HeavyKeeper outside the trial-to-trial noise floor somewhere in this grid. Fallback NOT triggered.')
else:
    print('\nVERDICT: PeelSketch does not beat HeavyKeeper anywhere beyond noise in this grid. Fallback trigger criterion met.')
"
```

(Every Python string literal above uses single quotes deliberately — the
whole script is one double-quoted shell argument, so any literal `"`
character inside it would need backslash-escaping to survive the shell.
Sticking to single quotes throughout avoids that entirely.)
Expected: prints one row per (skew, budget) combination with mean F1 for
both algorithms, the margin, the combined noise floor, and a per-row
WIN/no verdict, followed by an overall VERDICT line. Capture this full
output — it goes into the results document in Step 5, verbatim, not
paraphrased.

- [ ] **Step 4: Plot the comparison sweep**

Run: `python3 scripts/plot.py results/phase2_comparison.csv --out charts/`
Expected: prints `wrote 3 charts to charts/`, and `accuracy_vs_memory.png`,
`throughput_vs_memory.png`, and `skew_sensitivity.png` all exist and are
non-zero-size, each now showing four algorithm curves instead of three —
confirming `scripts/plot.py` needed no changes to plot the new algorithm.

- [ ] **Step 5: Write the results document**

Create `docs/superpowers/specs/2026-08-16-peelsketch-phase2-results.md`
using this structure, filling in the real output captured in Step 3 (do not
invent numbers — paste the actual command output):

````markdown
# PeelSketch Phase 2 — Comparison Results

## Command run

```
cargo run --release --bin bench -- sweep --cardinality 100000 --stream-length 2000000 --skew 0.8,1.2 --memory-budgets 4096,65536,1048576 --top-k 20 --trials 3 --seed 42 --output results/phase2_comparison.csv
```

## Fallback-trigger comparison (PeelSketch vs. HeavyKeeper, mean F1 by skew/budget)

[Paste the full table + VERDICT output captured in Task 3 Step 3, verbatim.]

## Verdict

[State plainly, per the design spec's Fallback Plan: did PeelSketch beat
HeavyKeeper outside the trial-to-trial noise floor anywhere in this grid?
If yes, the ID-free design is validated for this grid and the fallback is
not triggered — note where the win occurred and by how much. If no, the
fallback trigger criterion (as defined in
`docs/superpowers/specs/2026-08-15-peelsketch-phase2-design.md`'s Fallback
Plan) has been met, and the auxiliary-structure fallback sketched there is
the next step, not further iteration on the ID-free mechanism.]

## Charts

Produced via `python3 scripts/plot.py results/phase2_comparison.csv --out charts/`:
`accuracy_vs_memory.png`, `throughput_vs_memory.png`, `skew_sensitivity.png`
(gitignored, not committed — regenerate from the CSV above if needed).
````

- [ ] **Step 6: Commit the results document**

```bash
git add docs/superpowers/specs/2026-08-16-peelsketch-phase2-results.md
git commit -m "Add PeelSketch Phase 2 comparison results and fallback-trigger verdict"
```

(`results/phase2_comparison.csv` and `charts/*.png` stay gitignored per
Phase 1's `.gitignore` — only the results document itself is committed, as
the durable record of the experiment's outcome.)
