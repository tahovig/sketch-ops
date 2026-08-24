# Adversarial Plateau Workload Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a second synthetic workload generator — a "plateau" of near-tied
heavy items, deliberately different in shape from Zipfian — that stresses
real contention between co-heavy items, and wire it into the existing
`bench sweep` CLI and `scripts/plot.py` alongside Zipfian.

**Architecture:** One new file, `crates/workload/src/plateau.rs`, mirroring
`ZipfianGenerator`'s public shape exactly. A `--workload zipfian|plateau`
flag (default `zipfian`, unchanged behavior) on the existing `sweep`
subcommand selects between them; `SweepResult` gains four columns recording
the plateau parameters for full reproducibility. `scripts/plot.py` gains a
`--workload` filter so a CSV mixing both workload types cannot silently
blend into one misleading chart.

**Tech Stack:** Rust (edition 2021, matching the existing workspace); no new
dependencies. `PlateauGenerator` uses `sketches::hash::SplitMix64` (already
proven in `HeavyKeeper`'s decay logic and already a transitive dependency of
`workload` via `sketches`) rather than `rand`/`StdRng` — the building blocks
needed here (a coin flip, a uniform integer pick, a small weighted choice
among a handful of items) don't need `rand_distr`'s distribution machinery
the way Zipfian genuinely does, and reusing the project's own hand-rolled
PRNG avoids any dependency on `rand` 0.10's exact `Rng` trait method surface.

## Global Constraints

- Item type stays the existing non-generic `u64`; no sketch code changes in
  this plan.
- `PlateauGenerator` mirrors `ZipfianGenerator`'s public API exactly:
  `generate_ranks(&self, stream_length: usize) -> Vec<u64>` and
  `generate(&self, stream_length: usize) -> Vec<u64>` (the latter mapping
  ranks through the existing `keys::rank_to_key`), so both generators are
  interchangeable from `sweep.rs`'s point of view.
- Determinism: `generate_ranks` seeds a fresh `SplitMix64` from `self.seed`
  on every call (no stored RNG state on the struct) — calling it twice with
  the same seed must produce identical output, matching every other
  generator in this codebase.
- `num_heavy` must satisfy `num_heavy < cardinality`, unless
  `heavy_mass_fraction == 1.0` (in which case `num_heavy == cardinality` is
  allowed — no tail exists, but none is needed). `num_heavy` must also be
  `>= 1`. `heavy_jitter` must be in `[0.0, 1.0)`. `heavy_mass_fraction` must
  be in `(0.0, 1.0]`. All four are validated in the constructor via
  `assert!` with a message naming the violated parameter, and panicking is
  the correct behavior for a misconfigured generator (matching
  `ZipfianGenerator`'s own `.expect("valid zipf parameters...")` pattern).
- `--skew` stays `required = true` on the CLI unconditionally, even though
  it is unused when `--workload plateau` is given — no conditional-required
  validation is added. This is a deliberate simplification, not an
  oversight.
- `SweepResult`'s four new fields (`workload`, `num_heavy`, `heavy_jitter`,
  `heavy_mass_fraction`) are appended at the end of the struct, not
  interspersed — least disruptive to existing CSV column ordering.
- Because `SweepResult` gains required (non-`Option`, non-`Default`)
  fields, every struct literal that constructs one — currently only
  `crates/bench/src/sweep.rs`'s `build_row` — must be updated in the same
  task as the schema change, or the workspace does not compile. This is
  why the CLI, report-schema, and sweep-wiring changes are one task below,
  not three.
- `cargo build --workspace --release` is required before any timing/
  throughput numbers are trusted, per the existing project-wide constraint
  (unchanged, not touched by this plan).

---

### Task 1: `PlateauGenerator`

**Files:**
- Create: `crates/workload/src/plateau.rs`
- Modify: `crates/workload/src/lib.rs`
- Test: inline `#[cfg(test)] mod tests` in `crates/workload/src/plateau.rs`

**Interfaces:**
- Consumes: `sketches::hash::SplitMix64` (`SplitMix64::new(seed: u64) -> Self`, `.next_u64(&mut self) -> u64`, `.next_f64(&mut self) -> f64` — returns a value in `[0.0, 1.0)`); `crate::keys::rank_to_key(rank: u64) -> u64`; `sketches::exact::ExactCounter` (test-only).
- Produces: `pub struct PlateauGenerator`; `pub fn PlateauGenerator::new(cardinality: u64, num_heavy: u64, heavy_jitter: f64, heavy_mass_fraction: f64, seed: u64) -> Self`; `pub fn generate_ranks(&self, stream_length: usize) -> Vec<u64>`; `pub fn generate(&self, stream_length: usize) -> Vec<u64>`.

- [ ] **Step 1: Write the failing tests**

```rust
// crates/workload/src/plateau.rs
use sketches::hash::SplitMix64;

use crate::keys::rank_to_key;

pub struct PlateauGenerator {
    cardinality: u64,
    num_heavy: u64,
    heavy_jitter: f64,
    heavy_mass_fraction: f64,
    seed: u64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use sketches::exact::ExactCounter;
    use std::collections::HashSet;

    #[test]
    fn same_seed_produces_identical_stream() {
        let gen_a = PlateauGenerator::new(1000, 10, 0.1, 0.8, 7);
        let gen_b = PlateauGenerator::new(1000, 10, 0.1, 0.8, 7);
        let stream_a = gen_a.generate(500);
        let stream_b = gen_b.generate(500);
        assert_eq!(stream_a, stream_b);
    }

    #[test]
    fn top_heavy_items_dominate_and_stay_within_a_band() {
        let cardinality = 1000u64;
        let num_heavy = 10u64;
        let stream_length = 50_000usize;
        let gen = PlateauGenerator::new(cardinality, num_heavy, 0.1, 0.8, 99);
        let stream = gen.generate(stream_length);

        let mut exact = ExactCounter::new();
        for &item in &stream {
            exact.insert(item);
        }

        let expected_heavy_keys: HashSet<u64> = (1..=num_heavy).map(rank_to_key).collect();

        let top = exact.top_k(num_heavy as usize);
        assert_eq!(top.len(), num_heavy as usize, "expected exactly num_heavy items in top_k");

        let observed_keys: HashSet<u64> = top.iter().map(|&(k, _)| k).collect();
        assert_eq!(
            observed_keys, expected_heavy_keys,
            "the top num_heavy observed items must be exactly the intended heavy ranks"
        );

        let counts: Vec<u64> = top.iter().map(|&(_, c)| c).collect();
        let max_count = *counts.iter().max().unwrap();
        let min_count = *counts.iter().min().unwrap();
        assert!(
            (max_count as f64) < (min_count as f64) * 1.5,
            "heavy items should be near-tied, not dominated by one: max={max_count} min={min_count}"
        );
    }

    #[test]
    fn heavy_mass_fraction_is_approximately_respected() {
        let cardinality = 1000u64;
        let num_heavy = 10u64;
        let heavy_mass_fraction = 0.8;
        let stream_length = 50_000usize;
        let gen = PlateauGenerator::new(cardinality, num_heavy, 0.1, heavy_mass_fraction, 55);
        let stream = gen.generate(stream_length);

        let mut exact = ExactCounter::new();
        for &item in &stream {
            exact.insert(item);
        }

        let heavy_total: u64 = (1..=num_heavy).map(rank_to_key).map(|k| exact.query(k)).sum();
        let observed_fraction = heavy_total as f64 / stream_length as f64;

        assert!(
            (observed_fraction - heavy_mass_fraction).abs() < 0.05,
            "observed heavy mass fraction {observed_fraction} should be close to configured {heavy_mass_fraction}"
        );
    }

    #[test]
    #[should_panic(expected = "num_heavy")]
    fn num_heavy_greater_than_cardinality_panics() {
        PlateauGenerator::new(100, 200, 0.1, 0.8, 1);
    }

    #[test]
    #[should_panic(expected = "num_heavy")]
    fn num_heavy_equals_cardinality_with_partial_mass_panics() {
        PlateauGenerator::new(100, 100, 0.1, 0.8, 1);
    }

    #[test]
    fn num_heavy_equals_cardinality_with_full_mass_succeeds() {
        // No tail exists, but none is needed since all mass goes to the heavy pool.
        let gen = PlateauGenerator::new(100, 100, 0.1, 1.0, 1);
        let stream = gen.generate(1000);
        assert_eq!(stream.len(), 1000);
    }

    #[test]
    #[should_panic(expected = "num_heavy")]
    fn num_heavy_zero_panics() {
        PlateauGenerator::new(100, 0, 0.1, 0.8, 1);
    }

    #[test]
    #[should_panic(expected = "heavy_jitter")]
    fn heavy_jitter_at_one_panics() {
        PlateauGenerator::new(1000, 10, 1.0, 0.8, 1);
    }

    #[test]
    #[should_panic(expected = "heavy_jitter")]
    fn negative_heavy_jitter_panics() {
        PlateauGenerator::new(1000, 10, -0.1, 0.8, 1);
    }
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test -p workload plateau`
Expected: FAIL with a compile error — `PlateauGenerator::new`/`generate`
don't exist yet.

- [ ] **Step 3: Write the implementation**

```rust
// crates/workload/src/plateau.rs (add above the tests module)
impl PlateauGenerator {
    pub fn new(cardinality: u64, num_heavy: u64, heavy_jitter: f64, heavy_mass_fraction: f64, seed: u64) -> Self {
        assert!(num_heavy >= 1, "num_heavy must be at least 1, got {num_heavy}");
        assert!(
            num_heavy < cardinality || (num_heavy == cardinality && heavy_mass_fraction == 1.0),
            "num_heavy ({num_heavy}) must be < cardinality ({cardinality}), unless heavy_mass_fraction == 1.0 (no tail needed)"
        );
        assert!(
            (0.0..1.0).contains(&heavy_jitter),
            "heavy_jitter ({heavy_jitter}) must be in [0.0, 1.0)"
        );
        assert!(
            heavy_mass_fraction > 0.0 && heavy_mass_fraction <= 1.0,
            "heavy_mass_fraction ({heavy_mass_fraction}) must be in (0.0, 1.0]"
        );
        Self { cardinality, num_heavy, heavy_jitter, heavy_mass_fraction, seed }
    }

    pub fn generate_ranks(&self, stream_length: usize) -> Vec<u64> {
        let mut rng = SplitMix64::new(self.seed);

        // Draw jitter weights for the H heavy items first, fixed for the
        // rest of this call. A fresh call re-draws them identically from
        // the same seed, preserving determinism.
        let heavy_weights: Vec<f64> = (0..self.num_heavy)
            .map(|_| {
                let u = rng.next_f64() * 2.0 - 1.0; // uniform in [-1.0, 1.0)
                1.0 + self.heavy_jitter * u
            })
            .collect();
        let heavy_weight_sum: f64 = heavy_weights.iter().sum();

        let tail_count = self.cardinality - self.num_heavy;

        (0..stream_length)
            .map(|_| {
                if rng.next_f64() < self.heavy_mass_fraction {
                    // Heavy pool: weighted choice among the H heavy items
                    // (ranks 1..=num_heavy).
                    let mut draw = rng.next_f64() * heavy_weight_sum;
                    let mut chosen_rank = self.num_heavy; // fallback for float-rounding at the boundary
                    for (i, &w) in heavy_weights.iter().enumerate() {
                        if draw < w {
                            chosen_rank = (i as u64) + 1;
                            break;
                        }
                        draw -= w;
                    }
                    chosen_rank
                } else {
                    // Tail pool: uniform pick among the remaining ranks.
                    // Only reachable when tail_count > 0 — next_f64() is
                    // strictly < 1.0, so when heavy_mass_fraction == 1.0
                    // (the only case where tail_count may be 0) this branch
                    // is never taken.
                    let tail_offset = rng.next_u64() % tail_count;
                    self.num_heavy + 1 + tail_offset
                }
            })
            .collect()
    }

    pub fn generate(&self, stream_length: usize) -> Vec<u64> {
        self.generate_ranks(stream_length).into_iter().map(rank_to_key).collect()
    }
}
```

```rust
// crates/workload/src/lib.rs
//! Zipfian and plateau stream generation and ground-truth computation.
pub mod ground_truth;
pub mod keys;
pub mod plateau;
pub mod zipfian;
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test -p workload plateau`
Expected: PASS — all 9 tests (determinism, plateau shape, mass fraction,
four panic cases, and the two equals-cardinality edge cases).

- [ ] **Step 5: Run the full workload test suite to check for regressions**

Run: `cargo test -p workload`
Expected: PASS — the pre-existing `zipfian`/`ground_truth` tests remain
green.

- [ ] **Step 6: Commit**

```bash
git add crates/workload/src/plateau.rs crates/workload/src/lib.rs
git commit -m "Add PlateauGenerator: near-tied heavy-item workload generator"
```

---

### Task 2: CLI flags, `SweepResult` schema, and sweep wiring

**Files:**
- Modify: `crates/bench/src/cli.rs`
- Modify: `crates/bench/src/report.rs`
- Modify: `crates/bench/src/sweep.rs`
- Test: extends existing `#[cfg(test)] mod tests` in all three files

**Interfaces:**
- Consumes: `workload::plateau::PlateauGenerator::new(cardinality: u64, num_heavy: u64, heavy_jitter: f64, heavy_mass_fraction: f64, seed: u64) -> Self` and `.generate(stream_length: usize) -> Vec<u64>` (Task 1).
- Produces: `pub enum Workload { Zipfian, Plateau }` with `pub fn name(&self) -> &'static str`; four new `SweepArgs` fields (`workload`, `num_heavy`, `heavy_jitter`, `heavy_mass_fraction`); four new `SweepResult` fields (same names).

This task touches three files together because `SweepResult` gaining
required fields breaks `sweep.rs`'s existing struct literal — see Global
Constraints above.

- [ ] **Step 1: Write the failing tests**

```rust
// crates/bench/src/cli.rs — add this enum above SweepArgs
#[derive(clap::ValueEnum, Debug, Clone, Copy, PartialEq, Eq)]
pub enum Workload {
    Zipfian,
    Plateau,
}
```

Modify the existing test in `crates/bench/src/cli.rs`'s
`#[cfg(test)] mod tests`, adding four assertions to the end of
`parses_sweep_args_with_comma_separated_lists`:

```rust
// crates/bench/src/cli.rs — append inside parses_sweep_args_with_comma_separated_lists,
// after the existing assert_eq!(args.format, "csv");
assert_eq!(args.workload, Workload::Zipfian, "workload should default to zipfian when omitted");
assert_eq!(args.num_heavy, 20, "num_heavy should default to 20 when omitted");
assert_eq!(args.heavy_jitter, 0.1, "heavy_jitter should default to 0.1 when omitted");
assert_eq!(args.heavy_mass_fraction, 0.8, "heavy_mass_fraction should default to 0.8 when omitted");
```

Add a new test in the same module:

```rust
// crates/bench/src/cli.rs — new test in mod tests
#[test]
fn parses_plateau_workload_flags() {
    let cli = Cli::parse_from([
        "bench",
        "sweep",
        "--cardinality", "1000",
        "--stream-length", "50000",
        "--skew", "1.0",
        "--memory-budgets", "4096",
        "--workload", "plateau",
        "--num-heavy", "15",
        "--heavy-jitter", "0.2",
        "--heavy-mass-fraction", "0.9",
        "--output", "results/test.csv",
    ]);

    let args = match cli.command {
        Command::Sweep(args) => args,
    };

    assert_eq!(args.workload, Workload::Plateau);
    assert_eq!(args.num_heavy, 15);
    assert_eq!(args.heavy_jitter, 0.2);
    assert_eq!(args.heavy_mass_fraction, 0.9);
}
```

Modify `report.rs`'s existing `sample_row()` fixture (used by both
roundtrip tests), adding the four new fields:

```rust
// crates/bench/src/report.rs — sample_row(), add these fields at the end
// of the existing struct literal (after overestimate_count: 5,)
workload: "zipfian".to_string(),
num_heavy: 20,
heavy_jitter: 0.1,
heavy_mass_fraction: 0.8,
```

Modify `sweep.rs`'s existing `tiny_args()` fixture, adding the four new
fields:

```rust
// crates/bench/src/sweep.rs — tiny_args(), add these fields at the end
// of the existing struct literal (after format: "csv".to_string(),)
workload: Workload::Zipfian,
num_heavy: 20,
heavy_jitter: 0.1,
heavy_mass_fraction: 0.8,
```

Add a new fixture and test in `sweep.rs`'s `mod tests`:

```rust
// crates/bench/src/sweep.rs — new fixture + test in mod tests
fn tiny_plateau_args() -> SweepArgs {
    SweepArgs {
        cardinality: vec![200],
        stream_length: 5_000,
        skew: vec![1.0],
        memory_budgets: vec![4096, 16384],
        top_k: 10,
        trials: 2,
        seed: 42,
        warmup_fraction: 0.1,
        output: "results/tiny_plateau.csv".to_string(),
        format: "csv".to_string(),
        workload: Workload::Plateau,
        num_heavy: 10,
        heavy_jitter: 0.1,
        heavy_mass_fraction: 0.8,
    }
}

#[test]
fn tiny_plateau_sweep_produces_expected_row_count_and_sane_values() {
    let args = tiny_plateau_args();
    let results = run_sweep(&args);

    let expected_rows =
        args.cardinality.len() * args.skew.len() * args.memory_budgets.len() * 4 * args.trials as usize;
    assert_eq!(results.len(), expected_rows);

    for row in &results {
        assert_eq!(row.workload, "plateau");
        assert_eq!(row.num_heavy, 10);
        assert!(row.items_per_sec > 0.0, "throughput must be positive: {row:?}");
        assert!(row.actual_memory_bytes > 0);
        assert!((0.0..=1.0).contains(&row.precision_at_k));
        assert!((0.0..=1.0).contains(&row.recall_at_k));
        assert!((0.0..=1.0).contains(&row.f1_at_k));
        assert!(row.mean_relative_error >= 0.0);
        assert!(row.max_relative_error >= row.mean_relative_error - 1e-9);
    }

    // Same oracle reasoning as the Zipfian tiny-sweep test: at budget
    // 16384 with top_k 10, Space-Saving's m (507) exceeds this stream's
    // cardinality (200) regardless of which generator produced it, so
    // Space-Saving must be exact here too.
    let exact_space_saving_rows: Vec<_> = results
        .iter()
        .filter(|row| row.algorithm == "space_saving" && row.requested_memory_bytes == 16384)
        .collect();
    assert!(!exact_space_saving_rows.is_empty());
    for row in exact_space_saving_rows {
        assert_eq!(row.mean_relative_error, 0.0, "expected exact tracking: {row:?}");
        assert_eq!(row.f1_at_k, 1.0, "expected perfect top-k recovery: {row:?}");
    }
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test -p bench`
Expected: FAIL with compile errors — `Workload` type used in tests before
`SweepArgs`/`SweepResult` gain the new fields; `PlateauGenerator` unused
in `sweep.rs` yet.

- [ ] **Step 3: Write the implementation**

```rust
// crates/bench/src/cli.rs — replace SweepArgs in full (adds four fields
// at the end; every existing field is unchanged)
#[derive(Args, Debug, Clone)]
pub struct SweepArgs {
    #[arg(long, value_delimiter = ',', required = true)]
    pub cardinality: Vec<u64>,

    #[arg(long)]
    pub stream_length: u64,

    #[arg(long, value_delimiter = ',', required = true)]
    pub skew: Vec<f64>,

    #[arg(long, value_delimiter = ',', required = true)]
    pub memory_budgets: Vec<usize>,

    #[arg(long, default_value_t = 20)]
    pub top_k: usize,

    #[arg(long, default_value_t = 1)]
    pub trials: u32,

    #[arg(long, default_value_t = 42)]
    pub seed: u64,

    #[arg(long, default_value_t = 0.05)]
    pub warmup_fraction: f64,

    #[arg(long, default_value = "results/sweep.csv")]
    pub output: String,

    #[arg(long, default_value = "csv")]
    pub format: String,

    #[arg(long, value_enum, default_value = "zipfian")]
    pub workload: Workload,

    #[arg(long, default_value_t = 20)]
    pub num_heavy: u64,

    #[arg(long, default_value_t = 0.1)]
    pub heavy_jitter: f64,

    #[arg(long, default_value_t = 0.8)]
    pub heavy_mass_fraction: f64,
}

impl Workload {
    pub fn name(&self) -> &'static str {
        match self {
            Workload::Zipfian => "zipfian",
            Workload::Plateau => "plateau",
        }
    }
}
```

(Place the `impl Workload` block anywhere after the `Workload` enum
definition added in Step 1 — e.g. directly below it.)

```rust
// crates/bench/src/report.rs — replace SweepResult in full (adds four
// fields at the end; every existing field is unchanged)
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SweepResult {
    pub run_id: String,
    pub algorithm: String,
    pub trial: u32,
    pub seed: u64,
    pub cardinality: u64,
    pub skew: f64,
    pub stream_length: u64,
    pub top_k_target: usize,
    pub warmup_fraction: f64,
    pub requested_memory_bytes: usize,
    pub actual_memory_bytes: usize,
    pub construction_time_ns: u64,
    pub insert_elapsed_ns: u64,
    pub items_per_sec: f64,
    pub precision_at_k: f64,
    pub recall_at_k: f64,
    pub f1_at_k: f64,
    pub mean_relative_error: f64,
    pub max_relative_error: f64,
    pub underestimate_count: u64,
    pub overestimate_count: u64,
    pub workload: String,
    pub num_heavy: u64,
    pub heavy_jitter: f64,
    pub heavy_mass_fraction: f64,
}
```

```rust
// crates/bench/src/sweep.rs — replace the top-of-file imports in full
// (this is the complete, final import block)
use std::time::Instant;

use sketches::count_min::CountMinSketch;
use sketches::heavy_keeper::HeavyKeeper;
use sketches::peel_sketch::PeelSketch;
use sketches::space_saving::SpaceSaving;
use sketches::traits::HeavyHitterSketch;
use workload::ground_truth::GroundTruth;
use workload::plateau::PlateauGenerator;
use workload::zipfian::ZipfianGenerator;

use crate::cli::{SweepArgs, Workload};
use crate::metrics::{f1_score, precision_at_k, recall_at_k, relative_error_stats};
use crate::report::SweepResult;
```

```rust
// crates/bench/src/sweep.rs — inside run_sweep, replace the stream-generation
// block (currently just the three ZipfianGenerator lines) with:
let shared_seed = stream_seed(args.seed, cardinality, skew);
// `skew` only shapes Zipfian streams; PlateauGenerator ignores it
// entirely (see cli.rs's Workload doc and the design's Non-Goals on
// conditional CLI validation) — it is still looped over here so every
// requested skew value still produces its own row, even though the
// resulting stream is identical across skew values for plateau runs.
let stream = match args.workload {
    Workload::Zipfian => {
        let stream_gen = ZipfianGenerator::new(cardinality, skew, shared_seed);
        stream_gen.generate(args.stream_length as usize)
    }
    Workload::Plateau => {
        let stream_gen = PlateauGenerator::new(
            cardinality,
            args.num_heavy,
            args.heavy_jitter,
            args.heavy_mass_fraction,
            shared_seed,
        );
        stream_gen.generate(args.stream_length as usize)
    }
};
```

```rust
// crates/bench/src/sweep.rs — inside build_row, add these four fields at
// the end of the SweepResult struct literal (after overestimate_count: error_stats.overestimate_count,)
workload: ctx.args.workload.name().to_string(),
num_heavy: ctx.args.num_heavy,
heavy_jitter: ctx.args.heavy_jitter,
heavy_mass_fraction: ctx.args.heavy_mass_fraction,
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test -p bench`
Expected: PASS — all `cli.rs`, `report.rs`, and `sweep.rs` tests, including
the new `parses_plateau_workload_flags` and
`tiny_plateau_sweep_produces_expected_row_count_and_sane_values`.

- [ ] **Step 5: Run the full workspace suite and a manual smoke run**

Run: `cargo test --workspace`
Expected: PASS — every test across `sketches`, `workload`, and `bench`.

Run: `cargo run -p bench -- sweep --cardinality 500 --stream-length 20000 --skew 1.0 --memory-budgets 8192 --workload plateau --num-heavy 8 --top-k 10 --trials 1 --seed 1 --output /tmp/plateau_smoke.csv`
Expected: prints `wrote 4 rows to /tmp/plateau_smoke.csv` with no panic.

- [ ] **Step 6: Commit**

```bash
git add crates/bench/src/cli.rs crates/bench/src/report.rs crates/bench/src/sweep.rs
git commit -m "Wire the plateau workload into the sweep CLI and result schema"
```

---

### Task 3: `scripts/plot.py` `--workload` filter

**Files:**
- Modify: `scripts/plot.py`
- Test: manual runs against hand-crafted fixture CSVs (matching the existing project convention — no pytest harness)

**Interfaces:**
- Consumes: a CSV whose header includes `SweepResult`'s full field set (Task 2), including the new `workload`/`num_heavy`/`heavy_jitter`/`heavy_mass_fraction` columns.
- Produces: a new `--workload` CLI argument on `scripts/plot.py`; a new `filter_workload(df, workload, csv_path) -> pd.DataFrame` function.

- [ ] **Step 1: Create the hand-crafted fixture CSVs**

Create `/tmp/sketch_ops_plot_fixture_single.csv` (all rows `workload=zipfian`):

```csv
run_id,algorithm,trial,seed,cardinality,skew,stream_length,top_k_target,warmup_fraction,requested_memory_bytes,actual_memory_bytes,construction_time_ns,insert_elapsed_ns,items_per_sec,precision_at_k,recall_at_k,f1_at_k,mean_relative_error,max_relative_error,underestimate_count,overestimate_count,workload,num_heavy,heavy_jitter,heavy_mass_fraction
count_min-card1000-skew0.8-mem4096-trial0,count_min,0,42,1000,0.8,100000,10,0.05,4096,4088,900,80000000,1187500.0,0.9,0.9,0.9,0.05,0.2,1,1,zipfian,20,0.1,0.8
heavy_keeper-card1000-skew0.8-mem4096-trial0,heavy_keeper,0,42,1000,0.8,100000,10,0.05,8192,8176,1100,85000000,1117647.0,0.85,0.85,0.85,0.1,0.3,1,1,zipfian,20,0.1,0.8
```

Create `/tmp/sketch_ops_plot_fixture_mixed.csv` (the same two `zipfian` rows,
plus two `plateau` rows):

```csv
run_id,algorithm,trial,seed,cardinality,skew,stream_length,top_k_target,warmup_fraction,requested_memory_bytes,actual_memory_bytes,construction_time_ns,insert_elapsed_ns,items_per_sec,precision_at_k,recall_at_k,f1_at_k,mean_relative_error,max_relative_error,underestimate_count,overestimate_count,workload,num_heavy,heavy_jitter,heavy_mass_fraction
count_min-card1000-skew0.8-mem4096-trial0,count_min,0,42,1000,0.8,100000,10,0.05,4096,4088,900,80000000,1187500.0,0.9,0.9,0.9,0.05,0.2,1,1,zipfian,20,0.1,0.8
heavy_keeper-card1000-skew0.8-mem4096-trial0,heavy_keeper,0,42,1000,0.8,100000,10,0.05,8192,8176,1100,85000000,1117647.0,0.85,0.85,0.85,0.1,0.3,1,1,zipfian,20,0.1,0.8
count_min-card1000-skew1.0-mem4096-trial0,count_min,0,42,1000,1.0,100000,10,0.05,4096,4088,900,78000000,1217948.0,0.8,0.8,0.8,0.15,0.4,2,2,plateau,10,0.1,0.8
heavy_keeper-card1000-skew1.0-mem4096-trial0,heavy_keeper,0,42,1000,1.0,100000,10,0.05,8192,8176,1100,82000000,1158536.0,0.78,0.78,0.78,0.18,0.45,2,2,plateau,10,0.1,0.8
```

- [ ] **Step 2: Run the script against both fixtures to observe current (pre-fix) behavior**

Run: `python3 scripts/plot.py /tmp/sketch_ops_plot_fixture_single.csv --out /tmp/plot_charts_single`
Expected: FAILS — `validate_columns` rejects the fixture because
`REQUIRED_COLUMNS` doesn't yet know about the four new columns... actually
the fixture *has* them, so this currently succeeds. The real pre-fix gap is
`--workload` doesn't exist yet:

Run: `python3 scripts/plot.py /tmp/sketch_ops_plot_fixture_mixed.csv --workload plateau --out /tmp/plot_charts_mixed`
Expected: FAILS — `argparse` rejects the unrecognized `--workload` argument.

- [ ] **Step 3: Write the implementation**

```python
# scripts/plot.py — replace REQUIRED_COLUMNS in full
REQUIRED_COLUMNS = {
    "run_id", "algorithm", "trial", "seed", "cardinality", "skew",
    "stream_length", "top_k_target", "warmup_fraction", "requested_memory_bytes",
    "actual_memory_bytes", "construction_time_ns", "insert_elapsed_ns",
    "items_per_sec", "precision_at_k", "recall_at_k", "f1_at_k",
    "mean_relative_error", "max_relative_error", "underestimate_count",
    "overestimate_count", "workload", "num_heavy", "heavy_jitter",
    "heavy_mass_fraction"
}
```

```python
# scripts/plot.py — add this function, e.g. directly after validate_columns
def filter_workload(df: pd.DataFrame, workload: str, csv_path: str) -> pd.DataFrame:
    present = sorted(df["workload"].unique())
    if workload is not None:
        filtered = df[df["workload"] == workload]
        if filtered.empty:
            sys.exit(f"error: no rows with workload={workload!r} found in {csv_path} (present: {present})")
        return filtered
    if len(present) > 1:
        sys.exit(
            f"error: {csv_path} contains multiple workload types {present} — "
            "pass --workload to select one, plotting a mix would be misleading"
        )
    return df
```

```python
# scripts/plot.py — replace main() in full
def main() -> None:
    parser = argparse.ArgumentParser(description="Plot heavy-hitters sweep results.")
    parser.add_argument("csv_path", help="Path to a sweep-results CSV file")
    parser.add_argument("--out", default="charts", help="Output directory for PNG charts")
    parser.add_argument(
        "--workload",
        choices=["zipfian", "plateau"],
        default=None,
        help="Filter to one workload type. Required if the CSV contains more than one.",
    )
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    df = load_results(args.csv_path)
    validate_columns(df, args.csv_path)
    df = filter_workload(df, args.workload, args.csv_path)

    plot_accuracy_vs_memory(df, args.out)
    plot_throughput_vs_memory(df, args.out)
    plot_skew_sensitivity(df, args.out)

    print(f"wrote 3 charts to {args.out}/")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the script to verify the fix**

Run: `python3 scripts/plot.py /tmp/sketch_ops_plot_fixture_single.csv --out /tmp/plot_charts_single`
Expected: prints `wrote 3 charts to /tmp/plot_charts_single/` (single
workload present, no `--workload` flag needed).

Run: `python3 scripts/plot.py /tmp/sketch_ops_plot_fixture_mixed.csv --out /tmp/plot_charts_mixed_noflag`
Expected: exits with `error: /tmp/sketch_ops_plot_fixture_mixed.csv
contains multiple workload types ['plateau', 'zipfian'] — pass --workload
to select one, plotting a mix would be misleading` and a non-zero exit
code — no chart files are written.

Run: `python3 scripts/plot.py /tmp/sketch_ops_plot_fixture_mixed.csv --workload plateau --out /tmp/plot_charts_mixed_plateau`
Expected: prints `wrote 3 charts to /tmp/plot_charts_mixed_plateau/`.

Run: `ls -la /tmp/plot_charts_single/ /tmp/plot_charts_mixed_plateau/`
Expected: `accuracy_vs_memory.png`, `throughput_vs_memory.png`,
`skew_sensitivity.png` all exist and are non-zero-size in both directories;
`/tmp/plot_charts_mixed_noflag/` was never created or is empty.

- [ ] **Step 5: Commit**

```bash
git add scripts/plot.py
git commit -m "Add --workload filter to plot.py to prevent blending workload types"
```

---

### Task 4: Final verification

**Files:**
- No source files change in this task — it exercises Tasks 1–3 end to end,
  matching the pattern of prior phases' final verification tasks.

**Interfaces:**
- Consumes: the `bench` binary's `sweep` subcommand (Task 2, with the new
  `--workload` flags); `scripts/plot.py` (Task 3, with the new
  `--workload` filter).

- [ ] **Step 1: Run the full workspace test suite and release build**

Run: `cargo test --workspace`
Expected: PASS — every test from Tasks 1–2 plus the full pre-existing
suite.

Run: `cargo build --workspace --release`
Expected: PASS — release build succeeds with `lto = true` and
`codegen-units = 1` (unchanged from prior phases).

- [ ] **Step 2: Run real plateau and zipfian sweeps**

Run:
```bash
cargo run --release --bin bench -- sweep \
  --cardinality 5000 \
  --stream-length 200000 \
  --skew 1.0 \
  --memory-budgets 8192,65536 \
  --workload plateau \
  --num-heavy 15 \
  --heavy-jitter 0.15 \
  --heavy-mass-fraction 0.85 \
  --top-k 15 \
  --trials 2 \
  --seed 7 \
  --output results/plateau_smoke.csv
```
Expected: prints `wrote 16 rows to results/plateau_smoke.csv` (1
cardinality × 1 skew × 2 budgets × 4 algorithms × 2 trials = 16).

Run:
```bash
cargo run --release --bin bench -- sweep \
  --cardinality 5000 \
  --stream-length 200000 \
  --skew 1.0 \
  --memory-budgets 8192,65536 \
  --workload zipfian \
  --top-k 15 \
  --trials 2 \
  --seed 7 \
  --output results/zipfian_smoke.csv
```
Expected: prints `wrote 16 rows to results/zipfian_smoke.csv`.

- [ ] **Step 3: Build a combined mixed-workload CSV from the two real sweeps**

Run:
```bash
(head -n 1 results/plateau_smoke.csv && tail -n +2 results/plateau_smoke.csv && tail -n +2 results/zipfian_smoke.csv) > results/mixed_smoke.csv
```
Expected: `results/mixed_smoke.csv` has one header line followed by all 32
data rows from both files combined (`wc -l results/mixed_smoke.csv`
reports 33).

- [ ] **Step 4: Verify plot.py's real end-to-end behavior**

Run: `python3 scripts/plot.py results/plateau_smoke.csv --out charts/plateau/`
Expected: prints `wrote 3 charts to charts/plateau/` — confirms the real
`SweepResult` schema produced by `run_sweep` matches what `plot.py`
expects, not just the hand-crafted Task 3 fixtures.

Run: `python3 scripts/plot.py results/mixed_smoke.csv --out charts/mixed_noflag/`
Expected: exits with the multiple-workload-types error, non-zero exit
code — confirmed against real sweep output, not a fixture.

Run: `python3 scripts/plot.py results/mixed_smoke.csv --workload plateau --out charts/mixed_plateau/`
Expected: prints `wrote 3 charts to charts/mixed_plateau/`.

No commit for this task — `results/` and `charts/` are gitignored per
Phase 1's `.gitignore`; this is a verification-only pass with no source
changes.
