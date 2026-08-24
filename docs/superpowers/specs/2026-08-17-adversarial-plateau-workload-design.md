# Adversarial Plateau Workload — Design

## Context

Phase 1 built a trustworthy benchmark harness measuring memory/accuracy/
throughput tradeoffs for four heavy-hitter sketches (Count-Min, Space-
Saving, HeavyKeeper, PeelSketch) against synthetic Zipfian streams. Phase 2
validated a novel candidate sketch (PeelSketch) against that harness; the
result was a genuine negative, diagnosed as a cell-width tax rather than a
flaw in the core purity-detection idea.

This is the first of two follow-on sub-projects (the second, real network
trace validation, is a separate spec). It adds a second synthetic workload
generator — deliberately different in shape from Zipfian — designed to
stress a specific scenario the existing harness rarely exercises: real
contention *between* multiple heavy items, not just noise around one
dominant item. Zipfian's smooth power-law decay almost never puts two very
high-frequency items in genuine competition; the #1 item is nearly always
far ahead of #2. This is precisely the collision class HeavyKeeper's own
weakness (and PeelSketch's whole motivating premise, per the Hidden Sketch
research reviewed during Phase 2 design) targeted: collisions between
co-heavy items, not between a heavy item and background noise.

The generator must remain **algorithm-agnostic** — it cannot target any
specific sketch's hash function, because the harness's core methodology
(a Global Constraint since Phase 1) generates each stream once per
`(cardinality, skew)` and reuses the identical `Vec<u64>` across all
algorithms. An algorithm-targeted adversarial stream would break that
shared-stream guarantee, which is what keeps the cross-algorithm comparison
fair.

## Goals / Success Criteria

- A new `PlateauGenerator` in the `workload` crate, producing streams with
  a deliberate cluster of `num_heavy` near-tied heavy items (not a smooth
  Zipfian decay), fully algorithm-agnostic.
- Integrated into the existing `bench sweep` command via a `--workload`
  flag (default `zipfian`, unchanged behavior), not a new subcommand —
  reuses the existing sweep/report/plot pipeline rather than duplicating
  it.
- `scripts/plot.py` gains a `--workload` filter so a CSV mixing both
  workload types cannot silently blend into one misleading chart.
- Full reproducibility: every parameter that shapes a plateau stream is
  recorded per-row in `SweepResult`, matching the existing convention for
  `top_k_target`/`warmup_fraction`.

## Non-Goals (this phase)

- Algorithm-targeted (hash-function-specific) adversarial construction —
  explicitly rejected during design; incompatible with the shared-stream
  methodology.
- Combinatorial/provable worst-case guarantees against arbitrary hash
  families — real streaming-theory territory, over-engineered for a
  benchmark harness whose audience wants practical accuracy tradeoffs.
  Approach A (statistical plateau) is a probabilistic stress case, not a
  proof.
- Distribution-shift / non-stationary streams (heavy items changing
  identity mid-stream) — a different, complementary notion of
  "adversarial," out of scope here.
- Sweeping `num_heavy`/`heavy_jitter`/`heavy_mass_fraction` as
  comma-separated cross-product dimensions — they start as single CLI
  values with defaults; extending to sweepable lists is easy later if
  actually needed, not built preemptively.
- Conditional CLI validation between workload modes (e.g. rejecting
  `--skew` when `--workload plateau`) — `--skew` stays required always and
  is simply unused for plateau runs. A deliberate simplification, not an
  oversight.
- Real network trace validation — separate spec, separate phase.

## Architecture

```
crates/
├── workload/
│   └── src/
│       ├── plateau.rs           # NEW: PlateauGenerator
│       └── lib.rs               # MODIFIED: pub mod plateau;
└── bench/
    └── src/
        ├── cli.rs                # MODIFIED: --workload, --num-heavy,
        │                         #   --heavy-jitter, --heavy-mass-fraction
        ├── report.rs              # MODIFIED: SweepResult +4 fields
        └── sweep.rs               # MODIFIED: branch stream gen on workload
scripts/plot.py                   # MODIFIED: --workload filter argument
```

Unlike adding a new *algorithm* (which the Phase 1/2 designs kept to one
file + one enum variant, touching nothing else), adding a new *workload*
touches the CLI, the report schema, and the plotting script. That's
expected, not a design smell — the workload axis and the algorithm axis
are different kinds of extension points, and only the algorithm axis was
ever promised to be a one-file change.

## Generator Design

```rust
// crates/workload/src/plateau.rs
pub struct PlateauGenerator {
    cardinality: u64,
    num_heavy: u64,
    heavy_jitter: f64,
    heavy_mass_fraction: f64,
    seed: u64,
}

impl PlateauGenerator {
    pub fn new(cardinality: u64, num_heavy: u64, heavy_jitter: f64, heavy_mass_fraction: f64, seed: u64) -> Self;
    pub fn generate_ranks(&self, stream_length: usize) -> Vec<u64>;
    pub fn generate(&self, stream_length: usize) -> Vec<u64>;  // generate_ranks() mapped through keys::rank_to_key
}
```

Mirrors `ZipfianGenerator`'s public shape exactly (`generate_ranks` +
`generate`, same `rank_to_key` mapping, same fresh-RNG-seeded-per-call
determinism) so both generators are interchangeable from the sweep loop's
point of view and both produce keys in the same semantic space.

**Implementation note (settled during planning):** unlike
`ZipfianGenerator`, which genuinely needs `rand_distr::Zipf`'s distribution
machinery, `PlateauGenerator` only needs a coin flip, a uniform integer
pick, and a small weighted choice among `num_heavy` items — all cheaply
hand-rollable. It uses the project's existing `sketches::hash::SplitMix64`
(already proven in `HeavyKeeper`'s decay logic, and already a transitive
dependency of `workload` via `sketches`) rather than `rand`/`StdRng`. This
avoids depending on `rand` 0.10's exact `Rng` trait method surface and
keeps this generator's randomness as auditable as everything else hashing-
or PRNG-related in this project.

**Parameters:**
- `num_heavy` (H): count of near-tied heavy items, occupying ranks
  `1..=H`. Must be at least 1 (settled during planning — a "plateau" of
  zero heavy items with nonzero `heavy_mass_fraction` is the same kind of
  contradiction as the tail case below, and is rejected the same way, via
  a constructor panic). Must also satisfy `num_heavy < cardinality`,
  **unless**
  `heavy_mass_fraction == 1.0`, in which case `num_heavy == cardinality`
  is allowed (no tail exists, but none is needed either, since all mass
  goes to the heavy pool). The strict-less-than requirement otherwise
  exists because `num_heavy == cardinality` with `heavy_mass_fraction <
  1.0` is a contradiction — the tail-pool branch would need to sample
  from zero remaining ranks. Violating either condition panics via
  `assert!` in the constructor, with a message naming the violated
  parameter — validated eagerly at construction time rather than lazily
  in `generate_ranks`, unlike `ZipfianGenerator::generate_ranks`'s own
  `.expect("valid zipf parameters...")` guard (which validates at
  generation time since it delegates to `rand_distr::Zipf::new`'s own
  `Result`).
- `heavy_jitter`: fractional variation around an equal split among the H
  heavy items, in `[0.0, 1.0)`. `0.0` means perfectly tied; values
  approaching `1.0` mean wide variation. Deliberately non-zero by default
  — exactly-tied frequencies would mask real tie-breaking differences
  between algorithms (e.g. Space-Saving's deterministic tie handling).
  Validated to stay in `[0.0, 1.0)` — at `jitter == 1.0` an unlucky draw
  could push a heavy item's weight to exactly zero or below, which must
  not happen silently.
- `heavy_mass_fraction`: fraction of total stream mass concentrated in the
  H heavy items collectively, in `(0.0, 1.0]`. The remaining mass spreads
  uniformly across the `cardinality - num_heavy` tail items.

**Generation (per call to `generate_ranks`, two phases):**
1. Seed a fresh `SplitMix64` from `self.seed` (matching `ZipfianGenerator`'s
   own stateless, recompute-per-call pattern — no jitter state is stored
   on the struct). Draw the H heavy items' jitter values `u_1..u_H`
   (uniform `[-1.0, 1.0]`) from this RNG *first*, before generating any
   stream samples, giving weight `1.0 + heavy_jitter * u_i` per heavy
   item. These weights stay fixed for the rest of this call, so the H
   items have stable, distinct-but-close weights across the whole
   stream — but a fresh call to `generate_ranks` re-draws them
   identically from the same seed, preserving determinism.
2. For each of `stream_length` samples, drawn from the same RNG: a coin
   flip (probability `heavy_mass_fraction`) selects heavy-pool vs.
   tail-pool. If heavy-pool: a weighted choice among just the H heavy
   items using the weights from step 1. If tail-pool: a uniform pick
   among the `cardinality - num_heavy` tail ranks.

This is `O(1)` per sample for the coin flip and tail case, `O(H)` for the
heavy-pool weighted choice — no `O(cardinality)` alias table needed, so
generation stays fast even for the multi-million-item streams this harness
already exercises, keeping with the existing "generation timed separately
from insertion" convention.

## CLI and Harness Integration

**New CLI surface** (`crates/bench/src/cli.rs`, on the existing
`SweepArgs`):

```rust
#[derive(clap::ValueEnum, Debug, Clone, Copy, PartialEq, Eq)]
pub enum Workload {
    Zipfian,
    Plateau,
}
```

- `--workload zipfian|plateau` (default `zipfian`) — existing Zipfian
  behavior is unchanged when omitted.
- `--num-heavy` (default `20`), `--heavy-jitter` (default `0.1`),
  `--heavy-mass-fraction` (default `0.8`) — single values, ignored by
  Zipfian runs.
- `--skew` remains `required = true` unconditionally; it is simply unused
  when `--workload plateau` is given. No conditional-required validation
  is added (see Non-Goals).

**`SweepResult` gains four columns** (`crates/bench/src/report.rs`):
`workload: String`, `num_heavy: u64`, `heavy_jitter: f64`,
`heavy_mass_fraction: f64`. Populated meaningfully for plateau rows,
present-but-inert for Zipfian rows — every row is fully self-describing
from the CSV alone, matching how `top_k_target`/`warmup_fraction` are
already recorded per-row regardless of whether they vary within a given
sweep.

**`sweep.rs`** branches stream generation on `args.workload` where it
currently always constructs a `ZipfianGenerator`. Everything downstream —
ground truth computation, the per-algorithm/budget/trial loop, metrics —
is unchanged and already workload-agnostic.

**`scripts/plot.py`** gains a `--workload` argument. If the loaded CSV
contains rows from more than one workload value and `--workload` was not
given, the script must exit with a clear error rather than silently
plotting a blended, misleading comparison — this is the actual safety
property motivating the flag, not just an optional filter. If `--workload`
is given, or the CSV contains only one workload value, plotting proceeds
as today.

## Testing Strategy

**`workload` crate (`plateau.rs`):**
1. Deterministic reproducibility — same seed → identical stream (mirrors
   `ZipfianGenerator`'s `same_seed_produces_identical_stream`).
2. Plateau-shape smoke test — via `ExactCounter` ground truth, the top
   `num_heavy` observed items must be exactly ranks `1..=num_heavy`, with
   counts within a reasonable band of each other (loose statistical
   check, matching the style of `higher_skew_concentrates_more_mass_on_top_rank`).
3. `heavy_mass_fraction` respected — sum of heavy items' true counts /
   stream length approximates the configured fraction within statistical
   tolerance for a large stream.
4. `num_heavy > cardinality` panics with a clear message
   (`#[should_panic(expected = "...")]`).
5. `num_heavy == cardinality` with `heavy_mass_fraction < 1.0` panics
   with a clear message (the tail-needed-but-empty contradiction) —
   and, conversely, `num_heavy == cardinality` with
   `heavy_mass_fraction == 1.0` succeeds (the explicit no-tail escape
   hatch).
6. `heavy_jitter` boundary validation — a value at or above `1.0` (or
   negative) must be rejected, not silently produce a non-positive weight.

**`bench` crate:**
7. CLI parsing test for the four new flags, including verifying defaults
   apply when omitted — extends `cli.rs`'s existing test module.
8. **Update** (not addition) to `report.rs`'s existing CSV/JSON roundtrip
   test fixture (`sample_row()`) to include the four new `SweepResult`
   fields, or the crate fails to compile.
9. A new, separate sweep integration test for `--workload plateau`
   (not folded into the existing Zipfian-focused tiny-sweep test) with
   its own fixture and the same style of sanity-range checks.

**`scripts/plot.py`:**
10. Extend the existing fixture-CSV test to cover both workload types, and
   add the specific failure case the `--workload` flag exists to prevent:
   a CSV containing both workload values with no `--workload` given must
   fail with a clear error. This is the property that actually matters,
   not just "the flag parses."

## Verification Plan

1. `cargo test --workspace` — unit/property tests above, plus the full
   pre-existing Phase 1/2 suite unaffected.
2. `cargo build --workspace --release` — required before any timing
   numbers are trusted, per the existing project-wide constraint.
3. Run a small plateau sweep via the existing CLI (`--workload plateau`)
   alongside a Zipfian sweep, confirm both produce sane, distinct output.
4. `python3 scripts/plot.py` against a CSV containing only one workload
   (no `--workload` flag needed) and against a mixed CSV (`--workload`
   required, confirm the no-flag case errors clearly).
