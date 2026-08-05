# Heavy-Hitters Sketch Benchmark Suite — Phase 1 Design

## Context

Streaming telemetry systems (network backbones, security firewalls, exchange
trading logs) need to identify heavy hitters — the small fraction of items
(e.g. IP addresses) responsible for most traffic — in a single pass, using
memory that is sub-linear in the size of the stream. Classic sketches
(Count-Min Sketch, Space-Saving) and newer ones (HeavyKeeper) trade off
memory, accuracy, and throughput differently, and at wire-speed scale even a
1% improvement in accuracy per byte of memory is operationally significant.

This project is a two-phase POC. **Phase 1**, specified here, builds a
rigorous benchmark harness and implements three known heavy-hitter sketches
well enough to trust their measured tradeoffs. **Phase 2** (a future, separate
spec) will use this harness to validate a novel algorithmic improvement
against these same baselines. Phase 1's job is to produce a benchmark
trustworthy enough that Phase 2's "our sketch is better" claims are credible.

Scope is heavy-hitters detection only — not cardinality estimation
(HyperLogLog and similar are a separate problem and out of scope).

## Goals / Success Criteria

- Three correctly implemented sketches (Count-Min Sketch, Space-Saving,
  HeavyKeeper) behind one common interface.
- A CLI benchmark harness that sweeps memory budget × Zipfian skew and
  measures accuracy (precision/recall/F1/relative error against exact
  ground truth) and throughput (items/sec), single-threaded.
- Structured (CSV/JSON) output, visualized via a small plotting script, that
  clearly shows each algorithm's tradeoff curve.
- A design that lets Phase 2 add a fourth candidate sketch and benchmark it
  against these baselines with minimal harness changes.

## Non-Goals (Phase 1)

- Cardinality estimation (HyperLogLog).
- Multi-threaded/concurrent ingestion.
- The novel algorithmic improvement itself (Phase 2).
- Real network trace data (synthetic Zipfian only; a real-trace validation
  pass is a possible future addition, not required here).

## Architecture

A Cargo workspace with three crates:

```
sketch-ops/
├── Cargo.toml                  # [workspace], resolver = "2", release profile: lto=true, codegen-units=1
├── crates/
│   ├── sketches/               # lib: pure algorithms, no I/O, no CLI deps
│   │   └── src/
│   │       ├── traits.rs       # HeavyHitterSketch trait
│   │       ├── hash.rs         # hand-rolled multiplicative hash family
│   │       ├── exact.rs        # ExactCounter (ground-truth reference impl)
│   │       ├── count_min.rs
│   │       ├── space_saving.rs
│   │       ├── heavy_keeper.rs
│   │       └── indexed_heap.rs # shared min-heap w/ decrease-key (SS + HK top-k)
│   ├── workload/                # lib: stream generation & ground truth
│   │   └── src/
│   │       ├── zipfian.rs      # seeded Zipfian key-stream generator
│   │       ├── keys.rs         # rank -> synthetic u64 ("IP-like") key mapping
│   │       └── ground_truth.rs # exact frequency table + true top-k
│   └── bench/                   # bin: CLI, sweep orchestration, I/O
│       └── src/
│           ├── main.rs
│           ├── cli.rs           # clap derive
│           ├── sweep.rs         # nested sweep loop
│           ├── metrics.rs       # precision/recall/F1/relative-error
│           └── report.rs        # SweepResult (serde) + CSV/JSON writer
├── scripts/plot.py               # matplotlib: CSV -> charts/*.png
├── results/                      # gitignored
└── charts/                       # gitignored
```

Splitting `sketches`/`workload`/`bench` into separate crates keeps the
algorithms independently unit-testable and gives Phase 2 a clean library to
build a new candidate sketch against without touching CLI/reporting code.
The release profile enables LTO and a single codegen unit — trustworthy
throughput numbers are the point of the project, so the default,
un-tuned release profile isn't good enough.

## Key Dependencies

| Purpose | Crate | Notes |
|---|---|---|
| RNG core | `rand` (0.10) | seeded via `StdRng::seed_from_u64` for reproducibility |
| Zipfian sampling | `rand_distr` (0.6) | `Zipf` struct (finite support) — not `Zeta` (unbounded) |
| CLI | `clap` (4.6, derive) | sweep args as comma-separated lists |
| Serialization | `serde` + `csv` + `serde_json` | one `SweepResult` struct, two writers |
| Property testing | `proptest` (dev-dep) | CMS/Space-Saving invariants |

`criterion` is deliberately **not** used for the sweep harness — it's built
for repeated microbenchmarking of a single hot function with its own report
format, not multi-dimensional parameter sweeps emitting non-timing metrics
(accuracy, memory) into a unified schema. Plain `std::time::Instant` around a
hand-controlled warmup+measurement loop is the right tool for this
steady-state batch measurement. `criterion` could be added later, out of
band, for narrow questions like "cost of the hash function alone," as a
separate `benches/` target — not part of the sweep CLI.

## Common Interface

```rust
pub trait HeavyHitterSketch {
    fn new_with_budget(budget_bytes: usize, k: usize, seed: u64) -> Self where Self: Sized;
    fn insert(&mut self, item: u64);
    fn query(&self, item: u64) -> u64;
    fn top_k(&self, k: usize) -> Vec<(u64, u64)>;
    fn memory_bytes(&self) -> usize;
    fn name(&self) -> &'static str;
}
```

Item type is a non-generic `u64` for Phase 1 (a zero-extended IPv4 or a
hashed key) — this keeps the hot insert loop monomorphizable and all three
implementations simple. Generic string-key support is a documented, deferred
extension.

**Memory-accounting convention (applies to all three algorithms alike):**
`memory_bytes()` returns an analytical/logical byte count — `size_of::<T>() *
len` for the fixed-size arrays each structure allocates — not OS-reported
RSS, which is allocator/page-granularity-dependent and would make
cross-algorithm comparisons noisy. `/proc/self/status` RSS may be recorded as
an optional secondary diagnostic, never as a plotted x-axis.

**Top-k accounting is the single most important correctness detail in this
design.** Only Space-Saving has a top-k structure built into its core
design; Count-Min Sketch and HeavyKeeper both need an auxiliary top-k
min-heap bolted on. If that heap's memory isn't charged against the budget,
CMS/HeavyKeeper get a free memory advantage over Space-Saving at every
sweep point, invalidating the entire memory/accuracy comparison. Every
algorithm's `memory_bytes()` must include its top-k structure, and
`new_with_budget` must reserve `k * size_of::<HeapEntry>()` bytes for the
heap before sizing the rest of the sketch from what remains. At small
budgets with a large `k`, this can dominate the budget — that's a real,
reportable finding, not a bug to hide.

## Algorithms

**Count-Min Sketch** — `d` rows × `w` columns of `u32` counters in one flat
`Vec<u32>`. Fix `d` (recommend 4), derive `w` from the post-heap-reservation
budget. Hash functions are hand-rolled multiplicative hashes (`hash.rs`), not
a generic crate hasher — hash cost is part of what's being measured, and an
opaque hasher (e.g. SipHash, designed for flood-resistance irrelevant here)
would make that cost un-auditable. Primary invariant: CMS with plain
increment updates never undercounts — `query(x) >= exact.query(x)` for all
`x`. "Conservative update" is noted as a known accuracy-improving variant
but kept out of the Phase 1 baseline.

**Space-Saving** — exactly `m` monitored `{item, count, error}` entries,
indexed by a `HashMap<u64, usize>` plus a hand-rolled indexed min-heap
supporting decrease-key/removal-by-key (`std::collections::BinaryHeap`
doesn't support this; the same heap is reused by HeavyKeeper). `m` derives
from the budget at ~32 bytes/entry. Implement the indexed-heap version
(O(log m) per op) rather than the theoretically-optimal Stream-Summary
structure; revisit only if the sweep's own throughput data shows Space-Saving
disproportionately slow at large `m`. Invariant: `estimate - error <=
true_count <= estimate`, with `error == 0` for every item when `cardinality
<= m` (exactness check). Edge case the metrics layer must handle explicitly:
when `m < k`, recall is mechanically capped at `m/k` — no divide-by-zero, no
silent misreporting, and the plots should surface this rather than smooth it
away.

**HeavyKeeper** — same `d × w` layout idea as CMS, but 8-byte
`{fingerprint, count}` cells (vs CMS's 4-byte cells — this doubling at equal
budget is an expected, reportable part of the tradeoff, not a bug). On a
fingerprint mismatch, decay the existing count probabilistically
(`decay_base^(-count)`, paper's `b ≈ 1.08`); evict and overwrite at count 0.
This decay step is the most failure-prone part of the whole project — build
and test it against a hand-traced small example before trusting it at scale.
Needs its own lightweight seeded PRNG (e.g. splitmix64-style) for decay
decisions — a full `StdRng` per insert would be too slow. Because decay makes
HeavyKeeper non-deterministic given a fixed stream (unlike CMS/Space-Saving),
the harness runs multiple trials per config specifically to characterize
this variance. Fingerprint collisions are an accepted failure mode — must
not panic, just (correctly, per the algorithm's design) misattribute.

## Workload

A seeded Zipfian generator (`rand_distr::Zipf`) produces reproducible
`u64` key streams with configurable cardinality, skew, and stream length. An
exact `HashMap`-based ground-truth counter, run over the identical stream,
provides the reference frequency table and true top-k for accuracy scoring.

## Benchmark Harness

Nested sweep: skew × cardinality × stream length × trial × memory budget ×
algorithm. Ground truth is computed once per `(skew, cardinality,
stream_length)` combination and reused across every algorithm/budget/trial
for that combination, to avoid redundant work.

Design points that must hold to keep the numbers meaningful:

- **Fresh RNG per run.** Every run — the ground-truth pass and every
  algorithm×budget×trial pass — constructs its own `StdRng::seed_from_u64`
  from the same seed. Reusing one long-lived RNG across loop iterations is a
  classic bug that silently desyncs each "run" from a reproducible stream,
  and from ground truth.
- **Generation is timed separately from insertion.** Keys are generated into
  a reusable buffer outside the timed region; only the tight insert loop is
  timed, so throughput measures the sketch, not the Zipfian sampler.
- **Warmup.** The first 5–10% of the stream (configurable) is inserted but
  excluded from timing, so measurements reflect steady-state behavior
  (grown hashmaps, populated heaps, touched pages) rather than a cold start.
- **Static dispatch.** The sweep uses an `enum Algorithm { CountMin,
  SpaceSaving, HeavyKeeper }` matched once per run, calling a generic
  function so the compiler monomorphizes and inlines the hot insert path.
  `Box<dyn HeavyHitterSketch>` is avoided in the hot loop — a vtable call per
  item could mask the cache-effects the project exists to measure,
  especially at small budgets.

**Result schema** (one flat row per `(algorithm, memory_budget, skew,
cardinality, stream_length, trial)`, both CSV and JSON via one `serde`
struct): `run_id`, `algorithm`, `trial`, `seed`, `cardinality`, `skew`,
`stream_length`, `top_k_target`, `warmup_fraction`, `requested_memory_bytes`,
`actual_memory_bytes`, `construction_time_ns`, `insert_elapsed_ns`,
`items_per_sec`, `precision_at_k`, `recall_at_k`, `f1_at_k`,
`mean_relative_error`, `max_relative_error`, `underestimate_count`,
`overestimate_count`.

## Testing Strategy (build/TDD order)

1. `ExactCounter` first — it's the oracle every later test compares against.
2. `hash.rs` — basic uniformity/independence sanity checks.
3. Count-Min Sketch — property test: never undercounts vs `ExactCounter`.
4. Space-Saving — property test: `estimate - error <= true_count <=
   estimate`; exactness when `cardinality <= m`. `indexed_heap.rs` gets its
   own direct heap-invariant tests.
5. HeavyKeeper — hand-traced determinism test with a fixed internal seed;
   fingerprint-collision test asserting no panic.
6. `zipfian.rs` — reproducibility (same seed ⇒ identical sequence) and a
   loose statistical smoke test against known Zipf's-law behavior.
7. `ground_truth.rs` — unit test against a hand-crafted small stream.
8. `metrics.rs` — precision/recall/F1 edge cases: empty predicted set,
   predicted set larger than true set, exact match, no overlap, and the
   `m < k` degenerate Space-Saving case.

Sweep orchestration (`sweep.rs`/`main.rs`) is wired up last, once every
component underneath it is independently trusted — the highest-risk logic
(HeavyKeeper's decay, the error-bound invariants) should be pinned down
before it's buried inside a long-running sweep where bugs are hard to spot.

## Extensibility for Phase 2

Adding a novel candidate sketch later requires exactly: one new file in
`crates/sketches/src/` implementing `HeavyHitterSketch` (with honest
top-k/memory accounting per the convention above), and one new variant +
match arm in `sweep.rs`'s `Algorithm` enum. `workload/`, `metrics.rs`,
`report.rs`, `cli.rs`, and the sweep loop's structure are already
algorithm-agnostic; the CSV `algorithm` column just gains a new value that
flows through the existing plotting code unmodified.

## Verification Plan

1. `cargo test --workspace` — fast, deterministic, covers all unit/property
   tests above.
2. `cargo build --workspace --release` — timing runs must always use
   `--release`; debug-build numbers can have a qualitatively different
   relative ordering between algorithms and would be misleading.
3. **Smoke sweep** (seconds): small cardinality/stream-length, 2–3 skews,
   2–3 budgets, 1 trial. Sanity checks: Space-Saving's recall should
   visibly degrade at the smallest budget; CMS/HeavyKeeper throughput
   should be the same order of magnitude, with Space-Saving typically
   slower per-insert.
4. **Full sweep**: larger cardinality/stream-length, full skew list
   (e.g. `[0.8, 1.0, 1.2, 1.5]`), full budget list (e.g. `4KB`–`4MB`),
   `--trials 3` (needed because HeavyKeeper's decay is probabilistic).
   Run as a background job; use the smoke sweep's measured `items_per_sec`
   to estimate full-sweep runtime beforehand.
5. `python scripts/plot.py results/full_sweep.csv --out charts/` —
   produces accuracy-vs-memory, throughput-vs-memory, and skew-sensitivity
   charts. Verify chart files exist and axes/legends are sane.
