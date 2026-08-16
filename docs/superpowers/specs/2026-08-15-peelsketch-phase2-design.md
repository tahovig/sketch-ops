# PeelSketch — Phase 2 Candidate Heavy-Hitter Sketch Design

## Context

Phase 1 built a rigorous, trustworthy benchmark harness and three known
heavy-hitter sketches (Count-Min Sketch, Space-Saving, HeavyKeeper) behind a
common `HeavyHitterSketch` trait, explicitly to make Phase 2's "our sketch is
better" claims credible rather than asserted. This is that Phase 2 spec.

The goal is not to reimplement a published algorithm, but to find a genuine,
defensible point of departure from the current state of the art — validated
empirically against the Phase 1 baselines using the harness already built for
that purpose.

**Research grounding.** A two-pass literature review (broad SOTA survey, then
a focused follow-up on one specific mechanism) found that peeling-based /
successive-cancellation decoding — find a confidently-decodable signal,
subtract its contribution, repeat on the residual — is a real, active lineage
applied to per-flow frequency sketches, not a novel idea in the abstract:

- **FlowRadar** (Li et al., USENIX NSDI 2016) — origin point; IBLT-style
  buckets (key-count, key-XOR-sum, value-sum) peeled when a bucket holds
  exactly one key.
- **MV-Sketch** (Tang, Huang, Lee, IEEE INFOCOM 2019) — majority voting
  within each bucket to track a single "owner" flow, prioritizing throughput.
- **PR-Sketch** (Huang et al., PVLDB 14(10), 2021) — recovers per-key
  aggregates by solving linear equations relating counter values to keys.
- **Hidden Sketch** (arXiv:2505.12293, 2025) — explicit peeling via a
  Reversible Bloom Filter to find pure buckets, falling back to solving a
  linear system (SVD) when peeling stalls. Explicitly names the same
  weakness this project's HeavyKeeper baseline has: collisions *between*
  high-frequency items, which replacement/decay strategies "are powerless
  against."

**The gap.** Every one of these pays for peeling with a second, key-aware
auxiliary structure (a Reversible Bloom Filter, an explicit key-count field,
or a full linear system) to know *which* keys are candidates before peeling.
None of them determine cell purity from the counter's own statistics alone,
with no auxiliary structure — which is exactly the constraint that keeps
CMS/HeavyKeeper flat and cheap. That gap — statistics-only purity detection,
no second structure — is genuinely unexplored in everything the research
found, and is what this design bets on. If the bet doesn't pay off, this spec
also defines a concrete fallback rather than leaving that as an open risk
(see Fallback Plan).

## Goals / Success Criteria

- A new sketch, **PeelSketch**, implementing `HeavyHitterSketch`, whose cells
  determine purity from an in-place Boyer-Moore-style majority-vote signal —
  no auxiliary key-tracking structure.
- A single, bounded (one-level, non-recursive) residual-cascade subtraction
  at query time, used to refine estimates for items that collide with a
  cell's majority occupant.
- Integrated into the Phase 1 sweep harness as a fourth `Algorithm` variant,
  with zero changes to `workload`, `metrics.rs`, `report.rs`, or `cli.rs` —
  exactly the extensibility Phase 1's own design promised.
- A concrete, falsifiable success criterion: PeelSketch measurably beats
  HeavyKeeper (F1, outside the noise floor `--trials` already exposes)
  *somewhere* in the existing memory×skew sweep grid. This is the fallback
  trigger, not just a target.

## Non-Goals (this phase)

- Full iterative peeling with a linear-system fallback (Hidden Sketch's
  approach) — out of scope. PeelSketch does one bounded level of cascade;
  going further starts turning this into a re-implementation of published
  work rather than a novel bet.
- A dedicated adversarial/collision-forcing workload generator. The existing
  Zipfian sweep's high-skew/small-budget corners already produce heavy
  collision pressure; a purpose-built generator is reasonable future scope
  if that turns out to be insufficient, not part of this phase.
- Detailed design of the auxiliary-structure fallback. Sketched at a rough,
  directional level below; not worth fully speccing unless the primary bet
  actually fails.
- Multi-threading — inherited from Phase 1, unchanged.
- Depth as a first-class swept CLI dimension — see Algorithm Design.

## Architecture

Two file changes, matching Phase 1's stated extensibility contract exactly:

- New: `crates/sketches/src/peel_sketch.rs` — the `PeelSketch` struct,
  implementing `HeavyHitterSketch`.
- Modified: `crates/bench/src/sweep.rs` — one new `Algorithm::PeelSketch`
  variant and one new match arm. `Algorithm::all()` already drives every
  sweep combination through all variants with no CLI changes required, so
  `PeelSketch` is swept automatically once added.

No other file changes. `workload/`, `metrics.rs`, `report.rs`, `cli.rs`, and
`scripts/plot.py` are algorithm-agnostic per Phase 1's design; the CSV
`algorithm` column simply gains a fourth value.

## Algorithm Design

### Cell layout

```rust
#[derive(Debug, Clone, Copy)]
struct Cell {
    candidate_fingerprint: u32,  // current Boyer-Moore majority candidate
    vote_margin: u32,            // Boyer-Moore vote count; never negative
    raw_total: u32,              // CMS-style additive counter, saturating
}
```

12 bytes/cell — 1.5× HeavyKeeper's 8-byte `{fingerprint, count}` cell, 3×
CMS's bare 4-byte counter. This is a real, honest cost: at a fixed memory
budget, PeelSketch gets fewer, richer cells than either baseline. Whether the
purity signal earns back that width reduction is the central empirical
question this design exists to answer — not asserted here, measured by the
existing sweep harness.

`raw_total` and `vote_margin` use saturating arithmetic (not wrapping),
defensive against the multi-million-item streams the harness already
exercises.

### Sizing and depth

Same convention as CMS/HeavyKeeper: reserve `heap_bytes = k *
size_of::<HeapEntry>()` first, then size the remaining budget into `depth ×
width` cells.

`depth` is fixed at a `PEEL_DEPTH = 4` constant for the primary
`new`/`new_with_budget` constructors — matching `CMS_DEPTH`/`HK_DEPTH`
exactly, so any accuracy difference measured against the baselines is
attributable to the cell/cascade design, not a confounding change in row
count. A `with_dimensions(depth, width, k, seed)` constructor (mirroring
HeavyKeeper's existing escape hatch) exists for direct testing and future
exploratory depth-sensitivity analysis — not wired into the CLI or
`Algorithm` enum, matching how HeavyKeeper's own `with_dimensions` is used
today only by its unit tests.

Fingerprints reuse HeavyKeeper's exact scheme: `hash.hash(depth, item) as
u32`, with a `0` result coerced to `1` so it never collides with an
empty-cell sentinel.

### Insert

```
fp = fingerprint_of(item)
for each row r in 0..depth:
    cell = cells[r][hash_r(item)]
    cell.raw_total = cell.raw_total.saturating_add(1)
    if cell.raw_total == 1 {                 // first insert into this cell
        cell.candidate_fingerprint = fp
        cell.vote_margin = 1
    } else if fp == cell.candidate_fingerprint {
        cell.vote_margin = cell.vote_margin.saturating_add(1)
    } else {
        cell.vote_margin -= 1
        if cell.vote_margin == 0 {
            cell.candidate_fingerprint = fp
            cell.vote_margin = 1
        }
    }
feed current estimate into the existing update_top_k heap pattern
```

### Query / top-k

```
fp = fingerprint_of(item)
for each row r in 0..depth:
    cell = cells[r][hash_r(item)]
    if cell.candidate_fingerprint == fp {
        row_estimate[r] = cell.raw_total      // item is (probably) the row's majority
    } else {
        // Search only OTHER rows (r' != r) for a clean read on the blocking
        // candidate — row r's own raw_total is exactly the contaminated
        // value we're trying to refine, and including it here would let it
        // trivially satisfy its own "candidate matches" test, making
        // majority_estimate <= raw_total by construction and potentially
        // zeroing out the residual we're trying to isolate.
        majority_estimate = min over rows r' != r where cell'[r'].candidate_fingerprint
                             == cell.candidate_fingerprint of cell'[r'].raw_total
                             // 0 if no other row's candidate matches (this is
                             // always the case when depth == 1 — the cascade
                             // cannot refine anything with only one row)
        row_estimate[r] = cell.raw_total.saturating_sub(majority_estimate)
    }
estimate = min over r of row_estimate[r]
```

The cascade is deliberately bounded to **one level**: estimating a blocking
candidate's count only from *other* rows where it is directly the local
majority, never recursively chasing that candidate's own blockers. Full
iterative peeling (resolve one item, subtract everywhere, repeat until
nothing more resolves, falling back to solving a linear system when it
stalls — Hidden Sketch's approach) is real additional complexity explicitly
excluded as a non-goal. If no other row's candidate matches, `majority_estimate`
is `0` and the cascade contributes nothing — this degrades gracefully to
CMS's plain `min`-across-rows behavior for that row. Note this means the
cascade is inherently a no-op at `depth == 1`; the forced-collision unit
test (see Testing Strategy) must use `depth >= 2` via `with_dimensions` to
actually exercise it.

`top_k` reuses the same per-insert estimate and heap-update pattern already
used by `CountMinSketch`/`HeavyKeeper`.

### Property change: the never-undercount guarantee does not hold

CMS never undercounts — its counters only grow, so `min`-across-rows is
always ≥ the true count. PeelSketch's subtraction step breaks this outright:
over-estimating a blocking candidate's count and subtracting too much can
push a residual estimate *below* the true count. This is a real, deliberate
departure from every Phase 1 baseline, not an oversight, and must be stated
plainly rather than discovered during review.

The one structural guarantee PeelSketch *does* retain is Boyer-Moore's
classical majority theorem: if a single fingerprint accounts for a true
majority of a cell's inserts, it is provably the cell's final
`candidate_fingerprint`. That is a correctness property about the internal
mechanism, not an accuracy bound on estimates, and is worth its own test
(see Testing Strategy). Whether the trade — a clean worst-case bound for
better typical-case accuracy — actually pays off is an empirical question,
directly measurable via `SweepResult`'s existing `underestimate_count` /
`overestimate_count` columns.

## Fallback Plan

**Trigger.** Run PeelSketch through the existing sweep harness against
HeavyKeeper (the closest baseline, and Hidden Sketch's own stated target) at
matched memory budgets across the existing skew/budget grid, using
`--trials` for a noise floor. If PeelSketch is not better than HeavyKeeper
*anywhere* in that grid — dominated everywhere, or indistinguishable from
trial-to-trial noise — that is the signal to stop iterating on the ID-free
design and pivot.

**Rough shape of the fallback**, kept intentionally shallow: add a small
auxiliary key-aware structure (Reversible-Bloom-Filter-style, closest in
spirit to Hidden Sketch) so purity is determined by near-exact key tracking
instead of the statistical vote margin. This is a materially different,
more complex sketch — exactly why it is the fallback and not the plan.
Detailed design is deferred unless the trigger above actually fires.

## Testing Strategy

Unit tests in `crates/sketches/src/peel_sketch.rs`, following the patterns
already established by CMS/HeavyKeeper's test suites:

1. **Hand-traced Boyer-Moore sequence** — a fixed, deterministic sequence of
   inserts into a single cell, asserting the exact `candidate_fingerprint`/
   `vote_margin` trajectory (mirrors HeavyKeeper's
   `hand_traced_decay_sequence_is_deterministic`).
2. **Memory-budget reservation-ordering test** — same pattern as
   CMS/HeavyKeeper's `memory_bytes_reserves_heap_budget_before_sizing_*`
   tests, adapted for the 12-byte `Cell`.
3. **Forced-collision cascade test** — using `with_dimensions` to build a
   deliberately tiny sketch (`depth >= 2` — the cascade is a no-op at
   depth 1, see Query above — with `width = 1`) where two known items are
   guaranteed to collide, asserting the cascade's subtracted residual
   behaves sanely for the minority item.
4. **Boyer-Moore majority-theorem property test** (`proptest`, already a
   dev-dependency) — for arbitrary insert sequences into a single cell, if
   one fingerprint accounts for a true majority of inserts, it must end as
   `candidate_fingerprint`. This replaces the CMS-style never-undercounts
   invariant, which PeelSketch does not honestly satisfy.
5. **Structural fuzzing** (`proptest`) — `raw_total` only grows,
   `vote_margin` never goes negative, `candidate_fingerprint` is always a
   fingerprint that was actually inserted into that cell. Accuracy against
   ground truth is the benchmark harness's job, not the unit tests'.

Sketch-level tests are built and trusted before wiring `Algorithm::PeelSketch`
into `sweep.rs`, matching Phase 1's own build order (highest-risk logic
pinned down in isolation before it's buried inside a long-running sweep).

## Verification Plan

1. `cargo test --workspace` — unit and property tests above.
2. `cargo build --workspace --release` — required before any timing numbers
   are trusted, per Phase 1's global constraint (unchanged).
3. Sweep PeelSketch alongside the three existing algorithms using the
   existing CLI, unmodified — `Algorithm::all()` already includes every
   variant in every sweep, so no new flags are needed.
4. Apply the fallback-trigger comparison against HeavyKeeper across the
   swept grid.
5. `python3 scripts/plot.py` — unmodified; the new `algorithm` value flows
   through the existing charts.
