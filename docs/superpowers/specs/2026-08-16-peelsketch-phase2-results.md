# PeelSketch Phase 2 — Comparison Results

## Command run

```
cargo run --release --bin bench -- sweep --cardinality 100000 --stream-length 2000000 --skew 0.8,1.2 --memory-budgets 4096,65536,1048576 --top-k 20 --trials 3 --seed 42 --output results/phase2_comparison.csv
```

## Fallback-trigger comparison (PeelSketch vs. HeavyKeeper, mean F1 by skew/budget)

```
  skew     budget    peel_f1      hk_f1     margin  noise_floor    verdict
   0.8       4096     0.3667     0.8500    -0.4833       0.0577         no
   0.8      65536     0.9667     1.0000    -0.0333       0.0289         no
   0.8    1048576     1.0000     1.0000     0.0000       0.0000         no
   1.2       4096     0.7333     1.0000    -0.2667       0.0289         no
   1.2      65536     1.0000     1.0000     0.0000       0.0000         no
   1.2    1048576     1.0000     1.0000     0.0000       0.0000         no

VERDICT: PeelSketch does not beat HeavyKeeper anywhere beyond noise in this grid. Fallback trigger criterion met.
```

## Verdict

PeelSketch does **not** beat HeavyKeeper outside the trial-to-trial noise
floor anywhere in this grid. Every one of the six (skew, budget) cells comes
back "no" — three cells are exact ties (both algorithms hit F1 = 1.0 with
zero variance, at the two largest budgets), and the remaining three cells
show PeelSketch strictly *losing* to HeavyKeeper by a margin well outside
the noise floor:

- skew 0.8, 4096 B: PeelSketch 0.3667 vs. HeavyKeeper 0.8500 — a −0.4833
  margin against a 0.0577 noise floor. This is PeelSketch's worst showing,
  at the tightest memory budget and lowest skew (the hardest regime: less
  concentrated mass, less room to track candidates).
- skew 1.2, 4096 B: PeelSketch 0.7333 vs. HeavyKeeper 1.0000 — a −0.2667
  margin against a 0.0289 noise floor.
- skew 0.8, 65536 B: PeelSketch 0.9667 vs. HeavyKeeper 1.0000 — a smaller
  but still noise-floor-exceeding −0.0333 margin against 0.0289.

There is no cell in the grid where PeelSketch's mean F1 exceeds
HeavyKeeper's, let alone by more than the combined std-dev noise floor. The
best PeelSketch achieves relative to HeavyKeeper is a tie at generous memory
budgets (65536 B at skew 1.2, and 1048576 B at both skews), where both
sketches saturate at perfect F1 and there is nothing left to differentiate.
It never wins outright, and at the tightest budget — the regime where a
sketch's design is supposed to matter most — it loses by a wide margin.

Per the design spec's Fallback Plan
(`docs/superpowers/specs/2026-08-15-peelsketch-phase2-design.md`): the
trigger is "if PeelSketch is not better than HeavyKeeper *anywhere* in that
grid — dominated everywhere, or indistinguishable from trial-to-trial
noise." That is exactly the outcome observed here: PeelSketch is either
dominated (three cells, real margins beyond noise) or indistinguishable
(three cells, exact ties at F1 = 1.0). The fallback trigger criterion **has
been met**. The next step is the auxiliary-structure fallback sketched in
that document (a small Reversible-Bloom-Filter-style key-aware structure to
determine purity via near-exact key tracking instead of the Boyer-Moore
statistical vote margin), not further iteration on the current ID-free
mechanism.

## Why it lost

The design spec committed to measuring the accuracy trade "directly
measurable via `SweepResult`'s existing `underestimate_count` /
`overestimate_count` columns" (design spec, "Property change" section). That
diagnostic was run against `results/phase2_comparison.csv` after the fact;
this section reports what it shows.

**Undercounting never happens; overcounting is pervasive.** Across all 18
PeelSketch rows in the sweep, `underestimate_count` is 0 in every single
row — PeelSketch never undercounts a single top-*k* item anywhere in the
grid. `overestimate_count` (out of `top_k` = 20) is 20 — every top-*k* item
overestimated — in all 12 rows at the two smaller budgets (4096 and 65536),
and remains high (13–20) even at the largest budget (1,048,576 B), where it
never drops to 0 despite `mean_relative_error` shrinking to near-zero there.
In other words, PeelSketch essentially always overestimates every top-*k*
item, to some degree, everywhere in the grid. This is the **opposite** of
what the design's "Property change" discussion focused on: that section's
main worry was PeelSketch losing CMS's never-undercount guarantee, reasoning
through the case where the `raw_total - vote_margin` residual could sit
below the true count for a minority item. Empirically, that failure mode
never materializes for top-*k* items — the actual, dominant failure mode is
overcounting, which the design spec didn't examine.

**PeelSketch loses to Count-Min itself, not just HeavyKeeper, at the worst
cell.** At skew = 0.8, budget = 4096 — the hardest regime in the grid —
PeelSketch's `mean_relative_error` averages ≈3.00 across the three trials
(2.957, 3.010, 3.021), versus Count-Min's ≈0.92 (0.897, 0.936, 0.915) at the
same cell: PeelSketch's mean relative error is about **3.3×** worse than
plain Count-Min's, on the same memory budget and workload.

**Root cause: cell-width tax.** PeelSketch's `Cell` is 12 bytes
(`candidate_fingerprint: u32, vote_margin: u32, raw_total: u32` —
`crates/sketches/src/peel_sketch.rs`), versus HeavyKeeper's 8-byte
`{fingerprint, count}` cell and Count-Min's bare 4-byte counter. At
budget = 4096 with `top_k` = 20, the top-*k* heap reserves `20 *
size_of::<HeapEntry>() = 20 * 16 = 320` bytes, leaving 3,776 bytes for
cells. At `depth = 4`, that 3,776-byte remainder yields 3776/12 = 314 total
cells → 78 columns/row for PeelSketch, versus 3776/8 = 472 cells → 118
columns/row for HeavyKeeper, versus 3776/4 = 944 cells → 236 columns/row for
Count-Min. PeelSketch gets roughly a third of Count-Min's columns for the
same budget. And because a genuinely heavy item is almost always the
majority candidate in its own cell, `query()`'s majority branch fires for
it and returns the raw, unfiltered `raw_total` — the local-residual
refinement (`raw_total - vote_margin`) only ever applies to items that are
*not* the majority candidate in a given cell. So for the items that matter
most to top-*k* accuracy — the heavy hitters themselves — PeelSketch
effectively behaves like a narrower, worse-provisioned Count-Min Sketch: all
of the width penalty, none of the purity-signal benefit, because the purity
signal is consulted for minority collisions, not for the majority items
being scored.

**PeelSketch also lost to Count-Min directly, not just HeavyKeeper, at both
small-budget cells** — worth stating because only the HeavyKeeper comparison
was the pre-registered fallback-trigger criterion, and the gap to Count-Min
underscores how far off the result was. At skew = 0.8, budget = 4096: mean
F1 is 0.367 for PeelSketch vs. 0.783 for Count-Min. At skew = 1.2,
budget = 4096: mean F1 is 0.733 for PeelSketch vs. 0.917 for Count-Min.

**A note for whoever picks up the fallback.** The design spec's Fallback
Plan sketches an auxiliary Reversible-Bloom-Filter-style structure, with
detailed design deferred "unless the trigger fires" — it has now fired.
Whatever that auxiliary structure costs per cell (or as a side structure)
needs to be budgeted carefully: if it adds meaningfully more than
PeelSketch's 12 bytes/cell, it inherits the same cell-width tax identified
here — fewer, richer cells losing to more, plainer ones at a fixed memory
budget — unless the design sizes it deliberately to avoid that trap.

## Charts

Produced via `python3 scripts/plot.py results/phase2_comparison.csv --out charts/`:
`accuracy_vs_memory.png`, `throughput_vs_memory.png`, `skew_sensitivity.png`
(gitignored, not committed — regenerate from the CSV above if needed).
