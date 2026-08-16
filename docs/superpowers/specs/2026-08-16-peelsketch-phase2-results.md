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

## Charts

Produced via `python3 scripts/plot.py results/phase2_comparison.csv --out charts/`:
`accuracy_vs_memory.png`, `throughput_vs_memory.png`, `skew_sensitivity.png`
(gitignored, not committed — regenerate from the CSV above if needed).
