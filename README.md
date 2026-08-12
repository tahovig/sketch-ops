# sketch-ops

Proof-of-concept for sub-linear stream processing and heavy-hitters detection.

Phase 1 (in progress): a Rust benchmark suite implementing Count-Min Sketch,
Space-Saving, and HeavyKeeper behind a common trait, benchmarked over
synthetic Zipfian streams to characterize memory/accuracy/throughput
tradeoffs. See `docs/superpowers/specs/` for the design doc.

Phase 2 (future): use Phase 1's benchmark harness to validate a novel
heavy-hitter sketch improvement against these baselines.

## Usage

### Running a sweep

```
cargo run --release --bin bench -- sweep \
  --cardinality 100000 \
  --stream-length 2000000 \
  --skew 0.8,1.2 \
  --memory-budgets 4096,65536,1048576 \
  --top-k 20 \
  --trials 3 \
  --seed 42 \
  --output results/sweep.csv
```

`--cardinality`, `--skew`, and `--memory-budgets` accept comma-separated
lists and are run as a full cross product against all three algorithms
(Count-Min, Space-Saving, HeavyKeeper). `--format` may be `csv` (default) or
`json`.

**Always build with `--release` for real measurements.** Debug-mode timing
numbers are not meaningful for cross-algorithm comparison — this is a global
constraint of the project, not a suggestion.

### Plotting results

```
python3 scripts/plot.py results/sweep.csv --out charts/
```

Requires `pandas` and `matplotlib` (`pip install pandas matplotlib`).

### Interpreting the output

- `mean_relative_error` and `max_relative_error` are computed **only over the
  true top-k items** for that run (the `top_k` heaviest keys per the exact
  ground truth), not over every distinct key in the stream. A low error here
  says nothing directly about accuracy on the long tail.
- Space-Saving's `actual_memory_bytes` (governed by
  `SPACE_SAVING_BYTES_PER_ENTRY = 32` in
  `crates/sketches/src/space_saving.rs`) accounts only for the algorithm's
  core counter/error arrays, per the design spec's convention. It does not
  additionally charge for `std` collection overhead such as the internal
  hash-map index used for O(1) lookup. Treat cross-algorithm memory
  comparisons at very small budgets as approximate, not exact byte-for-byte
  accounting.
