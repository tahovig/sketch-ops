# sketch-ops

Proof-of-concept for sub-linear stream processing and heavy-hitters detection.

Phase 1 (in progress): a Rust benchmark suite implementing Count-Min Sketch,
Space-Saving, and HeavyKeeper behind a common trait, benchmarked over
synthetic Zipfian streams to characterize memory/accuracy/throughput
tradeoffs. See `docs/superpowers/specs/` for the design doc.

Phase 2 (future): use Phase 1's benchmark harness to validate a novel
heavy-hitter sketch improvement against these baselines.
