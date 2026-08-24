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

fn stream_seed(base_seed: u64, cardinality: u64, skew: f64) -> u64 {
    base_seed ^ cardinality.wrapping_mul(0x9E3779B97F4A7C15) ^ skew.to_bits().rotate_left(17)
}

fn sketch_seed(base_seed: u64, cardinality: u64, skew: f64, budget: usize, algorithm: Algorithm, trial: u32) -> u64 {
    stream_seed(base_seed, cardinality, skew)
        ^ (budget as u64).wrapping_mul(0xBF58476D1CE4E5B9)
        ^ (algorithm as u64).wrapping_mul(0x94D049BB133111EB)
        ^ (trial as u64).wrapping_add(1).wrapping_mul(0xD1B54A32D192ED03)
}

fn run_one<S: HeavyHitterSketch>(
    stream: &[u64],
    warmup_len: usize,
    budget_bytes: usize,
    k: usize,
    seed: u64,
) -> (S, u64, u64) {
    let construction_start = Instant::now();
    let mut sketch = S::new_with_budget(budget_bytes, k, seed);
    let construction_time_ns = construction_start.elapsed().as_nanos() as u64;

    for &item in &stream[..warmup_len] {
        sketch.insert(item);
    }

    let insert_start = Instant::now();
    for &item in &stream[warmup_len..] {
        sketch.insert(item);
    }
    let insert_elapsed_ns = insert_start.elapsed().as_nanos() as u64;

    (sketch, construction_time_ns, insert_elapsed_ns)
}

struct RunContext<'a> {
    args: &'a SweepArgs,
    cardinality: u64,
    skew: f64,
    ground_truth: &'a GroundTruth,
    true_items: &'a [u64],
    /// Number of leading stream items skipped as warmup, computed once per
    /// (cardinality, skew) sweep iteration and reused for both the insert-loop
    /// split (see `run_one`) and the `items_per_sec` calculation below, so the
    /// two never drift apart.
    warmup_len: usize,
}

fn build_row<S: HeavyHitterSketch>(
    ctx: &RunContext,
    algorithm: Algorithm,
    trial: u32,
    seed: u64,
    requested_memory_bytes: usize,
    construction_time_ns: u64,
    insert_elapsed_ns: u64,
    sketch: &S,
) -> SweepResult {
    let predicted_top_k = sketch.top_k(ctx.args.top_k);
    let predicted_items: Vec<u64> = predicted_top_k.iter().map(|&(item, _)| item).collect();

    let precision = precision_at_k(&predicted_items, ctx.true_items);
    let recall = recall_at_k(&predicted_items, ctx.true_items);
    let f1 = f1_score(precision, recall);

    let true_counts: Vec<(u64, u64)> = ctx
        .true_items
        .iter()
        .map(|&item| (item, ctx.ground_truth.query(item)))
        .collect();
    let error_stats = relative_error_stats(&true_counts, |item| sketch.query(item));

    let measured_items = ctx.args.stream_length.saturating_sub(ctx.warmup_len as u64);
    let items_per_sec = if insert_elapsed_ns == 0 {
        0.0
    } else {
        measured_items as f64 / (insert_elapsed_ns as f64 / 1_000_000_000.0)
    };

    SweepResult {
        run_id: format!(
            "{}-card{}-skew{}-mem{}-trial{}",
            algorithm.name(),
            ctx.cardinality,
            ctx.skew,
            requested_memory_bytes,
            trial
        ),
        algorithm: algorithm.name().to_string(),
        trial,
        seed,
        cardinality: ctx.cardinality,
        skew: ctx.skew,
        stream_length: ctx.args.stream_length,
        top_k_target: ctx.args.top_k,
        warmup_fraction: ctx.args.warmup_fraction,
        requested_memory_bytes,
        actual_memory_bytes: sketch.memory_bytes(),
        construction_time_ns,
        insert_elapsed_ns,
        items_per_sec,
        precision_at_k: precision,
        recall_at_k: recall,
        f1_at_k: f1,
        mean_relative_error: error_stats.mean_relative_error,
        max_relative_error: error_stats.max_relative_error,
        underestimate_count: error_stats.underestimate_count,
        overestimate_count: error_stats.overestimate_count,
        workload: ctx.args.workload.name().to_string(),
        num_heavy: ctx.args.num_heavy,
        heavy_jitter: ctx.args.heavy_jitter,
        heavy_mass_fraction: ctx.args.heavy_mass_fraction,
    }
}

pub fn run_sweep(args: &SweepArgs) -> Vec<SweepResult> {
    let mut results = Vec::new();

    for &cardinality in &args.cardinality {
        for &skew in &args.skew {
            // Generated once per (cardinality, skew): reused for ground truth
            // and every algorithm/budget/trial run below, per the Global
            // Constraints note on stream reuse.
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

            let ground_truth = GroundTruth::from_stream(&stream);
            let true_top_k = ground_truth.top_k(args.top_k);
            let true_items: Vec<u64> = true_top_k.iter().map(|&(item, _)| item).collect();

            let warmup_len = ((args.stream_length as f64) * args.warmup_fraction) as usize;

            let ctx = RunContext {
                args,
                cardinality,
                skew,
                ground_truth: &ground_truth,
                true_items: &true_items,
                warmup_len,
            };

            for &budget in &args.memory_budgets {
                for algorithm in Algorithm::all() {
                    for trial in 0..args.trials {
                        let sk_seed = sketch_seed(args.seed, cardinality, skew, budget, algorithm, trial);

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
                        results.push(row);
                    }
                }
            }
        }
    }

    results
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_args() -> SweepArgs {
        SweepArgs {
            cardinality: vec![200],
            stream_length: 5_000,
            skew: vec![1.2],
            memory_budgets: vec![4096, 16384],
            top_k: 10,
            trials: 2,
            seed: 42,
            warmup_fraction: 0.1,
            output: "results/tiny.csv".to_string(),
            format: "csv".to_string(),
            workload: Workload::Zipfian,
            num_heavy: 20,
            heavy_jitter: 0.1,
            heavy_mass_fraction: 0.8,
        }
    }

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

    #[test]
    fn tiny_sweep_produces_expected_row_count_and_sane_values() {
        let args = tiny_args();
        let results = run_sweep(&args);

        let expected_rows =
            args.cardinality.len() * args.skew.len() * args.memory_budgets.len() * 4 * args.trials as usize;
        assert_eq!(results.len(), expected_rows);

        for row in &results {
            assert!(row.items_per_sec > 0.0, "throughput must be positive: {row:?}");
            assert!(row.actual_memory_bytes > 0);
            assert!((0.0..=1.0).contains(&row.precision_at_k));
            assert!((0.0..=1.0).contains(&row.recall_at_k));
            assert!((0.0..=1.0).contains(&row.f1_at_k));
            assert!(row.mean_relative_error >= 0.0);
            assert!(row.max_relative_error >= row.mean_relative_error - 1e-9);
        }

        // Oracle assertion: at budget 16384 with top_k 10, Space-Saving's
        // monitored-entry count m works out to 507, which exceeds the
        // stream's cardinality of 200. That means Space-Saving can track
        // every distinct key with room to spare and must be exact. If
        // run_sweep were ever rewired so ground truth came from a
        // differently-seeded stream than what the sketches actually
        // consume, this assertion would catch it immediately, whereas the
        // range-only checks above would not.
        let exact_space_saving_rows: Vec<_> = results
            .iter()
            .filter(|row| row.algorithm == "space_saving" && row.requested_memory_bytes == 16384)
            .collect();
        assert!(
            !exact_space_saving_rows.is_empty(),
            "expected space_saving rows at requested_memory_bytes == 16384"
        );
        for row in exact_space_saving_rows {
            assert_eq!(row.mean_relative_error, 0.0, "expected exact tracking: {row:?}");
            assert_eq!(row.f1_at_k, 1.0, "expected perfect top-k recovery: {row:?}");
            assert_eq!(row.underestimate_count, 0, "expected no underestimates: {row:?}");
            assert_eq!(row.overestimate_count, 0, "expected no overestimates: {row:?}");
        }
    }
}
