use std::time::Instant;

use sketches::count_min::CountMinSketch;
use sketches::heavy_keeper::HeavyKeeper;
use sketches::space_saving::SpaceSaving;
use sketches::traits::HeavyHitterSketch;
use workload::ground_truth::GroundTruth;
use workload::zipfian::ZipfianGenerator;

use crate::cli::SweepArgs;
use crate::metrics::{f1_score, precision_at_k, recall_at_k, relative_error_stats};
use crate::report::SweepResult;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Algorithm {
    CountMin,
    SpaceSaving,
    HeavyKeeper,
}

impl Algorithm {
    pub fn all() -> [Algorithm; 3] {
        [Algorithm::CountMin, Algorithm::SpaceSaving, Algorithm::HeavyKeeper]
    }

    pub fn name(&self) -> &'static str {
        match self {
            Algorithm::CountMin => "count_min",
            Algorithm::SpaceSaving => "space_saving",
            Algorithm::HeavyKeeper => "heavy_keeper",
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

    let warmup_items = ((ctx.args.stream_length as f64) * ctx.args.warmup_fraction) as u64;
    let measured_items = ctx.args.stream_length.saturating_sub(warmup_items);
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
            let stream_gen = ZipfianGenerator::new(cardinality, skew, shared_seed);
            let stream = stream_gen.generate(args.stream_length as usize);

            let ground_truth = GroundTruth::from_stream(&stream);
            let true_top_k = ground_truth.top_k(args.top_k);
            let true_items: Vec<u64> = true_top_k.iter().map(|&(item, _)| item).collect();

            let ctx = RunContext {
                args,
                cardinality,
                skew,
                ground_truth: &ground_truth,
                true_items: &true_items,
            };
            let warmup_len = ((args.stream_length as f64) * args.warmup_fraction) as usize;

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
        }
    }

    #[test]
    fn tiny_sweep_produces_expected_row_count_and_sane_values() {
        let args = tiny_args();
        let results = run_sweep(&args);

        let expected_rows =
            args.cardinality.len() * args.skew.len() * args.memory_budgets.len() * 3 * args.trials as usize;
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
    }
}
