use crate::keys::rank_to_key;
use sketches::hash::SplitMix64;

pub struct PlateauGenerator {
    cardinality: u64,
    num_heavy: u64,
    heavy_jitter: f64,
    heavy_mass_fraction: f64,
    seed: u64,
}

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

    #[test]
    fn heavy_jitter_increases_spread_between_heavy_item_counts() {
        let cardinality = 1000u64;
        let num_heavy = 10u64;
        let stream_length = 50_000usize;
        let seed = 77u64;

        let spread_ratio = |jitter: f64| -> f64 {
            let gen = PlateauGenerator::new(cardinality, num_heavy, jitter, 0.8, seed);
            let stream = gen.generate(stream_length);
            let mut exact = ExactCounter::new();
            for &item in &stream {
                exact.insert(item);
            }
            let counts: Vec<u64> = (1..=num_heavy).map(rank_to_key).map(|k| exact.query(k)).collect();
            let max = *counts.iter().max().unwrap() as f64;
            let min = *counts.iter().min().unwrap() as f64;
            max / min
        };

        let tied_ratio = spread_ratio(0.0);
        let jittered_ratio = spread_ratio(0.9);

        assert!(tied_ratio < 1.3, "zero jitter should produce near-tied counts: ratio={tied_ratio}");
        assert!(
            jittered_ratio > tied_ratio * 1.5,
            "high jitter should produce materially more spread than zero jitter: tied={tied_ratio} jittered={jittered_ratio}"
        );
    }
}
