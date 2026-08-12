use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::{Distribution, Zipf};

use crate::keys::rank_to_key;

pub struct ZipfianGenerator {
    cardinality: u64,
    skew: f64,
    seed: u64,
}

impl ZipfianGenerator {
    pub fn new(cardinality: u64, skew: f64, seed: u64) -> Self {
        Self { cardinality, skew, seed }
    }

    pub fn generate_ranks(&self, stream_length: usize) -> Vec<u64> {
        let mut rng = StdRng::seed_from_u64(self.seed);
        let zipf = Zipf::new(self.cardinality as f64, self.skew)
            .expect("valid zipf parameters (n >= 1, s > 0)");
        (0..stream_length)
            .map(|_| zipf.sample(&mut rng).round() as u64)
            .collect()
    }

    pub fn generate(&self, stream_length: usize) -> Vec<u64> {
        self.generate_ranks(stream_length).into_iter().map(rank_to_key).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_seed_produces_identical_stream() {
        let gen_a = ZipfianGenerator::new(1000, 1.2, 7);
        let gen_b = ZipfianGenerator::new(1000, 1.2, 7);
        let stream_a = gen_a.generate(500);
        let stream_b = gen_b.generate(500);
        assert_eq!(stream_a, stream_b);
    }

    #[test]
    fn higher_skew_concentrates_more_mass_on_top_rank() {
        let low_skew = ZipfianGenerator::new(1000, 0.5, 99);
        let high_skew = ZipfianGenerator::new(1000, 1.5, 99);
        let n = 50_000usize;
        let low_ranks = low_skew.generate_ranks(n);
        let high_ranks = high_skew.generate_ranks(n);
        let low_top_share = low_ranks.iter().filter(|&&r| r == 1).count() as f64 / n as f64;
        let high_top_share = high_ranks.iter().filter(|&&r| r == 1).count() as f64 / n as f64;
        assert!(
            high_top_share > low_top_share,
            "higher skew should concentrate more mass on rank 1: low={low_top_share} high={high_top_share}"
        );
    }
}
