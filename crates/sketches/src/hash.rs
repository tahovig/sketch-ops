pub struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }

    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }
}

pub struct HashFamily {
    multipliers: Vec<u64>,
}

impl HashFamily {
    pub fn new(count: usize, seed: u64) -> Self {
        let mut rng = SplitMix64::new(seed);
        let multipliers = (0..count).map(|_| rng.next_u64() | 1).collect();
        Self { multipliers }
    }

    pub fn hash(&self, row: usize, item: u64) -> u64 {
        let mixed = item.wrapping_mul(self.multipliers[row]);
        mixed ^ (mixed >> 32)
    }

    pub fn hash_to_width(&self, row: usize, item: u64, width: usize) -> usize {
        (self.hash(row, item) as usize) % width
    }

    pub fn depth(&self) -> usize {
        self.multipliers.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splitmix64_next_f64_is_in_unit_interval() {
        let mut rng = SplitMix64::new(1);
        for _ in 0..1000 {
            let v = rng.next_f64();
            assert!((0.0..1.0).contains(&v), "value out of range: {v}");
        }
    }

    #[test]
    fn same_seed_produces_deterministic_hash_family() {
        let hf_a = HashFamily::new(4, 999);
        let hf_b = HashFamily::new(4, 999);
        for item in [1u64, 2, 100, 12345] {
            for row in 0..4 {
                assert_eq!(hf_a.hash(row, item), hf_b.hash(row, item));
            }
        }
    }

    #[test]
    fn hash_to_width_is_roughly_uniform() {
        let hf = HashFamily::new(4, 1234);
        let width = 256usize;
        let mut buckets = vec![0u32; width];
        let n = 100_000u64;
        for item in 0..n {
            let col = hf.hash_to_width(0, item, width);
            buckets[col] += 1;
        }
        let expected = n as f64 / width as f64;
        let max_deviation = buckets.iter().map(|&c| (c as f64 - expected).abs()).fold(0.0, f64::max);
        assert!(
            max_deviation < expected * 0.4,
            "bucket distribution too skewed: max_dev={max_deviation} expected={expected}"
        );
    }

    #[test]
    fn hash_rows_are_independent() {
        let hf = HashFamily::new(4, 1234);
        let width = 1024usize;
        let mut agreements = 0u32;
        let n = 1000u64;
        for item in 0..n {
            let c0 = hf.hash_to_width(0, item, width);
            let c1 = hf.hash_to_width(1, item, width);
            if c0 == c1 {
                agreements += 1;
            }
        }
        assert!(agreements < (n as u32) / 10, "hash rows 0 and 1 agree too often: {agreements}/{n}");
    }
}
