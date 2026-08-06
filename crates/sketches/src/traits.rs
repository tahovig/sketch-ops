pub trait HeavyHitterSketch {
    fn new_with_budget(budget_bytes: usize, k: usize, seed: u64) -> Self
    where
        Self: Sized;
    fn insert(&mut self, item: u64);
    fn query(&self, item: u64) -> u64;
    fn top_k(&self, k: usize) -> Vec<(u64, u64)>;
    fn memory_bytes(&self) -> usize;
    fn name(&self) -> &'static str;
}
