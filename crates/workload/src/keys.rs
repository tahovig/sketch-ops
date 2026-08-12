/// Maps a 1-indexed Zipfian rank to a synthetic, IP-like u64 key: multiplies
/// by a large odd constant and XOR-folds to spread ranks across the u64
/// space, deterministically and reproducibly.
pub fn rank_to_key(rank: u64) -> u64 {
    let mixed = rank.wrapping_mul(0x9E3779B97F4A7C15);
    mixed ^ (mixed >> 29)
}
