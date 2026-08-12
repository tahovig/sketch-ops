use std::io;
use std::path::Path;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SweepResult {
    pub run_id: String,
    pub algorithm: String,
    pub trial: u32,
    pub seed: u64,
    pub cardinality: u64,
    pub skew: f64,
    pub stream_length: u64,
    pub top_k_target: usize,
    pub warmup_fraction: f64,
    pub requested_memory_bytes: usize,
    pub actual_memory_bytes: usize,
    pub construction_time_ns: u64,
    pub insert_elapsed_ns: u64,
    pub items_per_sec: f64,
    pub precision_at_k: f64,
    pub recall_at_k: f64,
    pub f1_at_k: f64,
    pub mean_relative_error: f64,
    pub max_relative_error: f64,
    pub underestimate_count: u64,
    pub overestimate_count: u64,
}

pub fn write_csv(path: &Path, rows: &[SweepResult]) -> io::Result<()> {
    let mut writer = csv::Writer::from_path(path)?;
    for row in rows {
        writer.serialize(row).map_err(io::Error::other)?;
    }
    writer.flush()?;
    Ok(())
}

pub fn write_json(path: &Path, rows: &[SweepResult]) -> io::Result<()> {
    let file = std::fs::File::create(path)?;
    serde_json::to_writer_pretty(file, rows).map_err(io::Error::other)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_row() -> SweepResult {
        SweepResult {
            run_id: "run-0001".to_string(),
            algorithm: "count_min".to_string(),
            trial: 0,
            seed: 42,
            cardinality: 100_000,
            skew: 1.2,
            stream_length: 1_000_000,
            top_k_target: 20,
            warmup_fraction: 0.05,
            requested_memory_bytes: 65536,
            actual_memory_bytes: 65520,
            construction_time_ns: 1200,
            insert_elapsed_ns: 950_000_000,
            items_per_sec: 1_052_631.0,
            precision_at_k: 0.95,
            recall_at_k: 0.90,
            f1_at_k: 0.924,
            mean_relative_error: 0.02,
            max_relative_error: 0.11,
            underestimate_count: 3,
            overestimate_count: 5,
        }
    }

    #[test]
    fn csv_roundtrip_preserves_all_fields() {
        let path = std::env::temp_dir().join("sketch_ops_report_roundtrip_test.csv");
        let rows = vec![sample_row()];
        write_csv(&path, &rows).expect("write_csv should succeed");

        let mut reader = csv::Reader::from_path(&path).expect("csv::Reader::from_path should succeed");
        let read_back: Vec<SweepResult> = reader
            .deserialize()
            .collect::<Result<Vec<SweepResult>, csv::Error>>()
            .expect("deserialize should succeed");

        std::fs::remove_file(&path).ok();
        assert_eq!(read_back, rows);
    }

    #[test]
    fn json_roundtrip_preserves_all_fields() {
        let path = std::env::temp_dir().join("sketch_ops_report_roundtrip_test.json");
        let rows = vec![sample_row()];
        write_json(&path, &rows).expect("write_json should succeed");

        let contents = std::fs::read_to_string(&path).expect("read_to_string should succeed");
        let read_back: Vec<SweepResult> = serde_json::from_str(&contents).expect("from_str should succeed");

        std::fs::remove_file(&path).ok();
        assert_eq!(read_back, rows);
    }
}
