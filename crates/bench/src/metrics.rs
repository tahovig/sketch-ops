use std::collections::HashSet;

pub fn precision_at_k(predicted: &[u64], true_items: &[u64]) -> f64 {
    if predicted.is_empty() {
        return 0.0;
    }
    let true_set: HashSet<u64> = true_items.iter().copied().collect();
    let hits = predicted.iter().filter(|item| true_set.contains(item)).count();
    hits as f64 / predicted.len() as f64
}

pub fn recall_at_k(predicted: &[u64], true_items: &[u64]) -> f64 {
    if true_items.is_empty() {
        return 1.0;
    }
    let predicted_set: HashSet<u64> = predicted.iter().copied().collect();
    let hits = true_items.iter().filter(|item| predicted_set.contains(item)).count();
    hits as f64 / true_items.len() as f64
}

pub fn f1_score(precision: f64, recall: f64) -> f64 {
    if precision + recall == 0.0 {
        0.0
    } else {
        2.0 * precision * recall / (precision + recall)
    }
}

pub struct RelativeErrorStats {
    pub mean_relative_error: f64,
    pub max_relative_error: f64,
    pub underestimate_count: u64,
    pub overestimate_count: u64,
}

pub fn relative_error_stats<F: Fn(u64) -> u64>(true_counts: &[(u64, u64)], estimate_fn: F) -> RelativeErrorStats {
    if true_counts.is_empty() {
        return RelativeErrorStats {
            mean_relative_error: 0.0,
            max_relative_error: 0.0,
            underestimate_count: 0,
            overestimate_count: 0,
        };
    }

    let mut sum_rel_error = 0.0;
    let mut max_rel_error = 0.0_f64;
    let mut underestimate_count = 0u64;
    let mut overestimate_count = 0u64;

    for &(item, true_count) in true_counts {
        let estimate = estimate_fn(item);
        let rel_error = if true_count == 0 {
            0.0
        } else {
            (estimate as f64 - true_count as f64).abs() / true_count as f64
        };
        sum_rel_error += rel_error;
        if rel_error > max_rel_error {
            max_rel_error = rel_error;
        }
        if estimate < true_count {
            underestimate_count += 1;
        }
        if estimate > true_count {
            overestimate_count += 1;
        }
    }

    RelativeErrorStats {
        mean_relative_error: sum_rel_error / true_counts.len() as f64,
        max_relative_error: max_rel_error,
        underestimate_count,
        overestimate_count,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn precision_recall_f1_exact_match() {
        let predicted = vec![1, 2, 3];
        let truth = vec![1, 2, 3];
        assert_eq!(precision_at_k(&predicted, &truth), 1.0);
        assert_eq!(recall_at_k(&predicted, &truth), 1.0);
        assert_eq!(f1_score(1.0, 1.0), 1.0);
    }

    #[test]
    fn precision_recall_no_overlap() {
        let predicted = vec![1, 2, 3];
        let truth = vec![4, 5, 6];
        assert_eq!(precision_at_k(&predicted, &truth), 0.0);
        assert_eq!(recall_at_k(&predicted, &truth), 0.0);
        assert_eq!(f1_score(0.0, 0.0), 0.0);
    }

    #[test]
    fn precision_empty_predicted_set() {
        let predicted: Vec<u64> = vec![];
        let truth = vec![1, 2, 3];
        assert_eq!(precision_at_k(&predicted, &truth), 0.0);
        assert_eq!(recall_at_k(&predicted, &truth), 0.0);
    }

    #[test]
    fn predicted_larger_than_true_set() {
        let predicted = vec![1, 2, 3, 4, 5];
        let truth = vec![1, 2];
        assert_eq!(precision_at_k(&predicted, &truth), 2.0 / 5.0);
        assert_eq!(recall_at_k(&predicted, &truth), 1.0);
    }

    #[test]
    fn recall_is_capped_when_monitored_set_smaller_than_k_space_saving_case() {
        // Space-Saving with m=3 monitored entries but a top-k target of k=5:
        // even if every monitored item is a true heavy hitter, recall cannot
        // exceed m/k, and no divide-by-zero or panic occurs.
        let predicted = vec![10, 20, 30]; // m = 3
        let truth = vec![10, 20, 30, 40, 50]; // k = 5
        let recall = recall_at_k(&predicted, &truth);
        assert_eq!(recall, 3.0 / 5.0);
    }

    #[test]
    fn relative_error_stats_basic() {
        let true_counts = vec![(1u64, 100u64), (2u64, 50u64), (3u64, 10u64)];
        let estimates = |item: u64| match item {
            1 => 110u64,
            2 => 40u64,
            3 => 10u64,
            _ => 0u64,
        };
        let stats = relative_error_stats(&true_counts, estimates);
        assert!((stats.mean_relative_error - ((0.1 + 0.2 + 0.0) / 3.0)).abs() < 1e-9);
        assert!((stats.max_relative_error - 0.2).abs() < 1e-9);
        assert_eq!(stats.underestimate_count, 1);
        assert_eq!(stats.overestimate_count, 1);
    }

    #[test]
    fn relative_error_stats_empty_input_has_no_divide_by_zero() {
        let true_counts: Vec<(u64, u64)> = vec![];
        let stats = relative_error_stats(&true_counts, |_| 0u64);
        assert_eq!(stats.mean_relative_error, 0.0);
        assert_eq!(stats.max_relative_error, 0.0);
        assert_eq!(stats.underestimate_count, 0);
        assert_eq!(stats.overestimate_count, 0);
    }
}
