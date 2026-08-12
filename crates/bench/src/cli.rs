use clap::{Args, Parser, Subcommand};

#[derive(Parser, Debug)]
#[command(name = "bench", version, about = "Heavy-hitters sketch benchmark harness")]
pub struct Cli {
    #[command(subcommand)]
    pub command: Command,
}

#[derive(Subcommand, Debug)]
pub enum Command {
    Sweep(SweepArgs),
}

#[derive(Args, Debug, Clone)]
pub struct SweepArgs {
    #[arg(long, value_delimiter = ',', required = true)]
    pub cardinality: Vec<u64>,

    #[arg(long)]
    pub stream_length: u64,

    #[arg(long, value_delimiter = ',', required = true)]
    pub skew: Vec<f64>,

    #[arg(long, value_delimiter = ',', required = true)]
    pub memory_budgets: Vec<usize>,

    #[arg(long, default_value_t = 20)]
    pub top_k: usize,

    #[arg(long, default_value_t = 1)]
    pub trials: u32,

    #[arg(long, default_value_t = 42)]
    pub seed: u64,

    #[arg(long, default_value_t = 0.05)]
    pub warmup_fraction: f64,

    #[arg(long, default_value = "results/sweep.csv")]
    pub output: String,

    #[arg(long, default_value = "csv")]
    pub format: String,
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    #[test]
    fn parses_sweep_args_with_comma_separated_lists() {
        let cli = Cli::parse_from([
            "bench",
            "sweep",
            "--cardinality", "1000,10000",
            "--stream-length", "50000",
            "--skew", "0.8,1.2",
            "--memory-budgets", "4096,65536",
            "--top-k", "10",
            "--trials", "2",
            "--seed", "7",
            "--output", "results/test.csv",
        ]);

        let args = match cli.command {
            Command::Sweep(args) => args,
        };

        assert_eq!(args.cardinality, vec![1000, 10000]);
        assert_eq!(args.stream_length, 50000);
        assert_eq!(args.skew, vec![0.8, 1.2]);
        assert_eq!(args.memory_budgets, vec![4096, 65536]);
        assert_eq!(args.top_k, 10);
        assert_eq!(args.trials, 2);
        assert_eq!(args.seed, 7);
        assert_eq!(args.output, "results/test.csv");
        assert_eq!(args.warmup_fraction, 0.05);
        assert_eq!(args.format, "csv");
    }

    #[test]
    fn missing_required_list_args_fail_to_parse() {
        // --cardinality is required; omitting it must be a parse error, not an
        // empty Vec that silently produces a zero-row sweep.
        let result = Cli::try_parse_from([
            "bench",
            "sweep",
            "--stream-length", "50000",
            "--skew", "0.8,1.2",
            "--memory-budgets", "4096,65536",
            "--output", "results/test.csv",
        ]);

        assert!(result.is_err(), "expected parse error when --cardinality is omitted");
    }
}
