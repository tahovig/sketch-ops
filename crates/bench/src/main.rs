mod cli;
mod metrics;
mod report;
mod sweep;

use clap::Parser;

use cli::{Cli, Command};

fn main() {
    let cli = Cli::parse();
    match cli.command {
        Command::Sweep(args) => {
            if !(0.0..1.0).contains(&args.warmup_fraction) {
                eprintln!(
                    "error: --warmup-fraction must be in [0.0, 1.0), got {}",
                    args.warmup_fraction
                );
                std::process::exit(1);
            }

            if args.format != "csv" && args.format != "json" {
                eprintln!(
                    "error: --format must be one of \"csv\" or \"json\", got \"{}\"",
                    args.format
                );
                std::process::exit(1);
            }

            let results = sweep::run_sweep(&args);

            let output_path = std::path::Path::new(&args.output);
            if let Some(parent) = output_path.parent() {
                if !parent.as_os_str().is_empty() {
                    std::fs::create_dir_all(parent).expect("failed to create output directory");
                }
            }

            match args.format.as_str() {
                "csv" => report::write_csv(output_path, &results).expect("failed to write CSV output"),
                "json" => report::write_json(output_path, &results).expect("failed to write JSON output"),
                _ => unreachable!("format validated above"),
            }

            println!("wrote {} rows to {}", results.len(), args.output);
        }
    }
}
