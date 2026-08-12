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
            let results = sweep::run_sweep(&args);

            let output_path = std::path::Path::new(&args.output);
            if let Some(parent) = output_path.parent() {
                if !parent.as_os_str().is_empty() {
                    std::fs::create_dir_all(parent).expect("failed to create output directory");
                }
            }

            match args.format.as_str() {
                "json" => report::write_json(output_path, &results).expect("failed to write JSON output"),
                _ => report::write_csv(output_path, &results).expect("failed to write CSV output"),
            }

            println!("wrote {} rows to {}", results.len(), args.output);
        }
    }
}
