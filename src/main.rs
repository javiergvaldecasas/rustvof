//! VOF2D - 2D Volume of Fluid solver for wave simulation
//!
//! Usage:
//!   vof2d run <config.toml>


use clap::{Parser, Subcommand};
use std::path::PathBuf;

use rustvof::{Config, Simulation};

#[derive(Parser)]
#[command(name = "rustvof")]
#[command(about = "2D VOF solver for wave simulation")]
#[command(version)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Run simulation from configuration file
    Run {
        /// Path to configuration file (TOML)
        config: PathBuf,
        
        /// Enable GPU acceleration (Metal on Apple Silicon, Vulkan on others)
        #[arg(long)]
        gpu: bool,
    },

}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
        .format_timestamp_millis()
        .init();

    let cli = Cli::parse();

    match cli.command {
        Commands::Run { config, gpu } => {
            log::info!("Loading configuration from {:?}", config);
            let config = Config::from_file(&config)?;
            let mut sim = Simulation::new(config);
            

            
            #[cfg(not(feature = "gpu"))]
            if gpu {
                log::warn!("GPU feature not compiled. Rebuild with: cargo build --features gpu");
            }
            
            sim.run()?;
        }

    }

    Ok(())
}
