use std::path::Path;

use serde::Deserialize;

#[derive(Debug, Deserialize)]
pub struct Config {
    pub n_particles: usize,
    /// Number of grid cells along one axis for mass grid.
    pub n_grid: usize,
    /// Number of threads.
    pub n_threads: usize,
    pub seed: u64,
}

pub fn read_config() -> Config {
    let path = Path::new("parallel/config/config.toml");
    let path = if path.exists() {
        path
    } else {
        Path::new("config/config.toml")
    };
    let config = config::Config::builder()
        .add_source(config::File::with_name(path.to_str().unwrap()))
        .add_source(config::Environment::with_prefix("MASS"))
        .build()
        .unwrap();

    let settings: Config = config.try_deserialize().unwrap();
    settings
}
