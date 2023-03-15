use std::path::Path;

use serde::Deserialize;

#[derive(Debug, Deserialize)]
pub struct Settings {
    pub n_particles: usize,
    /// Number of grid cells along one axis for mass grid.
    pub n_grid: usize,
    /// Number of threads.
    pub n_threads: usize,
    pub seed: Option<u64>,
}

pub fn read_config() -> Settings {
    let path = Path::new("parallel/config/config.toml");
    let path = if path.exists() {
        path
    } else {
        Path::new("config/config.toml")
    };
    let config = config::Config::builder()
        .set_default("seed", None::<u64>)
        .expect("Impossible default value")
        .add_source(config::File::with_name(path.to_str().unwrap()))
        .add_source(config::Environment::with_prefix("MASS"))
        .build()
        .unwrap();

    let settings: Settings = config.try_deserialize().unwrap();
    settings
}
