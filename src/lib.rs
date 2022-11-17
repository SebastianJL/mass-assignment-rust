#![feature(float_next_up_down)]

pub mod coordinates;

/// Choose between f32 and f64 to change precision of floating point numbers.
pub type Float = f32;

/// Min value of space in simulation.
pub const MIN: Float = -1.;
/// Max value of space in simulation.
pub const MAX: Float = 1.;
/// Number of dimensions of space.
pub const DIM: usize = 3;
/// Total number of particles in simulation.
pub const N_PARTICLES: usize = 1024;
/// Number of grid cells for mass grid.
pub const N_GRID: usize = 16;