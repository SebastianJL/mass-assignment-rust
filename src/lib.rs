#![feature(float_next_up_down)]

pub mod coordinates;

/// Choose between f32 and f64 to change precision of floating point numbers.
pub type Float = f32;

/// Min value of space in simulation.
pub const MIN: Float = -0.5;
/// Max value of space in simulation.
pub const MAX: Float = 0.5;
/// Number of dimensions of space.
pub const DIM: usize = 2;
