use coordinates::SpaceCoordinate;
use ndarray::{Array, Array1, Dim};

pub mod config;
pub mod coordinates;
pub mod thread_comm;

/// Choose between f32 and f64 to change precision of floating point numbers.
pub type Float = f32;

/// Min value of space in simulation.
pub const MIN: Float = -0.5;
/// Max value of space in simulation.
pub const MAX: Float = 0.5;
/// Number of dimensions of space. Only 2 supported.
pub const DIM: usize = 2;

pub type MassEntry = u32;
pub type MassGrid = Array<MassEntry, Dim<[usize; DIM]>>;
// Single pencil of mass grid plus its grid index.
pub type MassPencil = Array1<MassEntry>;
pub type Particle = [SpaceCoordinate; DIM];
