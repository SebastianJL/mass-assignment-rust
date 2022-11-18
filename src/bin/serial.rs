use mass_assignment::{DIM, MAX, MIN, N_GRID, N_PARTICLES, coordinates::{SpaceCoordinate, GridCoordinate, FromSpaceCoordinate}};

use mass_assignment::{
    coordinates::{FromSpaceCoordinate, GridCoordinate, SpaceCoordinate},
    DIM, MAX, MIN, N_GRID, N_PARTICLES,
};

use ndarray::{Array, Array2, Dim};
use rand::Rng;

type MassGrid = Array<i32, Dim<[usize; DIM]>>;
type ParticleArray = Array2<SpaceCoordinate>;

fn main() {
    let particles = generate_particles();
    let mut mass_grid = MassGrid::zeros([N_GRID; DIM]);
    assign_masses(&particles, &mut mass_grid);

    // dbg!(&mass_grid);
    let total: i32 = mass_grid.iter().sum();
    dbg!(total);
}

/// Generate random particles. Particles are layed out as a simple array with shape (N_PARTICLE, DIM)
/// that describe coordinates of the particle.
fn generate_particles() -> ParticleArray {
    let mut particles = ParticleArray::zeros([N_PARTICLES, DIM]);
    let mut rng = rand::thread_rng();
    for particle_coord in particles.iter_mut() {
        *particle_coord = rng.gen_range(MIN..MAX).into();
    }
    particles
}

/// Assign masses according to nearest grid point algorithm.
fn assign_masses(particles: &ParticleArray, mass_grid: &mut MassGrid) {
    for space_coords in particles.outer_iter() {
        // Todo: Maybe replace with macro for ndim support.
        let grid_coords: [GridCoordinate; DIM] = space_coords
            .into_iter()
            .map(|coord| GridCoordinate::from_space_coordinate(*coord))
            .collect::<Vec<GridCoordinate>>()
            .try_into()
            .unwrap();
        mass_grid[grid_coords] += 1;
    }
}
