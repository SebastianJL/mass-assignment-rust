use mass_assignment::{DIM, MAX, MIN, N_GRID, N_PARTICLES, coordinates::{SpaceCoordinate, GridCoordinate, FromSpaceCoordinate}};

use ndarray::{Array2, Array3};
use rand::Rng;

fn main() {
    let particles = generate_particles();
    let mut mass_grid = Array3::<i32>::zeros([N_GRID as usize, N_GRID as usize, N_GRID as usize]);

    assign_masses(&particles, &mut mass_grid);

    // dbg!(&mass_grid);
    let total: i32 = mass_grid.iter().sum();
    dbg!(total);
}

/// Generate random particles. Particles are layed out as a simple array with shape (N_PARTICLE, DIM)
/// that describe coordinates of the particle.
fn generate_particles() -> Array2<SpaceCoordinate> {
    let mut particles = Array2::<SpaceCoordinate>::zeros([N_PARTICLES, DIM]);
    let mut rng = rand::thread_rng();
    for particle_coord in particles.iter_mut() {
        *particle_coord = rng.gen_range(MIN..MAX).into();
    }
    particles
}

/// Assign masses according to nearest grid point algorithm.
fn assign_masses(particles: &Array2<SpaceCoordinate>, mass_grid: &mut Array3<i32>) {
    for space_coords in particles.outer_iter() {
        let grid_coords = [
            GridCoordinate::from_space_coordinate(space_coords[0]),
            GridCoordinate::from_space_coordinate(space_coords[1]),
            GridCoordinate::from_space_coordinate(space_coords[2]),
            ];
        mass_grid[grid_coords] += 1;
    }
}
