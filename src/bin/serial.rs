#![feature(float_next_up_down)]
#![feature(test)]

use mass_assignment::{
    coordinates::{SpaceCoordinate, grid_coordinate_from},
    DIM, MAX, MIN
};

use ndarray::{Array, Array2, Dim};
use rand::Rng;

type MassGrid = Array<i32, Dim<[usize; DIM]>>;
type ParticleArray = Array2<SpaceCoordinate>;

// region Constants
/// Total number of particles in simulation.
pub const N_PARTICLES: usize = 1024;
/// Number of grid cells for mass grid.
pub const N_GRID: usize = 16;
// endregion

fn main() {
    let particles = generate_particles::<N_PARTICLES>();
    let mut mass_grid = MassGrid::zeros([N_GRID; DIM]);
    assign_masses::<N_GRID>(&particles, &mut mass_grid);

    // dbg!(&mass_grid);
    let total: i32 = mass_grid.iter().sum();
    dbg!(total);
}

/// Generate random particles. Particles are layed out as a simple array with shape (N_PARTICLE, DIM)
/// that describe coordinates of the particle.
fn generate_particles<const N_PARTICLES: usize>() -> ParticleArray {
    let mut particles = ParticleArray::zeros([N_PARTICLES, DIM]);
    let mut rng = rand::thread_rng();
    for particle_coord in particles.iter_mut() {
        *particle_coord = rng.gen_range(MIN..MAX).into();
    }
    particles
}

/// Assign masses according to nearest grid point algorithm.
fn assign_masses<const N_GRID: usize>(particles: &ParticleArray, mass_grid: &mut MassGrid) {
    for space_coords in particles.outer_iter() {
        let grid_coords = [
            grid_coordinate_from::<N_GRID>(space_coords[0]),
            grid_coordinate_from::<N_GRID>(space_coords[1]),
        ];
        mass_grid[grid_coords] += 1;
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use ndarray::array;
    extern crate test;
    use test::Bencher;

    #[test]
    fn test_mass_assignment() {
        // Warning! Only passes when DIM=2;
        let particles = array![
            [MIN, MIN],
            [MIN, MIN],
            [MAX.next_down(), MAX.next_down()],
            [MIN, (MAX.next_down() + MIN) / 2.],
            [(MAX + MIN) / 2., (MAX.next_down() + MIN) / 2.],
        ];

        const N_GRID: usize = 4;
        let mut mass_grid = MassGrid::zeros([N_GRID; DIM]);
        assign_masses::<N_GRID>(&particles, &mut mass_grid);
        let mass_grid_precalculated = array![
            [2, 1, 0, 0],
            [0, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
        ];
        assert_eq!(mass_grid, mass_grid_precalculated);
    }

    #[bench]
    fn bench_mass_assignment(b: &mut Bencher) {
        // Optionally include some setup
        const N_PARTICLES: usize= 1024;
        const N_GRID: usize = 64;
        let particles = generate_particles::<N_PARTICLES>();
        let mut mass_grid = MassGrid::zeros([N_GRID; DIM]);

        b.iter(|| {
            assign_masses::<N_GRID>(&particles, &mut mass_grid)
        });
    }
}
