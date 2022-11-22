#![feature(float_next_up_down)]
#![feature(test)]

use std::sync::atomic::{AtomicI32, Ordering};

use ndarray::{Array, Array2, Dim};
use ndarray::parallel::prelude::*;
use rand::Rng;

use mass_assignment::{
    coordinates::{grid_coordinate_from, SpaceCoordinate},
    DIM, MAX, MIN,
};

type MassGrid = Array<AtomicI32, Dim<[usize; DIM]>>;
type ParticleArray = Array2<SpaceCoordinate>;

// region Constants
/// Total number of particles in simulation.
pub const N_PARTICLES: usize = 1024;
/// Number of grid cells for mass grid.
pub const N_GRID: usize = 16;
// endregion

fn main() {
    let particles = generate_particles::<N_PARTICLES>();
    let mut mass_grid = MassGrid::default([N_GRID; DIM]);
    assign_masses::<N_GRID>(&particles, &mut mass_grid);
    dbg!(&mass_grid);

    let total: i32 = mass_grid.par_iter().map(|e| e.load(Ordering::SeqCst)).sum();
    dbg!(total);
}

/// Generate random particles. Particles are layed out as a simple array with shape (N_PARTICLE, DIM)
/// that describe coordinates of the particle.
fn generate_particles<const N_PARTICLES: usize>() -> ParticleArray {
    let mut particles = ParticleArray::zeros([N_PARTICLES, DIM]);
    particles.par_map_inplace(|particle_coord| {
        let mut rng = rand::thread_rng();
        *particle_coord = rng.gen_range(MIN..MAX).into();
    });
    particles
}

/// Assign masses according to nearest grid point algorithm.
fn assign_masses<const N_GRID: usize>(particles: &ParticleArray, mass_grid: &mut MassGrid) {
    particles.outer_iter().into_par_iter().for_each(|space_coords| {
        let grid_coords = [
            grid_coordinate_from::<N_GRID>(space_coords[0]),
            grid_coordinate_from::<N_GRID>(space_coords[1]),
        ];
        mass_grid[grid_coords].fetch_add(1, Ordering::SeqCst);
    });
}

#[cfg(test)]
mod test {
    use test::Bencher;

    use ndarray::array;

    use super::*;

    extern crate test;

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
        let mut mass_grid = MassGrid::default([N_GRID; DIM]);
        assign_masses::<N_GRID>(&particles, &mut mass_grid);
        let mass_grid_precalculated = array![
            [2, 1, 0, 0],
            [0, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
        ];
        assert_eq!(mass_grid.map(|e| e.load(Ordering::SeqCst)), mass_grid_precalculated);
    }

    #[bench]
    fn bench_particle_generation(b: &mut Bencher) {
        const N_PARTICLES: usize = 1024;

        b.iter(|| {
            generate_particles::<N_PARTICLES>()
        });
    }

    #[bench]
    fn bench_mass_assignment(b: &mut Bencher) {
        // Optionally include some setup
        const N_PARTICLES: usize = 1024;
        const N_GRID: usize = 64;
        let particles = generate_particles::<N_PARTICLES>();
        let mut mass_grid = MassGrid::default([N_GRID; DIM]);

        b.iter(|| {
            assign_masses::<N_GRID>(&particles, &mut mass_grid)
        });
    }
}
