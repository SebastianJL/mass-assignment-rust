#![feature(float_next_up_down)]

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
    assign_masses::<N_GRID>(&particles, &mut mass_grid);

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
fn assign_masses<const N_GRID: usize>(particles: &ParticleArray, mass_grid: &mut MassGrid) {
    for space_coords in particles.outer_iter() {
        let grid_coords = [
            GridCoordinate::from_space_coordinate::<N_GRID>(space_coords[0]),
            GridCoordinate::from_space_coordinate::<N_GRID>(space_coords[1]),
        ];
        mass_grid[grid_coords] += 1;
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use ndarray::array;

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
}
