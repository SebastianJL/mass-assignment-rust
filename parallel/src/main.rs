use std::thread;
use std::time::Instant;

use itertools::izip;
use lockfree::channel::mpsc::{Receiver, Sender};
use lockfree::channel::{mpsc, RecvErr};
use parallel::coordinates::{get_hunk_size, hunk_index_from_grid_index, get_chunk_size};
use ndarray::{Array, Dim};

use parallel::{
    coordinates::{grid_index_from_coordinate, GridIndex, SpaceCoordinate},
    DIM, MAX, MIN,
};
use rand::rngs::StdRng;

// region Constants
/// Total number of particles in simulation.
pub const N_PARTICLES: usize = 1024;
/// Number of grid cells for mass grid.
pub const N_GRID: usize = 16;
/// Number of threads.
const N_THREADS: usize = 3;
/// Number of hunks. A hunk is a collection of slabs. I.e a hunk of a 2d grid [N, N] might be [2, N].
const N_HUNKS: usize = N_THREADS;
// endregion

type MassEntry = usize;
type MassGrid = Array<MassEntry, Dim<[usize; DIM]>>;
type Particle = [SpaceCoordinate; 2];

#[derive(Debug)]
struct ThreadComm {
    rank: usize,
    // Channel for sending indices where mass has to be assigned.
    index_channel: IndexChannel,
    // Channel for gathering total mass in rank 0.
    mass_channel: MassChannel,
    // Channel for synchronization task.
    sync_channel: SyncChannel,
}

#[derive(Debug)]
struct IndexChannel {
    rx: Receiver<(GridIndex, GridIndex)>,
    tx: Vec<Sender<(GridIndex, GridIndex)>>,
}

#[derive(Debug)]
struct MassChannel {
    rx: Receiver<MassEntry>,
    tx: Vec<Sender<MassEntry>>,
}

#[derive(Debug)]
struct SyncChannel {
    rx: Receiver<bool>,
    tx: Vec<Sender<bool>>,
}

impl ThreadComm {
    fn create_communicators(number: usize) -> Vec<ThreadComm> {
        let (index_senders, index_receivers): (Vec<_>, Vec<_>) =
            (0..number).map(|_| mpsc::create()).unzip();
        let (mass_senders, mass_receivers): (Vec<_>, Vec<_>) =
            (0..number).map(|_| mpsc::create()).unzip();
        let (sync_senders, sync_receivers): (Vec<_>, Vec<_>) =
            (0..number).map(|_| mpsc::create()).unzip();

        let mut communicators: Vec<ThreadComm> = vec![];
        for (i, (index_receiver, mass_receiver, sync_receiver)) in
            izip!(index_receivers, mass_receivers, sync_receivers).enumerate()
        {
            let comm = ThreadComm {
                rank: i,
                index_channel: IndexChannel {
                    rx: index_receiver,
                    tx: index_senders.clone(),
                },
                mass_channel: MassChannel {
                    rx: mass_receiver,
                    tx: mass_senders.clone(),
                },
                sync_channel: SyncChannel {
                    rx: sync_receiver,
                    tx: sync_senders.clone(),
                },
            };
            communicators.push(comm);
        }
        communicators
    }
}

fn main() {
    let start = Instant::now();

    // Number of particles per thread.
    const CHUNK_SIZE: usize = get_chunk_size(N_PARTICLES, N_THREADS);
    // Number of slabs per hunk.
    const HUNK_SIZE: usize = get_hunk_size(N_GRID, N_HUNKS);
    let mut communicators = ThreadComm::create_communicators(N_THREADS);
    let particles = generate_particles::<N_PARTICLES>();
    thread::scope(|s| {
        for (comm, p_local) in communicators.iter_mut().zip(particles.chunks(CHUNK_SIZE)) {
            s.spawn(move || {
                let mut mass_grid = MassGrid::default([HUNK_SIZE, N_GRID]);
                assign_masses::<N_GRID, N_HUNKS>(p_local, &mut mass_grid, comm);

                // Calculate local mass sum and send.
                if comm.rank != 0 {
                    comm.mass_channel.tx[0].send(mass_grid.sum()).unwrap();
                }

                // Receive mass sums and calculate global mass sum.
                if comm.rank == 0 {
                    let mut total_mass = mass_grid.sum();
                    for _ in 0..(N_THREADS - 1) {
                        loop {
                            match comm.mass_channel.rx.recv() {
                                Ok(mass) => {
                                    total_mass += mass;
                                    break;
                                }
                                Err(RecvErr::NoMessage) => {}
                                Err(RecvErr::NoSender) => panic!("Somehow got disconnected"),
                            }
                        }
                    }
                    dbg!(total_mass);
                    dbg!(p_local.len());
                }
            });
        }
    });

    let runtime = start.elapsed();
    dbg!(runtime);
}

/// Generate random particles. Particles are layed out as a simple array with shape (`N_PARTICLES`, DIM)
/// that describe coordinates of the particle.
fn generate_particles<const N_PARTICLES: usize>() -> Vec<Particle> {
    let mut rng = <StdRng as rand::SeedableRng>::seed_from_u64(42);
    // let mut rng = rand::thread_rng();
    (0..N_PARTICLES)
        .map(|_| {
            [
                rand::Rng::gen_range(&mut rng, MIN..MAX),
                rand::Rng::gen_range(&mut rng, MIN..MAX),
            ]
        })
        .collect()
}

/// Assign masses according to nearest grid point algorithm.
fn assign_masses<const N_GRID: usize, const N_HUNKS: usize>(
    particles: &[Particle],
    mass_grid: &mut MassGrid,
    comm: &mut ThreadComm,
) {
    let hunk_size = get_hunk_size(N_GRID, N_HUNKS);
    // Process particles belonging to own thread and send others to their threads.
    for space_coords in particles.iter() {
        let grid_indices = (
            grid_index_from_coordinate::<N_GRID>(space_coords[0]).min(N_GRID - 1),
            grid_index_from_coordinate::<N_GRID>(space_coords[1]).min(N_GRID - 1),
        );
        let hunk_index = hunk_index_from_grid_index::<N_GRID, N_HUNKS>(grid_indices.0);
        if hunk_index == comm.rank {
            let local_grid_indices = (grid_indices.0 - hunk_index * hunk_size, grid_indices.1);
            mass_grid[local_grid_indices] += 1;
        } else {
            comm.index_channel.tx[hunk_index]
                .send(grid_indices)
                .unwrap();
        }
    }

    // Send finished mass assignments.
    for receiver in &comm.sync_channel.tx {
        receiver.send(true).unwrap();
    }

    // Receive mass assignment finished.
    for _ in 0..N_THREADS {
        loop {
            match comm.sync_channel.rx.recv() {
                Ok(_msg) => {
                    break;
                }
                Err(RecvErr::NoMessage) => {}
                Err(RecvErr::NoSender) => panic!("Somehow got disconnected"),
            }
        }
    }

    // Process particles sent from other threads.
    while let Ok(grid_indices) = comm.index_channel.rx.recv() {
        let hunk_index = hunk_index_from_grid_index::<N_GRID, N_HUNKS>(grid_indices.0);
        let local_grid_indices = (grid_indices.0 - hunk_index * hunk_size, grid_indices.1);
        mass_grid[local_grid_indices] += 1;
    }
}

#[cfg(test)]
mod test {

    use ndarray::array;

    use super::*;

    #[test]
    fn test_mass_assignment() {
        // Warning! Only passes when DIM=2;
        let particles = vec![
            [MIN, MIN],
            [MIN, MIN],
            [MAX, MAX],
            [MIN, (MAX + MIN) / 2.],
            [(MAX + MIN) / 2., (MAX + MIN) / 2.],
        ];

        const N_GRID: usize = 4;
        const N_THREADS: usize = 1;
        let mut mass_grid = MassGrid::default([N_GRID; DIM]);
        let mut communicators = ThreadComm::create_communicators(N_THREADS);
        assign_masses::<N_GRID, N_THREADS>(&particles, &mut mass_grid, &mut communicators[0]);
        let mass_grid_precalculated =
            array![[2, 0, 1, 0], [0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1],];
        assert_eq!(mass_grid, mass_grid_precalculated);
    }
}
