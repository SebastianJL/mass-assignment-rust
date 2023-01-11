use std::ops::AddAssign;
use std::thread;
use std::time::Instant;

use itertools::izip;
use itertools::Itertools;
use lockfree::channel::mpsc::{Receiver, Sender};
use lockfree::channel::{mpsc, RecvErr};
use ndarray::s;
use ndarray::Array1;
use ndarray::{Array, Dim};
use parallel::coordinates::{get_chunk_size, get_hunk_size, hunk_index_from_grid_index};
use parallel::{
    coordinates::{grid_index_from_coordinate, GridIndex, SpaceCoordinate},
    DIM, MAX, MIN,
};
use rand::rngs::StdRng;

type MassEntry = u32;
type MassGrid = Array<MassEntry, Dim<[usize; DIM]>>;
// Single slab of mass grid plus its grid index.
type MassSlab = Array1<MassEntry>;
type Particle = [SpaceCoordinate; 2];

#[derive(Debug)]
struct ThreadComm {
    rank: usize,
    // Total number of threads.
    size: usize,
    // Channel for sending a slab of a mass grid.
    slab_channel: SlabChannel,
    // Channel for gathering total mass in rank 0.
    mass_channel: MassChannel,
    // Channel for synchronization task.
    sync_channel: SyncChannel,
}

#[derive(Debug)]
struct SlabChannel {
    rx: Receiver<(GridIndex, MassSlab)>,
    tx: Vec<Sender<(GridIndex, MassSlab)>>,
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
        let (slab_sender, index_receivers): (Vec<_>, Vec<_>) =
            (0..number).map(|_| mpsc::create()).unzip();
        let (mass_senders, mass_receivers): (Vec<_>, Vec<_>) =
            (0..number).map(|_| mpsc::create()).unzip();
        let (sync_senders, sync_receivers): (Vec<_>, Vec<_>) =
            (0..number).map(|_| mpsc::create()).unzip();

        let mut communicators: Vec<ThreadComm> = vec![];
        for (i, (slab_receiver, mass_receiver, sync_receiver)) in
            izip!(index_receivers, mass_receivers, sync_receivers).enumerate()
        {
            let comm = ThreadComm {
                rank: i,
                size: number,
                slab_channel: SlabChannel {
                    rx: slab_receiver,
                    tx: slab_sender.clone(),
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

    /// Blocking barrier for all threads in ThreadComm.
    ///
    /// # Returns
    /// Returns Ok(_) if successful or Err(RecvErr::NoSender) if one of the connections got disconnected.
    fn barrier(&mut self) -> Result<(), RecvErr> {
        // Send signal.
        for receiver in &self.sync_channel.tx {
            receiver.send(true).unwrap();
        }

        // Receive signal.
        for _ in 0..self.size {
            loop {
                match self.sync_channel.rx.recv() {
                    Ok(_msg) => {
                        break;
                    }
                    Err(RecvErr::NoMessage) => {}
                    Err(RecvErr::NoSender) => return Err(RecvErr::NoSender),
                }
            }
        }

        Ok(())
    }
}

fn main() {
    let start = Instant::now();

    /// Total number of particles in simulation.
    const N_PARTICLES: usize = 1024usize.pow(2);
    dbg!(N_PARTICLES);
    /// Number of grid cells along one axis for mass grid.
    const N_GRID: usize = 1024;
    dbg!(N_GRID);
    /// Number of threads.
    const N_THREADS: usize = 4;
    /// Number of hunks. A hunk is a collection of slabs. I.e a hunk of a 2d grid [N, N] is [HUNK_SIZE, N].
    const N_HUNKS: usize = N_THREADS;

    // Number of particles per thread.
    const CHUNK_SIZE: usize = get_chunk_size(N_PARTICLES, N_THREADS);
    // Number of slabs per hunk.
    const HUNK_SIZE: usize = get_hunk_size(N_GRID, N_HUNKS);
    let mut communicators = ThreadComm::create_communicators(N_THREADS);
    let mut particles = generate_particles(N_PARTICLES);
    thread::scope(|s| {
        for (comm, p_local) in communicators
            .iter_mut()
            .zip(particles.chunks_mut(CHUNK_SIZE))
        {
            s.spawn(move || {
                let mut mass_grid = MassGrid::zeros([HUNK_SIZE, N_GRID]);
                p_local.sort_unstable_by(|p1, p2| p1[0].total_cmp(&p2[0]));
                assign_masses(p_local, &mut mass_grid, N_GRID, N_HUNKS, comm);

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
                    assert_eq!(total_mass as usize, N_PARTICLES);
                    dbg!(total_mass);
                }
            });
        }
    });

    let runtime = start.elapsed();
    dbg!(runtime);
}

/// Generate random particles. Particles are layed out as a simple array with shape (`N_PARTICLES`, DIM)
/// that describe coordinates of the particle.
fn generate_particles(n_particles: usize) -> Vec<Particle> {
    let mut rng = <StdRng as rand::SeedableRng>::seed_from_u64(42);
    // let mut rng = rand::thread_rng();
    (0..n_particles)
        .map(|_| {
            [
                rand::Rng::gen_range(&mut rng, MIN..MAX),
                rand::Rng::gen_range(&mut rng, MIN..MAX),
            ]
        })
        .collect()
}

/// Assign masses according to nearest grid point algorithm.
///
/// Expects particles to be sorted along first dimension.
fn assign_masses(
    particles: &[Particle],
    mass_grid: &mut MassGrid,
    n_grid: usize,
    n_hunks: usize,
    comm: &mut ThreadComm,
) {
    assert!(is_sorted(particles));

    // Find slab boundaries in sorted particles array.
    let mut slab_boundaries = vec![];
    let mut max_grid_idx = 0;
    slab_boundaries.push(0);
    for (i, &[x, _]) in particles.iter().enumerate() {
        let grid_index = grid_index_from_coordinate(x, n_grid).min(n_grid - 1);
        if grid_index > max_grid_idx {
            max_grid_idx = grid_index;
            slab_boundaries.push(i);
        }
    }
    slab_boundaries.push(particles.len());

    let hunk_size = get_hunk_size(n_grid, n_hunks);

    // Process particles slab by slab.
    for (i1, i2) in slab_boundaries.into_iter().tuple_windows() {
        let mut slab = MassSlab::zeros(n_grid);
        for &[_, y] in &particles[i1..i2] {
            let y_grid_index = grid_index_from_coordinate(y, n_grid).min(n_grid - 1);
            slab[y_grid_index] += 1;
        }
        let slab_index = grid_index_from_coordinate(particles[i1][0], n_grid).min(n_grid - 1);
        let hunk_index = hunk_index_from_grid_index(slab_index, n_grid, n_hunks);
        if hunk_index == comm.rank {
            let local_slab_index = slab_index - hunk_index * hunk_size;
            mass_grid
                .slice_mut(s![local_slab_index, ..])
                .add_assign(&slab);
        } else {
            comm.slab_channel.tx[hunk_index]
                .send((slab_index, slab))
                .unwrap();
        }
    }

    comm.barrier().unwrap();

    // Process particles sent from other threads.
    while let Ok((slab_index, slab)) = comm.slab_channel.rx.recv() {
        let hunk_index = hunk_index_from_grid_index(slab_index, n_grid, n_hunks);
        let local_slab_index = slab_index - hunk_index * hunk_size;
        mass_grid
            .slice_mut(s![local_slab_index, ..])
            .add_assign(&slab);
    }
}

fn is_sorted(particles: &[[f32; 2]]) -> bool {
    particles.windows(2).all(|w| w[0][0] <= w[1][0])
}

#[cfg(test)]
mod test {

    use ndarray::array;

    use super::*;

    #[test]
    fn test_mass_assignment() {
        // Warning! Only passes when DIM=2;
        let mut particles = vec![
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
        particles.sort_unstable_by(|p1, p2| p1[0].total_cmp(&p2[0]));
        assign_masses(
            &particles,
            &mut mass_grid,
            N_GRID,
            N_THREADS,
            &mut communicators[0],
        );
        let mass_grid_precalculated =
            array![[2, 0, 1, 0], [0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1],];
        assert_eq!(mass_grid, mass_grid_precalculated);
    }

    #[test]
    fn test_mass_assignment_multi_threaded() {
        // Warning! Only passes when DIM=2;
        let mut particles = vec![
            [MIN, MIN],
            [MIN, MIN],
            [MAX, MAX],
            [MIN, (MAX + MIN) / 2.],
            [(MAX + MIN) / 2., (MAX + MIN) / 2.],
        ];

        let n_particles = particles.len();
        const N_GRID: usize = 4;
        const N_THREADS: usize = 4;
        const N_HUNKS: usize = N_THREADS;

        // Number of particles per thread.
        let chunk_size: usize = get_chunk_size(n_particles, N_THREADS);
        // Number of slabs per hunk.
        const HUNK_SIZE: usize = get_hunk_size(N_GRID, N_HUNKS);
        let mut communicators = ThreadComm::create_communicators(N_THREADS);
        thread::scope(|s| {
            for (comm, p_local) in communicators
                .iter_mut()
                .zip(particles.chunks_mut(chunk_size))
            {
                s.spawn(move || {
                    let mut mass_grid = MassGrid::zeros([HUNK_SIZE, N_GRID]);
                    p_local.sort_unstable_by(|p1, p2| p1[0].total_cmp(&p2[0]));
                    assign_masses(p_local, &mut mass_grid, N_GRID, N_HUNKS, comm);

                    let mass_grid_precalculated = match comm.rank {
                        // array![[2, 0, 1, 0], [0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1],];
                        0 => array![2, 0, 1, 0],
                        1 => array![0, 0, 0, 0],
                        2 => array![0, 0, 1, 0],
                        3 => array![0, 0, 0, 1],
                        _ => panic!("There sholdn't be more then {N_THREADS} threads."),
                    };
                    // assert_eq!(mass_grid, mass_grid_precalculated);
                });
            }
        });
    }
}
