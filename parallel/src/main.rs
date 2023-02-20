use std::iter::once;
use std::ops::AddAssign;
use std::thread;
use std::time::{Duration, Instant};

use itertools::Itertools;
use lockfree::channel::RecvErr;
use ndarray::s;
use parallel::config::{read_config, Config};
use parallel::coordinates::{get_chunk_size, get_hunk_size, hunk_index_from_grid_index};
use parallel::thread_comm::{SlabMessage, ThreadComm};
use parallel::{coordinates::grid_index_from_coordinate, MAX, MIN};
use parallel::{MassGrid, MassSlab, Particle};
use rand::rngs::StdRng;

#[cfg(feature = "dhat-heap")]
#[global_allocator]
static ALLOC: dhat::Alloc = dhat::Alloc;

fn main() {
    #[cfg(feature = "dhat-heap")]
    let _profiler = dhat::Profiler::new_heap();

    let Config {
        n_particles,
        n_grid,
        n_threads,
        seed,
    } = read_config();
    dbg!(seed);
    dbg!(n_particles);
    dbg!(n_grid);
    dbg!(n_threads);

    // Number of particles per thread.
    let chunk_size: usize = get_chunk_size(n_particles, n_threads);
    // Number of slabs per hunk.
    let hunk_size: usize = get_hunk_size(n_grid, n_threads);
    let mut communicators = ThreadComm::create_communicators(n_threads);
    let mut particles = generate_particles(n_particles, seed);

    let start = Instant::now();
    thread::scope(|s| {
        for (comm, p_local) in communicators
            .iter_mut()
            .zip(particles.chunks_mut(chunk_size).chain(once(&mut [][..])))
        {
            s.spawn(move || {
                let mut mass_grid = MassGrid::zeros([hunk_size, n_grid]);
                p_local.sort_unstable_by(|p1, p2| p1[0].total_cmp(&p2[0]));
                assign_masses(p_local, &mut mass_grid, n_grid, comm);

                // Calculate local mass sum and send.
                if comm.rank != 0 {
                    comm.mass_channel.tx[0].send(mass_grid.sum()).unwrap();
                }

                // Receive mass sums and calculate global mass sum.
                if comm.rank == 0 {
                    let mut total_mass = mass_grid.sum();
                    for _ in 0..(n_threads - 1) {
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
                    assert_eq!(total_mass as usize, n_particles);
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
fn generate_particles(n_particles: usize, seed: u64) -> Vec<Particle> {
    let mut rng = <StdRng as rand::SeedableRng>::seed_from_u64(seed);
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
    comm: &mut ThreadComm,
) {
    assert!(is_sorted(particles));
    const MAX_PARTICLES_PROCESSED: usize = 1024;
    const MAX_BUFFERS: usize = 8;

    // Find slab boundaries in sorted particles array.
    let mut slab_boundaries = vec![];
    let mut max_grid_idx = 0;
    slab_boundaries.push(0);
    for (i, &[x, _]) in particles.iter().enumerate() {
        let x_grid_index = grid_index_from_coordinate(x, n_grid).min(n_grid - 1);
        if x_grid_index > max_grid_idx {
            max_grid_idx = x_grid_index;
            slab_boundaries.push(i);
        }
    }
    if particles.len() > 0 {
        slab_boundaries.push(particles.len());
    }

    let hunk_size = get_hunk_size(n_grid, comm.size);
    let mut slab_buffers = vec![MassSlab::zeros(n_grid); MAX_BUFFERS];

    // Process particles slab by slab.
    for (slab_min, slab_max) in slab_boundaries.into_iter().tuple_windows() {
        let slab_index = grid_index_from_coordinate(particles[slab_min][0], n_grid).min(n_grid - 1);
        let hunk_index = hunk_index_from_grid_index(slab_index, n_grid, comm.size);

        let mut i = 0;
        let mut local_slab = loop {
            match slab_buffers.pop() {
                Some(slab) => break slab,
                None => {
                    // thread::sleep(Duration::from_millis(2));
                    // i += 1;
                    // if i > 10_000 {
                    //     i = 0;
                    //     dbg!(comm.rank);
                    // };
                    process_received_slabs(&mut slab_buffers, mass_grid, n_grid, hunk_size, true, comm)
                }
            }
        };
        local_slab.fill(0);

        // Process particles in slab by chunks.
        for chunk in particles[slab_min..slab_max].chunks(MAX_PARTICLES_PROCESSED) {
            // Process particles in chunk.
            for &[_, y] in chunk {
                let y_grid_index = grid_index_from_coordinate(y, n_grid).min(n_grid - 1);
                local_slab[y_grid_index] += 1;
            }

            // process_received_slabs(&mut slab_buffers, mass_grid, n_grid, hunk_size, false, comm)
        }

        // Keep if local.
        if hunk_index == comm.rank {
            let local_slab_index = slab_index - hunk_index * hunk_size;
            mass_grid
                .slice_mut(s![local_slab_index, ..])
                .add_assign(&local_slab);
            // Put the buffer back to be used in the next iteration.
            slab_buffers.push(local_slab);
        // Send if foreign.
        } else {
            comm.slab_channel.tx[hunk_index]
                .send(SlabMessage::Msg {
                    sent_by: comm.rank,
                    slab_index,
                    slab: local_slab,
                })
                .unwrap();
        }
    }
    process_received_slabs(&mut slab_buffers, mass_grid, n_grid, hunk_size, false, comm);

    println!("{} done", comm.rank);
    comm.barrier().unwrap();

    process_received_slabs(&mut slab_buffers, mass_grid, n_grid, hunk_size, false, comm);
}

fn process_received_slabs(
    local_buffers: &mut Vec<MassSlab>,
    mass_grid: &mut MassGrid,
    n_grid: usize,
    hunk_size: usize,
    break_early: bool,
    comm: &mut ThreadComm,
) {
    while let Ok(msg) = comm.slab_channel.rx.recv() {
        match msg {
            SlabMessage::Msg {
                sent_by,
                slab_index,
                slab: foreign_slab,
            } => {
                let hunk_index = hunk_index_from_grid_index(slab_index, n_grid, comm.size);
                let local_slab_index = slab_index - hunk_index * hunk_size;
                mass_grid
                    .slice_mut(s![local_slab_index, ..])
                    .add_assign(&foreign_slab);
                comm.slab_channel.tx[sent_by]
                    .send(SlabMessage::Ack { slab: foreign_slab })
                    .expect("Rank {rank} disconnected");
            }
            SlabMessage::Ack { slab } => {
                local_buffers.push(slab);
                if break_early {
                    break;
                };
            }
        }
    }
}

fn is_sorted(particles: &[[f32; 2]]) -> bool {
    particles.windows(2).all(|w| w[0][0] <= w[1][0])
}

#[cfg(test)]
mod test {

    use std::{iter::once, vec};

    use ndarray::array;
    use parallel::DIM;

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
        assign_masses(&particles, &mut mass_grid, N_GRID, &mut communicators[0]);
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
        let n_threads: usize = 4;

        // Number of particles per thread.
        let chunk_size: usize = get_chunk_size(n_particles, n_threads);
        // Number of slabs per hunk.
        let hunk_size: usize = get_hunk_size(N_GRID, n_threads);
        let mut communicators = ThreadComm::create_communicators(n_threads);

        thread::scope(|s| {
            for (comm, p_local) in communicators
                .iter_mut()
                .zip(particles.chunks_mut(chunk_size).chain(once(&mut [][..])))
            {
                s.spawn(move || {
                    let mut mass_grid = MassGrid::zeros([hunk_size, N_GRID]);
                    p_local.sort_unstable_by(|p1, p2| p1[0].total_cmp(&p2[0]));
                    assign_masses(p_local, &mut mass_grid, N_GRID, comm);

                    let mass_grid_precalculated = match comm.rank {
                        0 => array![[2, 0, 1, 0]],
                        1 => array![[0, 0, 0, 0]],
                        2 => array![[0, 0, 1, 0]],
                        3 => array![[0, 0, 0, 1]],
                        _ => panic!("There shouldn't be more then {n_threads} threads."),
                    };
                    assert_eq!(mass_grid, mass_grid_precalculated);
                });
            }
        });
    }
}
