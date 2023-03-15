use std::iter::once;
use std::ops::AddAssign;
use std::thread;
use std::time::Instant;

use itertools::Itertools;
use lockfree::channel::RecvErr;
use ndarray::s;
use parallel::config::{read_config, Config};
use parallel::coordinates::{get_chunk_size, get_hunk_size, hunk_index_from_grid_index};
use parallel::thread_comm::{BufferMessage, ThreadComm};
use parallel::{coordinates::grid_index_from_coordinate, MAX, MIN};
use parallel::{MassGrid, MassPencil, Particle};
use rand::rngs::StdRng;
use rayon::prelude::*;

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
    // Number of pencils per hunk.
    let hunk_size: usize = get_hunk_size(n_grid, n_threads);
    let mut communicators = ThreadComm::create_communicators(n_threads);
    let mut particles = generate_particles(n_particles, seed);
    particles.par_sort_unstable_by_key(|p| {
        let i = grid_index_from_coordinate(p[0], n_grid).min(n_grid - 1);
        let j = grid_index_from_coordinate(p[1], n_grid).min(n_grid - 1);
        (i, j)
    });

    let start = Instant::now();
    thread::scope(|s| {
        for (comm, p_local) in communicators
            .iter_mut()
            .zip(particles.chunks_mut(chunk_size).chain(once(&mut [][..])))
        {
            s.spawn(move || {
                let mut mass_grid = MassGrid::zeros([hunk_size, n_grid]);
                // p_local.sort_unstable_by(|p1, p2| p1[0].total_cmp(&p2[0]));
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
    assert!(is_sorted(particles, n_grid));
    const MAX_PARTICLES_PROCESSED: usize = 1024;
    const MAX_BUFFERS: usize = 4;

    // Find pencil boundaries in sorted particles array.
    let mut pencil_boundaries = vec![];
    let mut max_grid_idx = 0;
    pencil_boundaries.push(0);
    for (i, &[x, _]) in particles.iter().enumerate() {
        let x_grid_index = grid_index_from_coordinate(x, n_grid).min(n_grid - 1);
        if x_grid_index > max_grid_idx {
            max_grid_idx = x_grid_index;
            pencil_boundaries.push(i);
        }
    }
    if !particles.is_empty() {
        pencil_boundaries.push(particles.len());
    }

    let hunk_size = get_hunk_size(n_grid, comm.size);
    let mut buffers = vec![MassPencil::zeros(n_grid); MAX_BUFFERS];

    // Process particles pencil by pencil.
    for (pencil_min, pencil_max) in pencil_boundaries.into_iter().tuple_windows() {
        let pencil_index =
            grid_index_from_coordinate(particles[pencil_min][0], n_grid).min(n_grid - 1);
        let hunk_index = hunk_index_from_grid_index(pencil_index, n_grid, comm.size);

        let mut local_buffer = loop {
            match buffers.pop() {
                Some(buffer) => break buffer,
                None => {
                    process_received_buffers(&mut buffers, mass_grid, n_grid, hunk_size, true, comm)
                }
            }
        };
        local_buffer.fill(0);

        // Process particles in pencil by chunks.
        for chunk in particles[pencil_min..pencil_max].chunks(MAX_PARTICLES_PROCESSED) {
            // Process particles in chunk.
            for &[_, y] in chunk {
                let y_grid_index = grid_index_from_coordinate(y, n_grid).min(n_grid - 1);
                local_buffer[y_grid_index] += 1;
            }

            process_received_buffers(&mut buffers, mass_grid, n_grid, hunk_size, false, comm)
        }

        // Keep if local.
        if hunk_index == comm.rank {
            let local_pencil_index = pencil_index - hunk_index * hunk_size;
            mass_grid
                .slice_mut(s![local_pencil_index, ..])
                .add_assign(&local_buffer);
            // Put the buffer back to be used in the next iteration.
            buffers.push(local_buffer);
        // Send if foreign.
        } else {
            comm.buffer_channel.tx[hunk_index]
                .send(BufferMessage::Msg {
                    sent_by: comm.rank,
                    pencil_index,
                    buffer: local_buffer,
                })
                .unwrap();
        }
    }

    // Synchronize threads.
    {
        // Make sure I get all my buffers back before sending "finished" signal.
        // This makes sure that no other thread gets my "finished" signal while it's still processing buffers sent by me.
        while buffers.len() < MAX_BUFFERS {
            process_received_buffers(&mut buffers, mass_grid, n_grid, hunk_size, false, comm);
        }

        // Send "finished" signal.
        for receiver in &comm.sync_channel.tx {
            receiver.send(true).unwrap();
        }

        // Receive "finished" signal and process incoming buffers.
        for _ in 0..comm.size {
            loop {
                // Wait for sync messages.
                match comm.sync_channel.rx.recv() {
                    Ok(_msg) => {
                        break;
                    }
                    Err(RecvErr::NoMessage) => {}
                    Err(RecvErr::NoSender) => panic!("Unexpected disconnect"),
                }

                // Process incoming buffers.
                process_received_buffers(&mut buffers, mass_grid, n_grid, hunk_size, false, comm);
            }
        }
    }
}

fn process_received_buffers(
    local_buffers: &mut Vec<MassPencil>,
    mass_grid: &mut MassGrid,
    n_grid: usize,
    hunk_size: usize,
    break_early: bool,
    comm: &mut ThreadComm,
) {
    while let Ok(msg) = comm.buffer_channel.rx.recv() {
        match msg {
            BufferMessage::Msg {
                sent_by,
                pencil_index,
                buffer: foreign_buffer,
            } => {
                let hunk_index = hunk_index_from_grid_index(pencil_index, n_grid, comm.size);
                let local_pencil_index = pencil_index - hunk_index * hunk_size;
                mass_grid
                    .slice_mut(s![local_pencil_index, ..])
                    .add_assign(&foreign_buffer);
                comm.buffer_channel.tx[sent_by]
                    .send(BufferMessage::Ack {
                        buffer: foreign_buffer,
                    })
                    .expect("Rank {rank} disconnected");
            }
            BufferMessage::Ack { buffer } => {
                local_buffers.push(buffer);
                if break_early {
                    break;
                };
            }
        }
    }
}

fn is_sorted(particles: &[[f32; 2]], n_grid: usize) -> bool {
    particles.windows(2).all(|win| {
        let i0 = grid_index_from_coordinate(win[0][0], n_grid).min(n_grid - 1);
        let j0 = grid_index_from_coordinate(win[0][1], n_grid).min(n_grid - 1);
        let i1 = grid_index_from_coordinate(win[1][0], n_grid).min(n_grid - 1);
        let j1 = grid_index_from_coordinate(win[1][1], n_grid).min(n_grid - 1);

        (i0, j0) <= (i1, j1)
    })
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
        // Number of pencils per hunk.
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

                    println!("rank {}, mass_grid = {}", comm.rank, &mass_grid);
                    let mass_grid_precalculated = match comm.rank {
                        0 => array![[2, 0, 1, 0]],
                        1 => array![[0, 0, 0, 0]],
                        2 => array![[0, 0, 1, 0]],
                        3 => array![[0, 0, 0, 1]],
                        _ => panic!("There shouldn't be more then {n_threads} threads."),
                    };
                    assert_eq!(
                        mass_grid, mass_grid_precalculated,
                        "offending thread: {}",
                        comm.rank
                    );
                });
            }
        });
    }
}
