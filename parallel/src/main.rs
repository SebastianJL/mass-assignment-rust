use std::iter::once;
use std::ops::AddAssign;
use std::thread;
use std::time::Instant;

use lockfree::channel::RecvErr;
use ndarray::s;
use rand::rngs::StdRng;
use rayon::prelude::*;

use parallel::config::{read_config, Config};
use parallel::coordinates::{get_chunk_size, get_hunk_size, hunk_index_from_grid_index, GridIndex};
use parallel::thread_comm::{SlabMessage, ThreadComm};
use parallel::{coordinates::grid_index_from_coordinate, MAX, MIN};
use parallel::{MassGrid, MassSlab, Particle};

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
    const MAX_BUFFERS: usize = 4;

    let hunk_size = get_hunk_size(n_grid, comm.size);
    let mut buffers = vec![MassSlab::zeros(n_grid); MAX_BUFFERS];

    let mut i = grid_index_from_coordinate(particles[0][0], n_grid).min(n_grid - 1);
    let mut buffer = buffers.pop().unwrap();
    for &[x, y] in particles {
        let new_i = grid_index_from_coordinate(x, n_grid).min(n_grid - 1);
        let j = grid_index_from_coordinate(y, n_grid).min(n_grid - 1);
        if new_i > i {
            i = new_i;

            flush_or_assign(buffer, &mut buffers, i, mass_grid, n_grid, hunk_size, comm);

            // Get new buffer
            {
                buffer = loop {
                    match buffers.pop() {
                        Some(slab) => break slab,
                        None => process_received_buffers(
                            &mut buffers,
                            mass_grid,
                            n_grid,
                            hunk_size,
                            true,
                            comm,
                        ),
                    }
                };
                buffer.fill(0);
            }
        }

        let mass = 1;
        buffer[j] += mass;

        process_received_buffers(&mut buffers, mass_grid, n_grid, hunk_size, false, comm)
    }
    flush_or_assign(buffer, &mut buffers, i, mass_grid, n_grid, hunk_size, comm);

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

                // Process incoming slabs.
                process_received_buffers(&mut buffers, mass_grid, n_grid, hunk_size, false, comm);
            }
        }
    }
}

/// Flush the buffer to another thread or assign it to mass grid.
fn flush_or_assign(
    buffer: MassSlab,
    buffers: &mut Vec<MassSlab>,
    pencil_index: GridIndex,
    mass_grid: &mut MassGrid,
    n_grid: usize,
    hunk_size: usize,
    comm: &mut ThreadComm,
) {
    let hunk_index = hunk_index_from_grid_index(pencil_index, n_grid, comm.size);
    // Keep if local.
    if hunk_index == comm.rank {
        let local_pencil_index = pencil_index - hunk_index * hunk_size;
        mass_grid
            .slice_mut(s![local_pencil_index, ..])
            .add_assign(&buffer);
        // Put the buffer back to be used in the next iteration.
        buffers.push(buffer);
    // Send if foreign.
    } else {
        comm.slab_channel.tx[hunk_index]
            .send(SlabMessage::Msg {
                sent_by: comm.rank,
                pencil_index: pencil_index,
                slab: buffer,
            })
            .unwrap();
    }
}

fn process_received_buffers(
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
                pencil_index: slab_index,
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
