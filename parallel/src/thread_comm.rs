use itertools::izip;
use lockfree::channel::mpsc::{Receiver, Sender};
use lockfree::channel::{mpsc, RecvErr};

use crate::coordinates::GridIndex;
use crate::{MassEntry, MassSlab};

// #[derive(Debug)]
// pub struct SlabMessage {
//     // Rank of thread that sent the message.
//     pub sent_by: usize,
//     pub slab_index: GridIndex,
//     pub slab: MassSlab,
// }

#[derive(Debug)]
pub enum SlabMessage {
    Msg {
        // Rank of thread that sent the message.
        sent_by: usize,
        slab_index: GridIndex,
        slab: MassSlab,
    },
    Ack {
        slab: MassSlab,
    },
}

#[derive(Debug)]
pub struct SlabChannel {
    pub rx: Receiver<SlabMessage>,
    pub tx: Vec<Sender<SlabMessage>>,
}

#[derive(Debug)]
pub struct MassChannel {
    pub rx: Receiver<MassEntry>,
    pub tx: Vec<Sender<MassEntry>>,
}

#[derive(Debug)]
pub struct SyncChannel {
    pub rx: Receiver<bool>,
    pub tx: Vec<Sender<bool>>,
}

#[derive(Debug)]
pub struct ThreadComm {
    pub rank: usize,
    // Total number of threads.
    pub size: usize,
    // Channel for sending a slab of a mass grid.
    pub slab_channel: SlabChannel,
    // Channel for gathering total mass in rank 0.
    pub mass_channel: MassChannel,
    // Channel for synchronization task.
    pub sync_channel: SyncChannel,
}

impl ThreadComm {
    pub fn create_communicators(number: usize) -> Vec<ThreadComm> {
        let (slab_senders, slab_receivers): (Vec<_>, Vec<_>) =
            (0..number).map(|_| mpsc::create()).unzip();
        let (mass_senders, mass_receivers): (Vec<_>, Vec<_>) =
            (0..number).map(|_| mpsc::create()).unzip();
        let (sync_senders, sync_receivers): (Vec<_>, Vec<_>) =
            (0..number).map(|_| mpsc::create()).unzip();

        let mut communicators: Vec<ThreadComm> = vec![];
        for (i, (slab_receiver, mass_receiver, sync_receiver)) in
            izip!(slab_receivers, mass_receivers, sync_receivers).enumerate()
        {
            let comm = ThreadComm {
                rank: i,
                size: number,
                slab_channel: SlabChannel {
                    rx: slab_receiver,
                    tx: slab_senders.clone(),
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
    pub fn barrier(&mut self) -> Result<(), RecvErr> {
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
