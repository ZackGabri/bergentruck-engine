use std::sync::atomic::Ordering;
use std::sync::{Arc, atomic::AtomicBool};
use std::{io, thread};

use shakmaty::{Chess, Position};

use super::SearchOptions;

use crate::history::MoveHistory;

pub fn stop_search(search_thread: &mut Option<SearchThread>) {
    if let Some(thread) = search_thread.take() {
        thread.stop_and_join();
    }
}

pub struct SearchThread {
    stop: Arc<AtomicBool>,
    handle: thread::JoinHandle<()>,
}

impl SearchThread {
    pub fn spawn(
        position: &Chess,
        history: &MoveHistory,
        options: SearchOptions,
    ) -> io::Result<Self> {
        let position = position.clone();
        let mut history = history.clone();

        let stop = Arc::clone(&options.stop);
        let thread_stop = Arc::clone(&stop);

        let infinite = options.infinite;

        let handle = thread::Builder::new()
            .name("search".into())
            .spawn(move || {
                let (best_move, _, _) = super::search(&position, Some(&mut history), options);

                // `go infinite` shouldn't print a move until `stop` is recieved, even if the search finishes
                if infinite {
                    while !thread_stop.load(Ordering::Relaxed) {
                        thread::park();
                    }
                }

                if let Some(m) = best_move {
                    println!("bestmove {}", m.to_uci(position.castles().mode()));
                }
            })?;

        Ok(Self { stop, handle })
    }

    pub fn stop_and_join(self) {
        self.stop.store(true, Ordering::Relaxed);
        self.handle.thread().unpark(); // wake it if it's parked after an `infinite` search
        let _ = self.handle.join();
    }

    pub fn is_running(&self) -> bool {
        !self.handle.is_finished()
    }
}
