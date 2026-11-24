use shakmaty::{Chess, Position};

use crate::transposition_table::TTHash;

const REPETITION_SIZE: usize = 2048;

#[derive(Clone, Debug)]
pub struct RepetitionTable {
    pub table: [TTHash; REPETITION_SIZE],
    pub index: usize,
}

impl RepetitionTable {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn push_position(&mut self, pos: &Chess) {
        let hash = pos.zobrist_hash(shakmaty::EnPassantMode::Legal);
        self.push_hash(hash);
    }

    pub fn push_hash(&mut self, item: TTHash) {
        self.table[self.index] = item;
        self.index += 1;
    }

    pub fn set_ply(&mut self, ply: usize, item: TTHash) {
        self.table[self.index + ply] = item;
    }

    pub fn is_repetition(&self, ply: usize, hash: &TTHash, halfmoves: u32) -> bool {
        if halfmoves < 4 {
            return false;
        }

        let end = self.index + ply;
        let limit = (halfmoves as usize).min(end);

        let mut i = 4;
        while i <= limit {
            if *hash == self.table[end - i] {
                return true;
            }
            i += 2;
        }
        false
    }

    pub fn reset(&mut self) {
        self.index = 0;
    }
}

impl Default for RepetitionTable {
    fn default() -> Self {
        Self {
            table: [0.into(); REPETITION_SIZE],
            index: 0,
        }
    }
}
