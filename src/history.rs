use shakmaty::{Chess, Move, Position};

use crate::transposition_table::TTHash;

const HISTORY_SIZE: usize = 2048;

#[derive(Clone, Debug)]
pub struct MoveHistory {
    pub history: [TTHash; HISTORY_SIZE],
    pub moves: [Option<Move>; HISTORY_SIZE],
    pub index: usize,
}

impl MoveHistory {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn push_position(&mut self, pos: &Chess, mv: Option<Move>) {
        let hash = pos.zobrist_hash(shakmaty::EnPassantMode::Legal);
        self.push_hash(hash, mv);
    }

    pub fn push_hash(&mut self, item: TTHash, mv: Option<Move>) {
        self.history[self.index] = item;
        self.moves[self.index] = mv;
        self.index += 1;
    }

    pub fn pop(&mut self) {
        self.index -= 1;
        self.moves[self.index] = None;
    }

    pub fn reset(&mut self) {
        self.index = 0;
        self.moves.fill(None);
    }

    pub fn hash_exists(&self, item: &TTHash) -> bool {
        for i in 0..self.index {
            if *item == self.history[i] {
                return true;
            }
        }

        false
    }

    pub fn last_move(&self) -> Option<Move> {
        if self.index == 0 {
            None
        } else {
            self.moves[self.index - 1]
        }
    }
}

impl Default for MoveHistory {
    fn default() -> Self {
        Self {
            history: [0.into(); HISTORY_SIZE],
            moves: [None; HISTORY_SIZE],
            index: 0,
        }
    }
}
