#![allow(unused)]

use shakmaty::{Chess, Color, Piece, Position, Role};

#[derive(Clone, Debug)]
pub struct TrainingSample {
    pub features: [f64; 64], // Board representation
    pub target: f64,         // Sigmoid of evaluation score, (0, 1)
}

pub fn piece_to_f64(piece: Option<Piece>) -> f64 {
    match piece {
        None => 0.0,
        Some(p) => {
            let value = match p.role {
                Role::Pawn => 0.1,
                Role::Knight => 0.3,
                Role::Bishop => 0.4,
                Role::Rook => 0.5,
                Role::Queen => 0.9,
                Role::King => 1.0,
            };

            match p.color {
                Color::White => value,
                Color::Black => -value,
            }
        }
    }
}

pub fn parse_board_to_features(board: &Chess) -> [f64; 64] {
    let mut features = [0.0; 64];
    for (square, piece) in board.board().iter() {
        features[square as usize] = piece_to_f64(Some(piece));
    }
    features
}

pub fn win_probability(centipawns: f64) -> f64 {
    1.0 / (1.0 + 10f64.powf(-centipawns / 400.0))
}

pub fn centipawns(probability: f64) -> f64 {
    -400.0 * (1.0 / probability - 1.0).log10()
}
