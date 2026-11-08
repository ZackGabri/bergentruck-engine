#![allow(unused)]

use shakmaty::{Chess, Color, Piece, Position, Role};

use super::math::*;
use super::math::*;
use crate::pgn_reader::GameRecord;
use crate::pgn_reader::GameResult;

#[derive(Clone, Debug)]
pub struct TrainingSample {
    pub features: [f64; 64], // Board representation
    pub target: f64,         // Sigmoid of evaluation score, (0, 1)
}

const WDL: f64 = 0.5;

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

pub fn games_to_samples(games: Vec<GameRecord>) -> Vec<TrainingSample> {
    let mut samples = Vec::new();

    for (i, game) in games.into_iter().enumerate() {
        let raw_result = game.result;

        for (pos, rm) in game.replay() {
            let Some(raw_score) = rm.score else { continue };

            // Filter mate scores
            if raw_score.abs() >= 90000 {
                continue;
            }

            let score = if pos.turn() == Color::White {
                raw_score
            } else {
                -raw_score
            };

            let result = match raw_result {
                GameResult::Draw => 0.5,
                GameResult::WhiteWin => (pos.turn() == Color::White) as u8 as f64,
                GameResult::BlackWin => (pos.turn() == Color::Black) as u8 as f64,
            };

            let target = get_target(WDL, result, score as f64);

            samples.push(TrainingSample {
                features: parse_board_to_features(&pos),
                target: target,
            });

            /*
            if i < 10 {
                println!("WDL: {WDL}, result: {result}, score: {score}, target: {target}");
            }
            */
        }
    }

    samples
}
