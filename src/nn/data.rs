use shakmaty::{Chess, Color, Position, Role, Piece};

pub struct TrainExample {
    pub features: [f32; 64], // Board representation
    pub target: f32,         // Sigmoid of evaluation score, (0, 1)
}

pub fn piece_to_f32(piece: Option<Piece>) -> f32 {
    match piece {
        None => 0.0,
        Some(p) => {
            let value = match p.role {
                Role::Pawn   => 0.1,
                Role::Knight => 0.3,
                Role::Bishop => 0.4,
                Role::Rook   => 0.5,
                Role::Queen  => 0.9,
                Role::King   => 1.0,
            };

            match p.color {
                Color::White => value,
                Color::Black => -value,
            }
        }
    }
}

pub fn parse_board_to_features(board: &Chess) -> [f32; 64] {
    let mut features = [0.0; 64];
    for (square, piece) in board.board().iter() {
        features[square as usize] = piece_to_f32(Some(piece));
    }
    features
}