use shakmaty::{Chess, Color, File, Move, Piece, Position, Role, Square};

const HIDDEN_SIZE: usize = 32;
const SCALE: i32 = 400;
const QA: i16 = 255;
const QB: i16 = 64;

static NNUE: Network =
    unsafe { std::mem::transmute(*include_bytes!("jarona.bin")) };

#[inline]
/// Square Clipped ReLU - Activation Function.
/// Note that this takes the i16s in the accumulator to i32s.
/// Range is 0.0 .. 1.0 (in other words, 0 to QA*QA quantized).
fn screlu(x: i16) -> i32 {
    let y = i32::from(x).clamp(0, i32::from(QA));
    y * y
}

/// This is the quantised format that bullet outputs.
#[repr(C)]
pub struct Network {
    /// Column-Major `HIDDEN_SIZE x 768` matrix.
    /// Values have quantization of QA.
    feature_weights: [Accumulator; 768],
    /// Vector with dimension `HIDDEN_SIZE`.
    /// Values have quantization of QA.
    feature_bias: Accumulator,
    /// Column-Major `1 x (2 * HIDDEN_SIZE)`matrix, 
    /// we use it like this to make the code nicer in `Network::evaluate`.
    /// Values have quantization of QB.
    output_weights: [i16; 2 * HIDDEN_SIZE],
    /// Scalar output bias.
    /// Value has quantization of QA * QB.
    output_bias: i16,
}

impl Network {
    /// Calculates the output of the network, starting from the already
    /// calculated hidden layer (done efficiently during makemoves).
    pub fn evaluate(&self, us: &Accumulator, them: &Accumulator) -> i32 {
        // Initialise output.
        let mut output = 0;

        // Feed forward for both Side-To-Move and Not-Side-To-Move accumulators respectively
        for (&input, &weight) in us.vals.iter().zip(&self.output_weights[..HIDDEN_SIZE]) {
            output += screlu(input) * i32::from(weight);
        }
        for (&input, &weight) in them.vals.iter().zip(&self.output_weights[HIDDEN_SIZE..]) {
            output += screlu(input) * i32::from(weight);
        }

        output /= i32::from(QA);  // Reduce quantization from QA * QA * QB to QA * QB.
        output += i32::from(self.output_bias);  // Add bias.
        output *= SCALE;  // Apply eval scale.
        output /= i32::from(QA) * i32::from(QB);  // Remove quantisation altogether.

        output
    }
}

/// A column of the feature-weights matrix.
/// Note the `align(64)`.
#[derive(Clone, Copy)]
#[repr(C, align(64))]
pub struct Accumulator {
    vals: [i16; HIDDEN_SIZE],
}

// Index into the 768 inputs, as seen from `perspective`.
fn feature_index(perspective: Color, piece: Piece, sq: Square) -> usize {
    let piece_type = piece.role as usize - 1; // shakmaty: Pawn = 1 .. King = 6
    // Our pieces occupy 0..384, enemy pieces 384..768
    let side = if piece.color == perspective { 0 } else { 384 };
    let square = match perspective {
        Color::White => sq as usize,
        Color::Black => sq as usize ^ 56, // flip vertically for Black
    };
    side + 64 * piece_type + square
}

impl Accumulator {
    /// Initialised with bias so we can just efficiently
    /// operate on it afterwards.
    pub fn new(net: &Network) -> Self {
        net.feature_bias
    }

    pub fn default() -> Self {
        NNUE.feature_bias
    }

    /// Add a feature to an accumulator.
    pub fn add_feature(&mut self, feature_idx: usize, net: &Network) {
        for (i, d) in self.vals.iter_mut().zip(&net.feature_weights[feature_idx].vals) {
            *i += *d
        }
    }

    /// Remove a feature from an accumulator.
    pub fn remove_feature(&mut self, feature_idx: usize, net: &Network) {
        for (i, d) in self.vals.iter_mut().zip(&net.feature_weights[feature_idx].vals) {
            *i -= *d
        }
    }
}

fn add_piece(acc: &mut [Accumulator; 2], piece: Piece, sq: Square) {
    acc[0].add_feature(feature_index(Color::White, piece, sq), &NNUE);
    acc[1].add_feature(feature_index(Color::Black, piece, sq), &NNUE);
}

fn remove_piece(acc: &mut [Accumulator; 2], piece: Piece, sq: Square) {
    acc[0].remove_feature(feature_index(Color::White, piece, sq), &NNUE);
    acc[1].remove_feature(feature_index(Color::Black, piece, sq), &NNUE);
}

// Updates both accumulators for `mov`, played by `mover` (the side to move BEFORE the move).
pub fn apply_move(acc: &mut [Accumulator; 2], mov: &Move, mover: Color) {
    match *mov {
        Move::Normal { role, from, capture, to, promotion } => {
            remove_piece(acc, Piece { color: mover, role }, from);
            if let Some(captured) = capture {
                remove_piece(acc, Piece { color: !mover, role: captured }, to);
            }
            add_piece(acc, Piece { color: mover, role: promotion.unwrap_or(role) }, to);
        }
        Move::EnPassant { from, to } => {
            // the captured pawn sits on the destination file, on the mover's starting rank
            let captured_sq = Square::from_coords(to.file(), from.rank());
            remove_piece(acc, Piece { color: mover, role: Role::Pawn }, from);
            remove_piece(acc, Piece { color: !mover, role: Role::Pawn }, captured_sq);
            add_piece(acc, Piece { color: mover, role: Role::Pawn }, to);
        }
        Move::Castle { king, rook } => {
            // shakmaty stores castling as king square -> rook square
            let rank = king.rank();
            let (king_to, rook_to) = if (rook as u32) > (king as u32) {
                (Square::from_coords(File::G, rank), Square::from_coords(File::F, rank))
            } else {
                (Square::from_coords(File::C, rank), Square::from_coords(File::D, rank))
            };
            remove_piece(acc, Piece { color: mover, role: Role::King }, king);
            remove_piece(acc, Piece { color: mover, role: Role::Rook }, rook);
            add_piece(acc, Piece { color: mover, role: Role::King }, king_to);
            add_piece(acc, Piece { color: mover, role: Role::Rook }, rook_to);
        }
        Move::Put { .. } => unreachable!("Poor truck doesn't understand this move..."),
    }
}

pub fn evaluate(acc: &[Accumulator; 2], stm: Color) -> i32 {
    let (us, them) = match stm {
        Color::White => (&acc[0], &acc[1]),
        Color::Black => (&acc[1], &acc[0]),
    };
    NNUE.evaluate(us, them)
}