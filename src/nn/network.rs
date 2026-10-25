use shakmaty::{Chess, Color, File, Move, Piece, Position, Role, Square};

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
use std::arch::x86_64::*;

const HIDDEN_SIZE: usize = 256;
const SCALE: i32 = 400;
const QA: i16 = 255;

// WARNING: QB = 64 ensures |w| ≤ 128
// changing it would put w above that bound and potentially break stuff
const QB: i16 = 64;

static NNUE: Network = unsafe { std::mem::transmute(*include_bytes!("networks/Hathy_I.bin")) };

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

/// Horizontal sum of the eight i32 lanes.
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[inline]
fn hsum_epi32(v: __m256i) -> i32 {
    unsafe {
        let lo = _mm256_castsi256_si128(v);
        let hi = _mm256_extracti128_si256::<1>(v);
        let s = _mm_add_epi32(lo, hi); // 8 -> 4 lanes
        let s = _mm_add_epi32(s, _mm_shuffle_epi32::<0b01_00_11_10>(s)); // 4 -> 2
        let s = _mm_add_epi32(s, _mm_shuffle_epi32::<0b10_11_00_01>(s)); // 2 -> 1
        _mm_cvtsi128_si32(s)
    }
}

impl Network {
    /// Dot product of SCReLU(acc) with the output weights, for both perspectives.
    /// Returns the raw sum, quantised at QA * QA * QB, before the final rescaling.
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    #[inline]
    fn dot(&self, us: &Accumulator, them: &Accumulator) -> i32 {
        const CHUNKS: usize = HIDDEN_SIZE / 16; // 16 x i16 per 256-bit register

        unsafe {
            let zero = _mm256_setzero_si256();
            let qa = _mm256_set1_epi16(QA);
            let mut sum = zero;

            for (acc, w) in [
                (us, &self.output_weights[..HIDDEN_SIZE]),
                (them, &self.output_weights[HIDDEN_SIZE..]),
            ] {
                let a = acc.vals.as_ptr() as *const __m256i;
                let w = w.as_ptr() as *const __m256i;

                for i in 0..CHUNKS {
                    // Accumulator is align(64), so aligned loads are fine.
                    let x = _mm256_load_si256(a.add(i));
                    let wt = _mm256_loadu_si256(w.add(i));

                    // Clipped ReLU: clamp to [0, QA]
                    let v = _mm256_min_epi16(_mm256_max_epi16(x, zero), qa);

                    // (v * w) stays in i16, requires |w| <= 128
                    let vw = _mm256_mullo_epi16(v, wt);

                    // (v * w) * v, with adjacent i16 pairs summed into i32 lanes
                    sum = _mm256_add_epi32(sum, _mm256_madd_epi16(vw, v));
                }
            }
            hsum_epi32(sum)
        }
    }

    // Scalar fallback for non-AVX2 builds.
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
    #[inline]
    fn dot(&self, us: &Accumulator, them: &Accumulator) -> i32 {
        #[inline]
        /// Square Clipped ReLU - Activation Function.
        /// Note that this takes the i16s in the accumulator to i32s.
        /// Range is 0.0 .. 1.0 (in other words, 0 to QA*QA quantized).
        fn screlu(x: i16) -> i32 {
            let y = i32::from(x).clamp(0, i32::from(QA));
            y * y
        }

        let mut out = 0;
        for (acc, w) in [
            (us, &self.output_weights[..HIDDEN_SIZE]),
            (them, &self.output_weights[HIDDEN_SIZE..]),
        ] {
            for (&x, &w) in acc.vals.iter().zip(w) {
                out += screlu(x) * i32::from(w);
            }
        }
        out
    }

    pub fn evaluate(&self, us: &Accumulator, them: &Accumulator) -> i32 {
        let mut output = self.dot(us, them);
        output /= i32::from(QA);
        output += i32::from(self.output_bias);
        output *= SCALE;
        output / (i32::from(QA) * i32::from(QB))
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
    pub fn default() -> Self {
        NNUE.feature_bias
    }

    /// Add a feature to an accumulator.
    pub fn add_feature(&mut self, feature_idx: usize, net: &Network) {
        for (i, d) in self
            .vals
            .iter_mut()
            .zip(&net.feature_weights[feature_idx].vals)
        {
            *i += *d
        }
    }

    /// Remove a feature from an accumulator.
    pub fn remove_feature(&mut self, feature_idx: usize, net: &Network) {
        for (i, d) in self
            .vals
            .iter_mut()
            .zip(&net.feature_weights[feature_idx].vals)
        {
            *i -= *d
        }
    }
}

/// Builds both perspective accumulators from scratch. Index 0 = White's view, 1 = Black's.
/// Use at the root of a search.
pub fn refresh(position: &Chess) -> [Accumulator; 2] {
    let mut acc = [Accumulator::default(); 2];
    for (sq, piece) in position.board().iter() {
        add_piece(&mut acc, piece, sq);
    }
    acc
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
        Move::Normal {
            role,
            from,
            capture,
            to,
            promotion,
        } => {
            remove_piece(acc, Piece { color: mover, role }, from);
            if let Some(captured) = capture {
                remove_piece(
                    acc,
                    Piece {
                        color: !mover,
                        role: captured,
                    },
                    to,
                );
            }
            add_piece(
                acc,
                Piece {
                    color: mover,
                    role: promotion.unwrap_or(role),
                },
                to,
            );
        }
        Move::EnPassant { from, to } => {
            // the captured pawn sits on the destination file, on the mover's starting rank
            let captured_sq = Square::from_coords(to.file(), from.rank());
            remove_piece(
                acc,
                Piece {
                    color: mover,
                    role: Role::Pawn,
                },
                from,
            );
            remove_piece(
                acc,
                Piece {
                    color: !mover,
                    role: Role::Pawn,
                },
                captured_sq,
            );
            add_piece(
                acc,
                Piece {
                    color: mover,
                    role: Role::Pawn,
                },
                to,
            );
        }
        Move::Castle { king, rook } => {
            // shakmaty stores castling as king square -> rook square
            let rank = king.rank();
            let (king_to, rook_to) = if (rook as u32) > (king as u32) {
                (
                    Square::from_coords(File::G, rank),
                    Square::from_coords(File::F, rank),
                )
            } else {
                (
                    Square::from_coords(File::C, rank),
                    Square::from_coords(File::D, rank),
                )
            };
            remove_piece(
                acc,
                Piece {
                    color: mover,
                    role: Role::King,
                },
                king,
            );
            remove_piece(
                acc,
                Piece {
                    color: mover,
                    role: Role::Rook,
                },
                rook,
            );
            add_piece(
                acc,
                Piece {
                    color: mover,
                    role: Role::King,
                },
                king_to,
            );
            add_piece(
                acc,
                Piece {
                    color: mover,
                    role: Role::Rook,
                },
                rook_to,
            );
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
