use std::io::{self, Write};

use shakmaty::{Chess, Color, EnPassantMode, Move, Position, Role};

use super::GameRecord;

/// Stored when a move has no usable score
pub const NO_SCORE: i16 = i16::MAX;

/// - A 64-bit occupied-piece bitboard, which must not be zero (a zero occupancy denotes a reserved extensionrather than a `Game`).
/// - A 32-entry array of 4-bit pieces, where the `i`th entry corresponds to the `i`th least-significant set bit in the occupied-piece bitboard.
///   - The lower three bits of a piece corresponds to its type: pawn is 0, knight is 1, bishop is 2, rook is 3, queen is 4, king is 5.
///   - Castling rights are represented by setting the piece type of the relevant rook (e.g. in classical chess, the a1 rook for queenside, or the h1 rook for kingside) to 6 to represent an unmoved rook.
///   - A piece type of 7 is never valid.
///   - A piece has its most-significant bit clear if it is a white piece, and set if it is a black piece.
///   - Nonexistent piece entries may be left at zero.
/// - An 8-bit side-to-move and en-passant field.
///   - The lower seven bits represent the en-passant target square, or 64 if there is no such square.
///   - The most-significant bit is clear if white is to move, and set if black is to move.
/// - An 8-bit halfmove clock.
///   - (This field may be left at zero.)
/// - A 16-bit fullmove counter.
///   - (This field may be left at zero.)
/// - A `Score` for the position.
///   - (This field may be left at zero.)
/// - An 8-bit game-result field; a black win is 0, a draw is 1, a white win is 2. No other values are valid.
/// - An unused extra byte.
pub fn pack_board(pos: &Chess, wdl: u8) -> [u8; 32] {
    let board = pos.board();
    let occ = board.occupied();
    let castling = pos.castles().castling_rights(); // rook squares that can still castle

    let mut pieces = [0u8; 16];
    for (i, sq) in occ.into_iter().enumerate() {
        let piece = board.piece_at(sq).expect("occupied square has a piece");

        let mut kind: u8 = match piece.role {
            Role::Pawn => 0,
            Role::Knight => 1,
            Role::Bishop => 2,
            Role::Rook => 3,
            Role::Queen => 4,
            Role::King => 5,
        };
        // Castling rights are encoded by marking the rook as an "unmoved rook" (type 6)
        if piece.role == Role::Rook && castling.contains(sq) {
            kind = 6;
        }
        if piece.color == Color::Black {
            kind |= 0b1000;
        }

        pieces[i / 2] |= kind << (4 * (i & 1));
    }

    let ep: u8 = pos
        .ep_square(EnPassantMode::Legal)
        .map(|s| s as u8)
        .unwrap_or(64);
    let stm_ep = ((pos.turn() == Color::Black) as u8) << 7 | ep;

    let mut out = [0u8; 32];

    // 64 bit / 8 byte occupied bitboard
    out[0..8].copy_from_slice(&u64::from(occ).to_le_bytes());
    // 128 bit / 16 byte piece array
    out[8..24].copy_from_slice(&pieces);
    // 8 bit side to move + en passant
    out[24] = stm_ep;
    // 8 bit halfmove clock
    out[25] = pos.halfmoves().min(255) as u8;
    // 16 bit full move counter
    out[26..28].copy_from_slice(&(pos.fullmoves().get() as u16).to_le_bytes());
    // 16 bit position score? i'm not sure what that means but it can be left as 0
    out[28..30].copy_from_slice(&0i16.to_le_bytes());
    // 8bit game result
    out[30] = wdl;
    // unused byte
    out[31] = 0;
    out
}

/// A `Move` is a structure packed into a 16-bit integer:
/// - A 6-bit from square.
/// - A 6-bit to square.
///   - Castling moves are represented as king-takes-rook; a classical chess kingside castling would be represented as "E1 to H1". This is done to support Chess960.
/// - A 2-bit promotion piece, where knight is 0, bishop is 1, rook is 2, queen is 3.
///   - If not a promotion move, this field may be left at zero.
/// - A 2-bit move type, where en-passant captures are 1, castling moves are 2, promotions (including capture-promotions) are 3.
/// - Any move not in the stated categories has a move type of 0.
/// - A `Score` is a signed 16-bit integer representing a white-relative score for said `Move`.
fn encode_move(mv: &Move) -> u16 {
    match mv {
        Move::Normal {
            from,
            to,
            promotion,
            ..
        } => {
            let (promo, kind) = match promotion {
                Some(role) => ((*role as u16) - 2, 3u16), // Knight = 2 in shakmaty
                None => (0, 0),
            };
            (*from as u16) | ((*to as u16) << 6) | (promo << 12) | (kind << 14)
        }
        Move::EnPassant { from, to } => (*from as u16) | ((*to as u16) << 6) | (1 << 14),
        Move::Castle { king, rook } => (*king as u16) | ((*rook as u16) << 6) | (2 << 14),
        Move::Put { .. } => unreachable!("wattafak bro"),
    }
}

fn score_to_i16(score: i32) -> i16 {
    score.clamp(-31000, 31000) as i16
}

/// Writes one game: board, (move, score) pairs, then four zero bytes.
pub fn write_game<W: Write>(writer: &mut W, game: &GameRecord) -> io::Result<()> {
    writer.write_all(&pack_board(&game.start, game.result.wdl_u8()))?;
    for rm in &game.moves {
        writer.write_all(&encode_move(&rm.mv).to_le_bytes())?;
        writer.write_all(&rm.score.map(score_to_i16).unwrap_or(NO_SCORE).to_le_bytes())?;
    }
    writer.write_all(&[0u8; 4])
}
