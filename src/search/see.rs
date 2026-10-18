use shakmaty::Color::{self, Black, White};
use shakmaty::{Bitboard, Board, Chess, Move, Position, Rank, Role, Square, attacks};
use std::cmp::max;

// The default shakmaty attacks_to is based on color and it would be inefficient to call it twice
#[inline]
fn attackers_to(board: &Board, square: Square, occupancies: Bitboard) -> Bitboard {
    let white_board = board.by_color(White);
    let black_board = board.by_color(Black);

    let diagonals = board.bishops() | board.queens();
    let orthogonals = board.rooks() | board.queens();

    let white_pawn_attacks = attacks::pawn_attacks(Black, square) & (board.pawns() & white_board);
    let black_pawn_attacks = attacks::pawn_attacks(White, square) & (board.pawns() & black_board);
    let knight_attacks = attacks::knight_attacks(square) & board.knights();
    let diagonal_attacks = attacks::bishop_attacks(square, occupancies) & diagonals;
    let orthogonal_attacks = attacks::rook_attacks(square, occupancies) & orthogonals;
    let king_attacks = attacks::king_attacks(square) & board.kings();

    white_pawn_attacks
        | black_pawn_attacks
        | knight_attacks
        | diagonal_attacks
        | orthogonal_attacks
        | king_attacks
}

#[inline]
fn consider_xrays(board: &Board, occupied: Bitboard, to_sq: Square) -> Bitboard {
    let bishop_sliders = board.bishops() | board.queens();
    let rook_sliders = board.rooks() | board.queens();

    ((attacks::bishop_attacks(to_sq, occupied) & bishop_sliders)
        | (attacks::rook_attacks(to_sq, occupied) & rook_sliders))
        & occupied
}

#[inline]
fn piece_value(role: Role) -> i32 {
    #[cfg(test)]
    return [100, 300, 300, 500, 900, 0][role as usize - 1];

    #[cfg(not(test))]
    return super::eval::PIECE_VALUES_MIDDLEGAME[role as usize - 1];
}

#[inline]
fn get_least_valuable_piece(
    board: &Board,
    attadef: Bitboard,
    color: Color,
) -> Option<(Square, Role)> {
    let attackers = attadef & board.by_color(color);
    if attackers.is_empty() {
        return None;
    }

    let (role, subset) = if !(attackers & board.pawns()).is_empty() {
        (Role::Pawn, board.pawns())
    } else if !(attackers & board.knights()).is_empty() {
        (Role::Knight, board.knights())
    } else if !(attackers & board.bishops()).is_empty() {
        (Role::Bishop, board.bishops())
    } else if !(attackers & board.rooks()).is_empty() {
        (Role::Rook, board.rooks())
    } else if !(attackers & board.queens()).is_empty() {
        (Role::Queen, board.queens())
    } else if !(attackers & board.kings()).is_empty() {
        (Role::King, board.kings())
    } else {
        return None;
    };

    let sq = (attackers & subset).first()?;
    Some((sq, role))
}

pub fn see(position: &Chess, mov: &Move) -> i32 {
    let mut gain = [0; 32];
    let mut d = 0;

    let board = position.board();
    let from_sq = mov.from().unwrap();
    let to_sq = mov.to();
    let target = mov.capture();
    let role = mov.role();

    // May xray pieces: pawns, bishops, rooks, queens
    let may_xray = board.pawns() | board.bishops() | board.rooks() | board.queens();
    let mut from_set = Bitboard::from_square(from_sq);
    let mut occupied = board.occupied();

    if mov.is_en_passant() {
        let ep_square = Square::from_coords(to_sq.file(), from_sq.rank());
        occupied ^= Bitboard::from_square(ep_square);
    }

    let mut attadef = attackers_to(board, to_sq, occupied);

    gain[0] = target.map_or(0, piece_value)
        + mov.promotion().map_or(0, |promotion| {
            piece_value(promotion) - piece_value(Role::Pawn)
        });

    let mut current_piece = mov.promotion().unwrap_or(role);
    let mut stm = position.turn().other();
    let promo_square = matches!(to_sq.rank(), Rank::First | Rank::Eighth);

    loop {
        d += 1;
        gain[d] = piece_value(current_piece) - gain[d - 1];

        attadef &= !from_set;
        occupied ^= from_set;

        if (from_set & may_xray).any() {
            attadef |= consider_xrays(board, occupied, to_sq);
        }

        match get_least_valuable_piece(board, attadef, stm) {
            Some((sq, piece)) => {
                if piece == Role::King && (attadef & board.by_color(stm.other())).any() {
                    break;
                }

                from_set = Bitboard::from_square(sq);
                if piece == Role::Pawn && promo_square {
                    // Q - P, and a queen now sits on the square
                    gain[d] += piece_value(Role::Queen) - piece_value(Role::Pawn);
                    current_piece = Role::Queen;
                } else {
                    current_piece = piece;
                }
            }
            None => break,
        }

        stm = stm.other();
    }

    while d > 1 {
        d -= 1;
        gain[d - 1] = -max(-gain[d - 1], gain[d]);
    }

    gain[0]
}
